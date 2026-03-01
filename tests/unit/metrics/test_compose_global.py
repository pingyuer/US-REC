"""Self-check tests for compose_global_from_local / local_from_global.

Three mandatory sanity checks
------------------------------
1. Identity test:     all-I locals  → all-I globals  → GPE/LPE ≈ 0
2. Constant-translation test: 1mm x-shift per frame → global[i] = i*1mm
3. Round-trip test:   arbitrary global → derive local → re-compose → match

These can be run standalone (``python tests/test_compose_global.py``) or
via pytest (``pytest tests/test_compose_global.py -v``).
"""

from __future__ import annotations

import pytest
pytestmark = pytest.mark.unit

import sys
import torch
import math

# Ensure repo root is importable when running standalone
sys.path.insert(0, ".")

# Import directly from the module to avoid triggering heavy trainers/__init__.py
from trainers.metrics.tusrec import (
    compose_global_from_local,
    local_from_global,
)


# --------------------------------------------------------------------------
# 1. Identity test
# --------------------------------------------------------------------------

def test_identity():
    """All-identity locals → global should be all-identity → GPE/LPE = 0."""
    T = 20
    local_T = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)  # (T, 4, 4)
    global_T = compose_global_from_local(local_T, convention="prev_from_curr")

    assert global_T.shape == (T, 4, 4), f"shape mismatch: {global_T.shape}"
    eye = torch.eye(4)
    max_err = float(torch.abs(global_T - eye.unsqueeze(0)).max().item())
    assert max_err < 1e-6, f"Identity test FAILED: max error = {max_err}"
    print(f"[PASS] Identity test: max_err = {max_err:.2e}")


# --------------------------------------------------------------------------
# 2. Constant translation test
# --------------------------------------------------------------------------

def test_constant_translation():
    """Every frame shifts +1mm in x → global[i] should have tx = i mm."""
    T = 50
    local_T = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)  # (T, 4, 4)
    for i in range(1, T):
        local_T[i, 0, 3] = 1.0  # +1mm x shift per frame

    global_T = compose_global_from_local(local_T, convention="prev_from_curr")

    # global[i] should be identity with tx = i
    max_err = 0.0
    for i in range(T):
        expected_tx = float(i)
        actual_tx = float(global_T[i, 0, 3].item())
        err = abs(actual_tx - expected_tx)
        if err > max_err:
            max_err = err
        # y, z translations should be 0
        assert abs(float(global_T[i, 1, 3].item())) < 1e-6, f"ty != 0 at frame {i}"
        assert abs(float(global_T[i, 2, 3].item())) < 1e-6, f"tz != 0 at frame {i}"

    assert max_err < 1e-5, f"Constant translation FAILED: max tx error = {max_err}"
    print(f"[PASS] Constant translation test: max_err = {max_err:.2e}")


# --------------------------------------------------------------------------
# 3. Round-trip test
# --------------------------------------------------------------------------

def test_round_trip():
    """global → local → re-compose → should match original global (< 1e-4)."""
    T = 30
    torch.manual_seed(42)

    # Build a realistic global trajectory: small rotations + translations.
    global_T_original = torch.zeros(T, 4, 4)
    global_T_original[0] = torch.eye(4)
    for i in range(1, T):
        # Small random rotation (< 5°) and translation (< 2mm)
        angle = (torch.rand(1).item() - 0.5) * math.radians(5)
        c, s = math.cos(angle), math.sin(angle)
        R = torch.eye(4)
        R[0, 0] = c;  R[0, 1] = -s
        R[1, 0] = s;  R[1, 1] = c
        R[0, 3] = (torch.rand(1).item() - 0.5) * 2.0
        R[1, 3] = (torch.rand(1).item() - 0.5) * 2.0
        R[2, 3] = (torch.rand(1).item() - 0.5) * 2.0
        global_T_original[i] = global_T_original[i - 1] @ R

    # Derive local from global.
    local_T = local_from_global(global_T_original)

    # Re-compose global from local.
    global_T_recon = compose_global_from_local(local_T, convention="prev_from_curr")

    max_err = float(torch.abs(global_T_recon - global_T_original).max().item())
    assert max_err < 1e-4, f"Round-trip FAILED: max error = {max_err}"
    print(f"[PASS] Round-trip test: max_err = {max_err:.2e}")


# --------------------------------------------------------------------------
# 4. curr_from_prev convention test
# --------------------------------------------------------------------------

def test_curr_from_prev_convention():
    """Test that curr_from_prev inputs are correctly inverted."""
    T = 10
    local_pfc = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
    for i in range(1, T):
        local_pfc[i, 0, 3] = 1.0  # +1mm x

    # curr_from_prev is the inverse of prev_from_curr
    local_cfp = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
    local_cfp[1:] = torch.linalg.inv(local_pfc[1:])

    g1 = compose_global_from_local(local_pfc, convention="prev_from_curr")
    g2 = compose_global_from_local(local_cfp, convention="curr_from_prev")

    max_err = float(torch.abs(g1 - g2).max().item())
    assert max_err < 1e-5, f"Convention test FAILED: {max_err}"
    print(f"[PASS] curr_from_prev convention test: max_err = {max_err:.2e}")


# --------------------------------------------------------------------------
# 5. Batched test
# --------------------------------------------------------------------------

def test_batched():
    """Ensure (B, T, 4, 4) batched input gives same result as unbatched."""
    T, B = 15, 3
    torch.manual_seed(99)
    local_T = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
    for i in range(1, T):
        local_T[i, 0, 3] = float(i) * 0.5
        local_T[i, 1, 3] = float(i) * -0.3

    single = compose_global_from_local(local_T, convention="prev_from_curr")
    batched_input = local_T.unsqueeze(0).repeat(B, 1, 1, 1)
    batched_result = compose_global_from_local(batched_input, convention="prev_from_curr")

    for b in range(B):
        err = float(torch.abs(batched_result[b] - single).max().item())
        assert err < 1e-6, f"Batched test FAILED at b={b}: {err}"
    print(f"[PASS] Batched test")


# --------------------------------------------------------------------------
# Run all
# --------------------------------------------------------------------------

def run_all():
    test_identity()
    test_constant_translation()
    test_round_trip()
    test_curr_from_prev_convention()
    test_batched()
    print("\n=== All compose_global_from_local self-checks PASSED ===")


if __name__ == "__main__":
    run_all()
