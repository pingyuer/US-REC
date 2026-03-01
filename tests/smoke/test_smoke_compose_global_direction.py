"""Smoke test 3: compose_global / local_from_global direction checks.

Three mandatory sanity checks plus round-trip verification.  Uses the
**authoritative** implementation from ``metrics.compose``.
"""

import math
import pytest
import torch

from metrics.compose import compose_global_from_local, local_from_global


@pytest.mark.smoke
class TestSmokeComposeGlobalDirection:

    # ---- 1. Identity ---------------------------------------------------
    def test_identity(self):
        """All-I locals -> all-I globals."""
        T = 20
        local_T = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
        global_T = compose_global_from_local(local_T)
        eye = torch.eye(4).unsqueeze(0).expand_as(global_T)
        assert float(torch.abs(global_T - eye).max()) < 1e-6

    # ---- 2. Constant translation ----------------------------------------
    def test_constant_translation(self):
        """Each step +1mm x -> global[i].tx == i."""
        T = 50
        local_T = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
        for i in range(1, T):
            local_T[i, 0, 3] = 1.0  # +1mm x per frame

        global_T = compose_global_from_local(local_T)

        for i in range(T):
            assert abs(float(global_T[i, 0, 3]) - float(i)) < 1e-5
            assert abs(float(global_T[i, 1, 3])) < 1e-6
            assert abs(float(global_T[i, 2, 3])) < 1e-6

    # ---- 3. Round-trip --------------------------------------------------
    def test_round_trip(self):
        """global -> local -> re-compose -> error < 1e-4."""
        T = 30
        torch.manual_seed(42)
        global_T = torch.zeros(T, 4, 4)
        global_T[0] = torch.eye(4)
        for i in range(1, T):
            angle = (torch.rand(1).item() - 0.5) * math.radians(5)
            c, s = math.cos(angle), math.sin(angle)
            R = torch.eye(4)
            R[0, 0] = c; R[0, 1] = -s
            R[1, 0] = s; R[1, 1] = c
            R[0, 3] = (torch.rand(1).item() - 0.5) * 2.0
            R[1, 3] = (torch.rand(1).item() - 0.5) * 2.0
            R[2, 3] = (torch.rand(1).item() - 0.5) * 2.0
            global_T[i] = global_T[i - 1] @ R

        local_T = local_from_global(global_T)
        recon = compose_global_from_local(local_T)
        assert float(torch.abs(recon - global_T).max()) < 1e-4

    # ---- 4. Batched consistency -----------------------------------------
    def test_batched(self):
        T, B = 15, 3
        local_T = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
        for i in range(1, T):
            local_T[i, 0, 3] = float(i) * 0.5
        single = compose_global_from_local(local_T)
        batched = compose_global_from_local(local_T.unsqueeze(0).expand(B, -1, -1, -1))
        for b in range(B):
            assert float(torch.abs(batched[b] - single).max()) < 1e-6

    # ---- 5. curr_from_prev convention -----------------------------------
    def test_curr_from_prev(self):
        T = 10
        local_pfc = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
        for i in range(1, T):
            local_pfc[i, 0, 3] = 1.0
        local_cfp = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
        local_cfp[1:] = torch.linalg.inv(local_pfc[1:])

        g1 = compose_global_from_local(local_pfc, convention="prev_from_curr")
        g2 = compose_global_from_local(local_cfp, convention="curr_from_prev")
        assert float(torch.abs(g1 - g2).max()) < 1e-5
