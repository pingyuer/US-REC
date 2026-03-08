"""Evaluation pipeline diagnostics — shape/dtype/direction/GT-baseline checks.

Run automatically inside KRootDualTrainer.evaluate() when diagnostics_level > 0.
Can also be called standalone::

    from eval.diagnostics import run_pipeline_diagnostics
    report = run_pipeline_diagnostics(
        fused_global, short_global, long_global, gt_global,
        anchor_indices, tform_calib, scan_id="052/RH_Per_C_DtP",
        diagnostics_level=2,
    )
    print(report["summary"])

Levels
------
0  silence (no diagnostics)
1  sanity checks only (shapes, dtypes, NaN/Inf, trajectory divergence)
2  level 1 + GT-baseline score (one full tusrec pass with GT as prediction)
3  level 2 + DDF correspondence check (expensive, requires calib)
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional, Tuple

import torch

from metrics.compose import compose_global_from_local, local_from_global

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline_diagnostics(
    fused_global: torch.Tensor,
    short_global: torch.Tensor,
    long_global: torch.Tensor,
    gt_global: torch.Tensor,
    anchor_indices: torch.Tensor,
    tform_calib: Optional[torch.Tensor] = None,
    *,
    scan_id: str = "unknown",
    diagnostics_level: int = 1,
    frames: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Run all diagnostic checks and return a structured report dict.

    Parameters
    ----------
    fused_global, short_global : (T, 4, 4) — dense global transforms
    long_global : (M, 4, 4) — sparse anchor global transforms
    gt_global : (T, 4, 4)
    anchor_indices : (M,) — frame indices of long anchors
    tform_calib : (4, 4) — pixel→mm calibration transform
    scan_id : str — for log messages
    diagnostics_level : 0–3
    frames : (T, H, W) — only needed for level 3

    Returns
    -------
    report : dict with keys
        "ok" : bool — True when all checks pass
        "errors" : list[str] — failing checks
        "warnings" : list[str]
        "trajectory_divergence" : dict
        "gt_baseline" : dict | None  (level >= 2)
        "summary" : str  — printable block
    """
    if diagnostics_level <= 0:
        return {"ok": True, "errors": [], "warnings": [], "summary": ""}

    errors: list[str] = []
    warns: list[str] = []
    report: Dict[str, Any] = {
        "scan_id": scan_id,
        "errors": errors,
        "warnings": warns,
        "trajectory_divergence": {},
        "gt_baseline": None,
    }

    T = gt_global.shape[0]
    M = anchor_indices.shape[0]
    device = gt_global.device

    # ------------------------------------------------------------------
    # A. Shape / dtype / finiteness checks
    # ------------------------------------------------------------------
    def _check_tensor(name: str, t: torch.Tensor, expected_shape: Tuple):
        if tuple(t.shape) != expected_shape:
            errors.append(
                f"[{scan_id}] {name}: shape={tuple(t.shape)} expected={expected_shape}"
            )
        if t.dtype not in (torch.float32, torch.float64):
            errors.append(f"[{scan_id}] {name}: dtype={t.dtype} (must be float32/float64)")
        n_nan = int(t.isnan().sum().item())
        n_inf = int(t.isinf().sum().item())
        if n_nan:
            errors.append(f"[{scan_id}] {name}: {n_nan} NaN values")
        if n_inf:
            errors.append(f"[{scan_id}] {name}: {n_inf} Inf values")
        t_abs_max = float(t.abs().max().item())
        if t_abs_max > 1e4:
            warns.append(
                f"[{scan_id}] {name}: max abs value={t_abs_max:.1f} "
                "(>1e4 — check units, expected mm)"
            )

    _check_tensor("fused_global", fused_global, (T, 4, 4))
    _check_tensor("short_global", short_global, (T, 4, 4))
    _check_tensor("long_global", long_global, (M, 4, 4))
    _check_tensor("gt_global", gt_global, (T, 4, 4))

    # Check global[0] ≈ I
    for name, g in [("fused_global", fused_global), ("short_global", short_global), ("gt_global", gt_global)]:
        eye_err = float((g[0] - torch.eye(4, device=device, dtype=g.dtype)).abs().max().item())
        if eye_err > 1e-3:
            warns.append(
                f"[{scan_id}] {name}[0] deviates from I by {eye_err:.6f} "
                "(global[0] should be identity)"
            )

    # Check anchor_indices coverage
    if M > 0:
        if int(anchor_indices[0].item()) != 0:
            warns.append(
                f"[{scan_id}] anchor_indices[0]={int(anchor_indices[0].item())} != 0; "
                "first anchor should be frame 0"
            )
        if int(anchor_indices[-1].item()) >= T:
            errors.append(
                f"[{scan_id}] anchor_indices[-1]={int(anchor_indices[-1].item())} >= T={T}"
            )

    # ------------------------------------------------------------------
    # B. Trajectory divergence: fused vs short vs long-interpolated
    # ------------------------------------------------------------------
    long_full = _expand_anchor_global(long_global, anchor_indices, T, device, gt_global.dtype)

    td: Dict[str, float] = {}
    for n_a, a in [("fused", fused_global), ("short", short_global), ("long_full", long_full)]:
        for n_b, b in [("fused", fused_global), ("short", short_global), ("long_full", long_full)]:
            if n_a >= n_b:  # skip self and duplicates (alphabetic)
                continue
            key = f"{n_a}_vs_{n_b}"
            t_diff = (a[:, :3, 3] - b[:, :3, 3]).norm(dim=-1)
            td[f"{key}_t_mean_mm"] = float(t_diff.mean().item())
            td[f"{key}_t_max_mm"] = float(t_diff.max().item())
            # Rotation angle difference
            R_a = a[:, :3, :3]
            R_b = b[:, :3, :3]
            R_rel = torch.matmul(R_a, R_b.transpose(-2, -1))
            tr = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)
            cos_a = ((tr - 1.0) / 2.0).clamp(-1 + 1e-7, 1 - 1e-7)
            angles_rad = torch.acos(cos_a)
            angles_deg = angles_rad * (180.0 / math.pi)
            td[f"{key}_R_mean_deg"] = float(angles_deg.mean().item())
            td[f"{key}_R_max_deg"] = float(angles_deg.max().item())

    # Warn if fused ≈ short (stitching is not correcting anything)
    fused_vs_short_t = td.get("fused_vs_short_t_mean_mm", 0.0)
    if fused_vs_short_t < 1.0:
        warns.append(
            f"[{scan_id}] fused_vs_short mean_t={fused_vs_short_t:.3f} mm — "
            "fused ≈ short. Long correction has no effect. "
            "Likely cause: model outputs near-identity (not converged) "
            "OR long_global[0]=I makes first interval stitch = short."
        )

    fused_vs_long_t = td.get("fused_vs_long_full_t_mean_mm", 0.0)
    if fused_vs_long_t < 1.0:
        warns.append(
            f"[{scan_id}] fused_vs_long mean_t={fused_vs_long_t:.3f} mm — "
            "fused ≈ long. Short refinement has no effect."
        )

    # ALL THREE identical: only if BOTH fused≈short AND fused≈long
    if fused_vs_short_t < 1.0 and fused_vs_long_t < 1.0:
        warns.append(
            f"[{scan_id}] ALL THREE trajectories are effectively identical "
            f"(fused_vs_short={fused_vs_short_t:.3f}mm, fused_vs_long={fused_vs_long_t:.3f}mm). "
            "Strong evidence: model predicts identity transforms everywhere. "
            "Check that training converged (loss decreasing, LR >1e-5)."
        )

    report["trajectory_divergence"] = td

    # ------------------------------------------------------------------
    # C. GT-baseline score (level >= 2, requires tform_calib)
    # ------------------------------------------------------------------
    if diagnostics_level >= 2 and tform_calib is not None and frames is not None:
        try:
            from trainers.metrics.tusrec import compute_tusrec_metrics  # noqa: PLC0415
            gt_metrics = compute_tusrec_metrics(
                frames=frames.to(device),
                gt_transforms=gt_global,
                pred_transforms=gt_global,
                calib={"tform_calib": tform_calib.to(device)},
                compute_scores=True,
                enforce_lp_gp_distinct=False,
            )
            gt_gpe = gt_metrics.get("GPE_mm", None)
            gt_lpe = gt_metrics.get("LPE_mm", None)
            gt_score = gt_metrics.get("final_score", None)
            if gt_gpe is not None and gt_gpe > 1e-3:
                errors.append(
                    f"[{scan_id}] GT-baseline GPE={gt_gpe:.4f} mm (expected ~0). "
                    "evaluate pipeline itself has a bug — GT fed as pred should give GPE≈0."
                )
            if gt_lpe is not None and gt_lpe > 1e-3:
                errors.append(
                    f"[{scan_id}] GT-baseline LPE={gt_lpe:.4f} mm (expected ~0). "
                    "local_from_global → compute LPE pipeline has a roundtrip error."
                )
            if gt_score is not None and gt_score < 0.99:
                warns.append(
                    f"[{scan_id}] GT-baseline final_score={gt_score:.4f} (expected ≈1.0). "
                    "Scoring implementation may have a bug."
                )
            report["gt_baseline"] = {
                "GPE_mm": gt_gpe, "LPE_mm": gt_lpe, "final_score": gt_score,
            }
        except Exception as exc:
            warns.append(f"[{scan_id}] GT-baseline check failed: {exc}")

    # ------------------------------------------------------------------
    # D. DDF direction check (level >= 3)
    # ------------------------------------------------------------------
    if diagnostics_level >= 3 and tform_calib is not None:
        try:
            _check_ddf_direction(
                fused_global=fused_global,
                gt_global=gt_global,
                tform_calib=tform_calib,
                scan_id=scan_id,
                errors=errors,
                warns=warns,
            )
        except Exception as exc:
            warns.append(f"[{scan_id}] DDF direction check failed: {exc}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    ok = len(errors) == 0
    report["ok"] = ok

    lines = [f"[Diagnostics] scan={scan_id}  T={T}  M={M}  level={diagnostics_level}"]
    if errors:
        lines.append(f"  ERRORS ({len(errors)}):")
        for e in errors:
            lines.append(f"    ✗ {e}")
    if warns:
        lines.append(f"  WARNINGS ({len(warns)}):")
        for w in warns:
            lines.append(f"    ⚠ {w}")
    lines.append("  Trajectory divergence (mean translation mm):")
    for k, v in sorted(td.items()):
        if k.endswith("_t_mean_mm"):
            lines.append(f"    {k}: {v:.3f}")
    if report["gt_baseline"]:
        b = report["gt_baseline"]
        lines.append(f"  GT-baseline: GPE={b['GPE_mm']:.4f}mm  LPE={b['LPE_mm']:.4f}mm  score={b['final_score']:.4f}")
    if ok:
        lines.append("  → all checks PASSED")

    report["summary"] = "\n".join(lines)
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_anchor_global(
    long_global: torch.Tensor,
    anchor_indices: torch.Tensor,
    T: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand sparse anchor globals to dense (T, 4, 4) via piecewise-constant."""
    M = anchor_indices.shape[0]
    out = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(T, -1, -1).clone()
    for m in range(M):
        a = int(anchor_indices[m].item())
        b = int(anchor_indices[m + 1].item()) if m + 1 < M else T
        out[a:b] = long_global[m].unsqueeze(0).expand(b - a, -1, -1)
    return out


def _check_ddf_direction(
    fused_global: torch.Tensor,
    gt_global: torch.Tensor,
    tform_calib: torch.Tensor,
    *,
    scan_id: str,
    errors: list,
    warns: list,
    n_pixels: int = 8,
    frame_idx: int = 5,
) -> None:
    """Verify that fused_global[i] maps frame-i pixels to frame-0 coordinates.

    Samples n_pixels random pixels and checks the 3D distance between
    GT-projected and pred-projected points for a single frame.
    """
    T = fused_global.shape[0]
    if frame_idx >= T:
        frame_idx = T - 1
    device = fused_global.device
    dtype = fused_global.dtype
    tform_calib = tform_calib.to(device=device, dtype=dtype)

    # Build n_pixels random pixel homogeneous coords
    import random  # noqa: PLC0415
    # Use a fixed seed for reproducibility
    rng = random.Random(42)
    rows = [rng.randint(0, 99) for _ in range(n_pixels)]
    cols = [rng.randint(0, 99) for _ in range(n_pixels)]
    pts_px = torch.tensor(
        [rows, cols, [0] * n_pixels, [1] * n_pixels],
        device=device, dtype=dtype,
    )  # (4, n_pixels)

    pts_tool = tform_calib @ pts_px  # (4, n_pixels)

    # GT projection: apply frame_idx global transform
    gt_pts = (gt_global[frame_idx] @ pts_tool)[:3, :]  # (3, n_pixels)
    pred_pts = (fused_global[frame_idx] @ pts_tool)[:3, :]  # (3, n_pixels)

    err = (pred_pts - gt_pts).norm(dim=0)  # (n_pixels,)
    mean_err = float(err.mean().item())
    max_err = float(err.max().item())

    # As a sanity check, verify that applying IDENTITY give a larger error than pred
    identity = torch.eye(4, device=device, dtype=dtype)
    id_pts = (identity @ pts_tool)[:3, :]
    id_err = (id_pts - gt_pts).norm(dim=0).mean()

    warns.append(
        f"[{scan_id}] DDF direction check @ frame {frame_idx}: "
        f"pred_error={mean_err:.2f}mm (mean) {max_err:.2f}mm (max), "
        f"identity_error={float(id_err.item()):.2f}mm — "
        + ("pred better than identity ✓" if mean_err < float(id_err.item()) else
           "pred WORSE than identity ✗  check transform direction/units")
    )
