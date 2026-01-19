"""Calibration helpers for rec/rec-reg trainer."""

from __future__ import annotations

from data.utils.calib import read_calib_matrices


def load_calibration(filename_calib, resample_factor, device):
    """Load calibration matrices and return (scale, R_T, calib)."""
    return read_calib_matrices(
        filename_calib=filename_calib,
        resample_factor=resample_factor,
        device=device,
    )
