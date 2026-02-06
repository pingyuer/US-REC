"""TUS-REC metric compatibility exports."""

from __future__ import annotations

import warnings

from trainers.metrics.tusrec import compute_tusrec_metrics as _compute_tusrec_metrics


def compute_tusrec_metrics(*args, **kwargs):
    warnings.warn(
        "utils.metrics.tusrec_metrics is deprecated; import from trainers.metrics instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _compute_tusrec_metrics(*args, **kwargs)


__all__ = ["compute_tusrec_metrics"]
