"""Metric compatibility exports."""

import warnings


def _warn() -> None:
    warnings.warn(
        "utils.metrics is deprecated; import metrics from trainers.metrics.",
        DeprecationWarning,
        stacklevel=2,
    )


def compute_tusrec_metrics(*args, **kwargs):
    _warn()
    from trainers.metrics import compute_tusrec_metrics as _fn
    return _fn(*args, **kwargs)

__all__ = ["compute_tusrec_metrics"]
