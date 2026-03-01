"""Unified metrics package — single source of truth.

The authoritative global/local compose helpers live in
``metrics.compose``.  Other metric functions are available via
``trainers.metrics`` which re-exports everything.

Usage::

    from metrics.compose import compose_global_from_local, local_from_global
    from trainers.metrics import compute_tusrec_metrics, translation_error_mm
"""

from .compose import (  # noqa: F401
    compose_global_from_local,
    local_from_global,
)

__all__ = [
    "compose_global_from_local",
    "local_from_global",
]

