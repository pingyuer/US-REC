"""Trainers module — reconstruction / pose estimation."""

from .builder import build_optimizer
from .context import TrainingContext

__all__ = ["TrainingContext", "build_optimizer"]
