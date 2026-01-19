from .base_hook import Hook
from .registry import register_hook


@register_hook("EvalHook")
class EvalHook(Hook):
    """
    Hook that runs validation every N epochs.
    """

    def __init__(self, interval: int = 1):
        self.interval = interval

    def after_epoch(self, trainer, log_buffer=None):
        if (trainer.epoch + 1) % self.interval == 0:
            trainer.call_hooks("before_val")
            results = trainer.validate()
            # results is dict like {"val_loss": ..., "val_dice": ...}
            trainer.call_hooks("after_val", log_buffer=results)
