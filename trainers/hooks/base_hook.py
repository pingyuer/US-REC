class Hook:
    """Base Hook: defines event methods. Trainer triggers these."""

    def before_run(self, trainer, mode: str = "train"): pass
    def after_run(self, trainer, mode: str = "train"): pass
    def on_exception(self, trainer, exc: BaseException): pass

    def before_train(self, trainer): pass
    def after_train(self, trainer): pass

    def before_epoch(self, trainer): pass
    def after_epoch(self, trainer, log_buffer=None): pass

    def before_step(self, trainer): pass
    def after_step(self, trainer, log_buffer=None): pass

    def before_val(self, trainer): pass
    def after_val(self, trainer, log_buffer=None): pass

    def before_test(self, trainer): pass
    def after_test(self, trainer, log_buffer=None): pass
