# Hooks

Hooks are lightweight callbacks that receive trainer events. They should not own the training loop; they attach behavior (logging, recording, checkpointing, etc.).

## Lifecycle events
Defined in `trainers/hooks/base_hook.py` and dispatched by `trainers/trainer.py`:

- Run: `before_run(mode)`, `after_run(mode)`, `on_exception(exc)`
- Train: `before_train`, `after_train`, `before_epoch`, `after_epoch(log_buffer)`, `before_step`, `after_step(log_buffer)`
- Validation: `before_val`, `after_val(log_buffer)`
- Test: `before_test`, `after_test(log_buffer)`

`log_buffer` is a dict that the trainer/evaluator fills with values like `loss`, `lr`, `epoch`, and metrics.

## Built-in hooks

### LoggerHook
File: `trainers/hooks/logger_hook.py`
- Writes a local heartbeat log (file + optional console)
- Logs metrics to MLflow when a run is active
- Optionally uploads `ctx.run_dir` as MLflow artifacts and deletes local run dir after success

### RecordRawHook
File: `trainers/hooks/record_raw_hook.py`
- Records raw input images (dataset original bytes) during `val/test`
- Optionally records `*_pred.png` and `*_gt.png` masks (binary 0/255)
- Writes `manifest.jsonl` under the records directory
- Implementation uses `Evaluator` callbacks (no duplicate eval loop in the hook)

See: `docs/records.md`

## How hooks are registered
Current default wiring is in `main.py` (you can add more `trainer.register_hook(...)` calls there).

