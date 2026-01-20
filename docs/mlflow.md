# MLflow

This project uses MLflow via `MLflowHook` and a logger adapter.

## Minimal config

```yaml
mlflow:
  tracking_uri: "http://127.0.0.1:5000"
  experiment_name: "seg"
  run_name: null
  tags: {}

  # Archive everything under ctx.run_dir as artifacts on run end.
  archive_run_dir: true
  artifact_path: "run"

  # Optional: delete local ctx.run_dir after successful upload.
  delete_local_run_dir: false
```

Notes:
- If `mlflow` block is missing, MLflow is disabled and training still runs.
- Metrics and artifacts are logged by `MLflowHook` when `mlflow.enabled: true`.
- Artifacts are archived by `LoggerHook.after_run()` by uploading the whole `ctx.run_dir`.

## What gets uploaded
By default (depending on enabled hooks/config), artifacts include:
- `config.yaml` (saved by `TrainingContext.save_config`)
- `train.log` (written by `LoggerHook`)
- `records/raw/...` (written by `RecordRawHook` if enabled)
- checkpoints (if you add a checkpoint hook that writes into `ctx.run_dir`)

## Integration test (opt-in)

```bash
RUN_MLFLOW_INTEGRATION=1 pytest -q tests/test_mlflow_integration.py
```
