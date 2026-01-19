# Docs

This repo is a small mmseg-style training framework: config → data/model/trainer build → hooks → MLflow artifacts.

## Start here
- `README.md` (project quick start)
- `docs/core.md` (repo layout + config conventions)
- `docs/training.md` (Trainer/Evaluator flow, modes)
- `docs/hooks.md` (hook lifecycle + built-in hooks)
- `docs/rec.md` (free-hand reconstruction: entrypoint + S3 config)

## Reference
- `docs/datasets.md` (dataset builders + TUI/TUIS3 split files)
- `docs/records.md` (record raw images + pred/gt masks as artifacts)
- `docs/mlflow.md` (MLflow config + artifact archiving)
- `docs/metrics.md` (metrics package usage)
- `docs/losses.md` (loss factory + supported losses)
- `docs/testing.md` (pytest layout + opt-in integration tests)
