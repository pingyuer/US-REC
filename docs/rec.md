# Free-hand Reconstruction (rec)

This document describes how to run the free-hand reconstruction/registration pipeline via `main_rec.py`.

## Entry point

Use `main_rec.py` with a rec config:

```bash
python main_rec.py --config configs/demo_rec24_ete.yml
python main_rec.py --config configs/demo_rec24_meta.yml

## Evaluation script

Compute TUS-REC metrics from saved predictions:

```bash
python evaluate_tus_rec.py --pred pred_T.pt --gt gt_T.pt
python evaluate_tus_rec.py --pred pred_T.npy --gt gt_T.npy \\
  --pred-ddf pred_ddf.pt --gt-ddf gt_ddf.pt \\
  --pred-vol pred_vol.pt --gt-vol gt_vol.pt
```

## RecEvaluator

Validation/testing for reconstruction runs through `trainers/rec_evaluator.py`,
which computes pose/drift metrics (and optional volume metrics) on REC batches.
```

- `demo_rec24_ete.yml`: `inter=nointer`, `meta=nonmeta`
- `demo_rec24_meta.yml`: `inter=iteratively`, `meta=meta`

## Required environment (.env)

The rec configs are S3-first. Ensure your `.env` has the S3 endpoint and credentials.

Example keys:

```
TUS_REC24_BUCKET=tus-rec-24
TUS_REC24_PREFIX_TRAIN=train
TUS_REC24_PREFIX_VAL=val
TUS_REC24_REGION=us-east-1
TUS_REC24_ENDPOINT=http://172.16.240.77:9000
TUS_REC24_FORCE_PATH_STYLE=true
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Calibration is read from S3 by default:

```
FILENAME_CALIB: s3://tus-rec-24/train/calib_matrix.csv
```

You can override any config value via dotlist:

```bash
python main_rec.py --config configs/demo_rec24_ete.yml MINIBATCH_SIZE_rec=1 NUM_EPOCHS=1
```

## Logging & MLflow

`main_rec.py` writes a local run directory under `logs/<experiment>/<run>/`. If you enable MLflow in your config, `LoggerHook` can upload the run directory to MLflow.

Minimal MLflow block:

```yaml
mlflow:
  tracking_uri: ${oc.env:MLFLOW_TRACKING_URI,http://172.16.240.33:5000}
  experiment_name: ${oc.env:MLFLOW_EXPERIMENT_NAME,rec}
  run_name: ${oc.env:MLFLOW_RUN_NAME,null}
  artifact_path: run
  archive_run_dir: true
  delete_local_run_dir: false
```

## Notes

- `data/` now only keeps `datasets/`, `transforms/`, `builder.py`, and `data/utils/` for data-layer helpers.
- The reconstruction trainer lives in `trainers/rec_trainer.py`.
- If Matplotlib fails to write cache, set:
  ```bash
  MPLCONFIGDIR=/openbayes/home/.cache/matplotlib
  ```
