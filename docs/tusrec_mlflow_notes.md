# TUS-REC Metrics + MLflow

## Enable MLflow
- In config, set `mlflow.enabled: true` (or `logging.mlflow.enabled: true` for rec configs).
- Provide `tracking_uri`, `experiment_name`, and optional `run_name`/`tags`.
- Example (rec):
  - `logging.mlflow.enabled: true`
  - `logging.mlflow.best_metric: final_score`
  - `logging.mlflow.best_mode: max`

## What gets logged
- Params: flattened config + CLI command + git hash (if available) + device info.
- Metrics:
  - Train: loss, lr.
  - Val: loss + pose-space metrics + TUS-REC metrics (GPE_mm/GLE_mm/LPE_mm/LLE_mm + normalized + final_score + runtime_s_per_scan).
- Artifacts:
  - `checkpoints/best/best_<metric>.pt`
  - `checkpoints/last/last.pt`
  - config yaml
  - `metrics/val_tusrec_per_scan.json` (per-scan summary)

## TUS-REC metric definitions
- Global: frame `i -> 0` reconstruction error (i>0).
- Local: frame `i -> i-1` reconstruction error (i>0).
- Pixel: mean 3D Euclidean error over all pixels (mm).
- Landmark: mean 3D Euclidean error over landmarks (mm, if provided).
- Normalization:
  - `largest_*` uses identity prediction for the same scan.
  - `metric_norm = (metric - largest) / (0 - largest)`.
  - `final_score = 0.25 * (GPE_norm + GLE_norm + LPE_norm + LLE_norm)`.

## Sanity checks
1. `pred == gt` → GPE/GLE/LPE/LLE ≈ 0.
2. `pred == identity` → metric == largest, norm == 0.
3. `final_score` stays in [0, 1] (when all four norms exist).
