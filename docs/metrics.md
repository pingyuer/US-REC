# Metrics

This project exposes evaluation metrics in `metrics/` (backed by `trainers/metrics/`).

## What’s available

### Epoch metrics (recommended)

Use `metrics.ConfusionMatrix` to accumulate per-batch predictions over a full validation epoch and then compute:

- `Accuracy`
- `mIoU`
- `Dice`

Implementation: `trainers/metrics/segmentation.py` (`ConfusionMatrix`).

### Functional helpers (legacy / script usage)

- `metrics.iou_score(output, target) -> (iou, dice)` (binary thresholded IoU/Dice)
- `metrics.edge_f1score(output, target, cls)` (edge-based F1 using OpenCV Canny)

Implementations:
- `metrics/functional.py`
- `metrics/edge.py` (requires `cv2`)

### 3) 3D reconstruction/registration metrics

These metrics support freehand reconstruction and registration evaluation:

- `metrics.translation_error(pred_t, gt_t) -> L2 translation error`
- `metrics.rotation_error(pred_R, gt_R) -> rotation error (degrees)`
- `metrics.se3_error(pred_T, gt_T) -> combined SE(3) error`
- `metrics.cumulative_drift(pred_Ts, gt_Ts) -> drift over sequence`
- `metrics.loop_closure_error(pred_Ts, gt_Ts) -> loop return error`
- `metrics.ddf_rmse(pred_ddf, gt_ddf)`
- `metrics.ddf_mae(pred_ddf, gt_ddf)`
- `metrics.volume_ssim(pred_vol, gt_vol)`
- `metrics.volume_ncc(pred_vol, gt_vol)`
- `metrics.volume_dice(pred_vol, gt_vol)`

Example:

```python
from metrics import (
    translation_error,
    rotation_error,
    se3_error,
    cumulative_drift,
    ddf_rmse,
    volume_ncc,
)

trans_err = translation_error(pred_t, gt_t)
rot_err = rotation_error(pred_R, gt_R)
se3 = se3_error(pred_T, gt_T)
drift = cumulative_drift(pred_Ts, gt_Ts)
rmse = ddf_rmse(pred_ddf, gt_ddf)
ncc = volume_ncc(pred_vol, gt_vol)
```

### 4) Standalone evaluation script

Use `evaluate_tus_rec.py` to compute metrics from saved tensors:

```bash
python evaluate_tus_rec.py --pred pred_T.pt --gt gt_T.pt
python evaluate_tus_rec.py --pred pred_T.npy --gt gt_T.npy \\
  --pred-vol pred_vol.pt --gt-vol gt_vol.pt
```

## How to use in training

### 1) Configure metrics in YAML

Add a `metrics` list to your config:

```yaml
model:
  num_classes: 2

loss:
  type: DiceCELoss
  ignore_index: 255

metrics:
  - mIoU
  - Dice
  - Accuracy
```

Notes:
- `model.num_classes` is required for `ConfusionMatrix`.
- `loss.ignore_index` (if present) will be used to ignore those labels during metric accumulation.

### 2) Validation output

`Trainer.validate()` returns a dict like:

```python
{
  "val_loss": 0.123,
  "mIoU": 0.78,
  "Dice": 0.85,
  "Accuracy": 0.93,
}
```

Your `LoggerHook` / MLflow logging will receive this dict in `after_val(log_buffer=...)`.

## How to use in scripts

### Confusion matrix usage

```python
import torch
from trainers.metrics import ConfusionMatrix

conf = ConfusionMatrix(num_classes=2, ignore_index=255)
conf.update(preds=logits, target=mask)  # logits: (B,C,H,W), mask: (B,H,W)
results = conf.compute(["mIoU", "Dice", "Accuracy"])
```

### Edge F1 usage (optional)

`edge_f1score` depends on OpenCV (`cv2`). If you don’t have it installed, it will raise an error.

```python
from trainers.metrics import edge_f1score

f1 = edge_f1score(output=logits, target=target, cls=2)
```
