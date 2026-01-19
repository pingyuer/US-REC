# Records (materials)

The records system saves experiment materials into `ctx.run_dir` so they are archived together (and can be uploaded to MLflow as artifacts).

## Output layout
When enabled, files are written under:

- `logs/<exp>/<run>/records/raw/<split>/epoch_<k>/...`
- `logs/<exp>/<run>/records/raw/manifest.jsonl`

Per sample:
- raw image: `<img_file>.png` (original dataset bytes)
- prediction mask: `<img_file>_pred.png` (0/255)
- ground-truth mask: `<img_file>_gt.png` (0/255)

## Config
Add to YAML (defaults live in `configs/base.yml`):

```yaml
records:
  enabled: true
  splits: ["val", "test"]
  interval_epochs: 1
  num_samples: 16
  out_dir: "records/raw"
  save_pred_mask: true
  save_gt_mask: true
  threshold: 0.5
```

Notes:
- `interval_epochs` applies to `val` only; `test` is usually run once.
- `threshold` is used only for 1-channel sigmoid models; multi-class models use `argmax`.

## How it works
`RecordRawHook` installs an evaluator callback before `val/test`. The callback receives:
- `batch["meta"]` (to locate the original raw file via `meta.img_file`)
- `outputs` and `masks` (to export pred/gt masks aligned to the raw image size)

## MLflow archiving
If `mlflow.archive_run_dir: true`, `LoggerHook` uploads the whole `ctx.run_dir` at the end of the run, which includes `records/`.

