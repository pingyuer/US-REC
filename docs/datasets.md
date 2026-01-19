# Dataset Module

This repo supports local and S3-backed datasets. The entry point always builds datasets via `data/builder.py`.

## Loading and augmentation
- Use `data/builder.py` to call `build_dataset`, `build_dataloader`, and `build_pipeline` with the desired split.
- Transform registry (`data/transforms/registry.py`) lets you register custom augmentations and reuse them across splits.
- The pipeline can integrate torchvision, Albumentations, or custom transforms while keeping masks paired with images.

## Dataset is indexing-only (recommended)
Datasets should only build an index and return references; all decoding/mask-building should live in transforms.

Expected keys:
- Dataset output: `img_ref` (local path or `s3://bucket/key`), `segments` (polygons), `meta` (annotation dict)
- After pipeline: `image` and `mask` tensors (plus `meta`)

Example pipeline (registry + Albumentations as a step):
```yaml
pipeline:
  train:
    - type: LoadRawBytes
    - type: DecodeImage
    - type: BuildMaskFromSegments
    - type: ApplyAlbumentations
      transforms:
        - type: HorizontalFlip
          p: 0.5
        - type: Normalize
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
        - type: ToTensorV2
    - type: FinalizeSegSample
```

## TUI (local)
Implementation: `data/datasets/TUI.py`
- Annotations are JSON list entries; each entry includes polygons under `nodule_location[*].segment`.
- `meta.bn` can exist (`benign/malignant`) but current training is segmentation-only.
- `split_file` is a plain text file listing `img_file` entries (one per line).

## TUI (S3)
Implementation: `data/datasets/TUI_s3.py`
- `split_file` can be a local path or an S3-relative key (resolved under `dataset.prefix`).
- Default split files in `configs/demo_tui.yml` match the current bucket layout:
  - `filter/train_list.txt`
  - `filter/val_list.txt`

S3 connection fields:
- `dataset.bucket`, `dataset.prefix`, `dataset.img_dir`
- `dataset.region`, `dataset.endpoint` (MinIO/S3-compatible), `dataset.force_path_style` (defaults to true)
- Credentials are read from environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (optional `AWS_SESSION_TOKEN`)

Common pattern: train/val with split files; test without split.
- Configure `dataset.prefix` as a per-split mapping (train/val/test).
- Set `dataset.split_file.test: null` to use the entire test prefix without filtering.
