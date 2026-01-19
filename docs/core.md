# Core Module

## Project overview
- `main.py` is the entry point for the image segmentation training pipeline. It is configuration-driven (OmegaConf YAML files + dotlist overrides) and ties together data/model/trainer/hooks/MLflow.
- Core directories:
  - `data/` hosts the dataset, dataloader, and transform builders.
  - `models/` defines segmentation models and `models/losses/`.
  - `trainers/` implements the `Trainer`, `Evaluator`, `TrainingContext`, and hooks.
  - `trainers/metrics/` holds evaluation metrics (not in `utils/`).
  - `utils/` contains small generic utilities only (no dataset code).
  - `configs/` stores YAML templates (`base.yml`, `demo.yml`, `demo_tui.yml`, `hjunet_probnew13.yaml`).
  - `tests/` mirrors the code structure for automated validation.

## Configuration-driven flow
1. `main.py` loads the selected YAML config via OmegaConf.
2. `build_dataset`, `build_dataloader`, and `build_pipeline` instantiate the dataset and transforms.
3. `build_model` resolves the configured architecture.
4. `Trainer` builds the loss/optimizer/scheduler and executes the training loop.
5. Hooks handle logging, evaluation scheduling, checkpointing, and artifacts (MLflow is optional).

## Dataset vs pipeline responsibilities
- Dataset should be "indexing only": read split/index files, store sample references, return lightweight dicts.
- Pipeline/transforms own preprocessing: read/decode bytes, build masks/targets, augment, normalize, to-tensor.

Canonical sample schema for segmentation:
- Dataset output: `{"img_ref": "...", "segments": [...], "meta": {...}}` (S3 uses `s3://bucket/key` in `img_ref`)
- Pipeline output: `{"image": Tensor, "mask": Tensor, "meta": {...}}`

## Quick start commands
```bash
pip install -r requirements.txt
pytest -q
python main.py --config configs/demo_tui.yml trainer.max_epochs=50 optimizer.lr=5e-5
```
- Override any setting with dotlist syntax (`trainer.log_interval=25`, `model.num_classes=3`).
- The trainer supports PolyLR and configurable loss/metrics.

## Next docs
- Training & modes: `docs/training.md`
- Hooks: `docs/hooks.md`
- Records/materials: `docs/records.md`
