# Training

## Entry point
Run everything from `main.py`:

```bash
python main.py --config configs/demo_tui.yml
python main.py --config configs/demo_tui.yml --mode val
python main.py --config configs/demo_tui.yml --mode test
```

Dotlist overrides are supported:

```bash
python main.py --config configs/demo_tui.yml trainer.max_epochs=2 optimizer.lr=1e-4
```

## Data/model construction
`main.py` builds (by split) and wires:
- dataset/dataloader: `data/builder.py` (`build_dataset`, `build_dataloader`, `build_pipeline`)
- model: `models.build_model(cfg.model)`
- trainer: `trainers/trainer.py::Trainer`

## Trainer/Evaluator flow
- Training loop lives in `Trainer.train()` / `Trainer.train_one_epoch()`.
- Validation/test loop is shared by `trainers/evaluator.py::Evaluator`.
- `Trainer.validate()` / `Trainer.test()` delegate to `Evaluator.run(...)`.

## Modes and splits
`main.py` supports:
- `--mode train`: train + (optional) val each `trainer.eval_interval`
- `--mode val`: run validation only
- `--mode test`: run test only

Splits are selected by:
- `--train-split train`
- `--val-split val`
- `--test-split test`

These map to `cfg.dataset` fields (per-split mapping supported in YAML).

## Hooks (high level)
Trainer events are dispatched to hooks:
- run lifecycle: `before_run`, `after_run`, `on_exception`
- training lifecycle: `before_train`, `before_epoch`, `before_step`, `after_step`, `after_epoch`, `after_train`
- eval lifecycle: `before_val/after_val`, `before_test/after_test`

Built-ins:
- `LoggerHook`: local log + MLflow metrics + archive run dir
- `RecordRawHook`: record raw images + optional pred/gt masks during eval/test

See: `docs/hooks.md`, `docs/records.md`, `docs/mlflow.md`.
