# Losses

This project keeps loss functions in `models/losses/` and constructs them via `models.build_loss(cfg.loss)`.

## Where losses live

- Implementations: `models/losses/`
- Factory: `models/__init__.py` (`build_loss`)
- Trainer usage: `trainers/trainer.py` (`_build_loss` calls `build_loss`)

## Supported losses (factory)

Configured via `loss.type`:

- `CrossEntropyLoss` (PyTorch)
- `BCEWithLogitsLoss` (PyTorch)
- `DiceLoss` (project)
- `DiceCELoss` (project)
- `BCEDiceLoss` (project)
- `LovaszHingeLoss` (project)

## Tensor shapes (important)

### Multi-class segmentation (recommended default)

- Model output (logits): `(B, C, H, W)`
- Target mask (class indices): `(B, H, W)` with values in `[0, C-1]`

Works with: `CrossEntropyLoss`, `DiceLoss`, `DiceCELoss`.

### Binary segmentation (legacy)

- Model output (logits): typically `(B, 1, H, W)`
- Target mask: `(B, 1, H, W)` or `(B, H, W)` depending on your dataset/pipeline

Works with: `BCEWithLogitsLoss`, `BCEDiceLoss`, `LovaszHingeLoss` (hinge-style).

## How to configure

### CrossEntropyLoss (multi-class)

```yaml
loss:
  type: CrossEntropyLoss
  ignore_index: 255
```

### DiceCELoss (multi-class, recommended)

```yaml
loss:
  type: DiceCELoss
  dice_weight: 1.0
  ce_weight: 1.0
  ignore_index: 255
  include_background: true
```

Notes:
- `ignore_index` is respected by both CE and Dice parts.
- `include_background=false` will exclude class 0 from Dice averaging (useful when background dominates).

### DiceLoss (multi-class)

```yaml
loss:
  type: DiceLoss
  ignore_index: 255
  include_background: true
```

### BCEWithLogitsLoss (binary)

```yaml
loss:
  type: BCEWithLogitsLoss
```

### BCEDiceLoss (binary)

```yaml
loss:
  type: BCEDiceLoss
```

### LovaszHingeLoss (binary)

```yaml
loss:
  type: LovaszHingeLoss
```

## Practical tips

- If you train with `CrossEntropyLoss`/`DiceCELoss`, ensure your dataset returns `mask` as class indices (`torch.long`) shaped `(H,W)` (or batched `(B,H,W)`).
- If your masks are one-hot, convert to indices before feeding CE/Dice.
- Keep `ignore_index` consistent between dataset, loss, and metrics.

