# TUS-Freehand-REC Debug Summary (Dataset + IO)

## Context
- Entry: `main_rec.py` with `configs/demo_rec24_ete.yml` / `configs/demo_rec24_meta.yml`.
- S3 layout:
  - Train: `train/frames/<subject>/<scan>.h5` (contains `frames` + `tforms`)
  - Val/Test: `val/frames/<subject>/<scan>.h5` and `val/transfs/<subject>/<scan>.h5` (tforms split)
  - Landmarks: `train/landmark/landmark_<subject>.h5`, `val/landmark/landmark_<subject>.h5`

## Root Cause Found
- `main_rec` failed because it tried to read `train/transfs/...` which does not exist.
- Train H5 already embeds `tforms`; Val/Test use separate `transfs` H5.

## Code Changes Applied
- `data/datasets/TUS_rec_s3.py`
  - Added configurable `cache_size` (in‑memory LRU).
  - Added `TUSRecS3PairBuffer` (IterableDataset) implementing slice‑level buffered pair sampling:
    - Per epoch: pick subset of slices (active_slice_ratio / active_slice_max)
    - From each slice: sample pairs by ratio (pair_sample_ratio)
    - Store in in‑memory buffer (buffer_gb) and shuffle globally
    - `set_epoch()` for epoch‑wise re‑sampling
  - Added optional timing logs (`debug_timing`, `timing_samples`) for load/pair sampling.

- `trainers/rec_trainer.py`
  - DataLoader now disables shuffle for IterableDataset.
  - Calls `set_epoch()` if dataset supports it.
  - Uses `get_example()` for IterableDataset to infer image size.

- Configs switched to buffered dataset and split‑specific tform_dir:
  - `configs/demo_rec24_ete.yml`
  - `configs/demo_rec24_meta.yml`

## Tests Added/Updated
- `tests/test_tus_rec_s3.py`
  - `test_separate_tform_dir` (frames/tforms split)
  - `test_split_specific_tform_dir` (train uses embedded tforms; val uses transfs)
  - `test_cache_size_controls_reads`
  - `test_pair_buffer_sampling`

## Current Config (core)
```yaml
dataset:
  name: data.datasets.TUS_rec_s3.TUSRecS3PairBuffer
  frame_dir: frames
  tform_dir:
    train: null
    val: transfs
    test: transfs
  pair_sample_ratio: 0.2
  buffer_gb: 8.0
  active_slice_ratio: 0.5
  active_slice_max: null
  cache_size: 0
```

## Quick Timing (manual test)
- Example timing output:
  - `[rec_s3_timing] load_h5=3.432s train/frames/006/RH_Par_S_DtP.h5`
  - `[rec_s3_timing] sample_pairs=0.000s k=107`
- Indicates main cost is H5 read/decode, not sampling.

## Known Issues / Next Steps
- If output is slow: reduce `active_slice_ratio` or `active_slice_max`, or use small `buffer_gb`.
- If GPU under‑utilized: increase DataLoader workers or batch size; consider subject‑grouped sampling.
- Optional: add per‑epoch buffer summary logging (buffer build time, pair count).

## Useful Commands
- Light dataset timing (no training):
```bash
python - <<'PY'
from omegaconf import OmegaConf
from main_rec import load_dotenv
from data.builder import build_dataset
load_dotenv('.env', override=False)
cfg = OmegaConf.load('configs/demo_rec24_ete.yml')
cfg = OmegaConf.merge(cfg, OmegaConf.create({'dataset': {'debug_timing': True, 'timing_samples': 3, 'active_slice_max': 1, 'buffer_gb': 0.1}}))
ds = build_dataset(cfg, split='train')
if hasattr(ds, 'set_epoch'):
    ds.set_epoch(0)
for i, _ in enumerate(ds):
    if i >= 4:
        break
PY
```

