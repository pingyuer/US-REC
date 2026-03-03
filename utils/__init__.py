"""Generic project-wide utilities (non-dataset, non-model).

Sub-modules:
  geometry       - image points, plane normals, angle helpers (canonical)
  interpolation  - 3-D scattered-data interpolation
  rotation       - rotation representations & conversions
  rotation_loss  - geodesic / chordal rotation losses
  transform      - label / prediction transform helpers
  funcs          - volume math (common_volume, wrapped_pred_dist)
  loggers/       - experiment loggers (MLflow, base)

Model factory: models.pairwise.build_model
Rec ops:       trainers.utils.rec_ops
"""
