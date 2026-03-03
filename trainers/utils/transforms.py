"""Transform builders for rec/rec-reg trainer."""

from __future__ import annotations

from utils.transform import LabelTransform, PredictionTransform


def build_transforms(
    *,
    label_type,
    pred_type,
    data_pairs,
    image_points,
    tform_calib,
    tform_calib_R_T,
    rotation_rep="se3_expmap",
):
    """Return (label_transform, prediction_transform)."""
    label_transform = LabelTransform(
        label_type,
        pairs=data_pairs,
        image_points=image_points,
        in_image_coords=True,
        tform_image_to_tool=tform_calib,
        tform_image_mm_to_tool=tform_calib_R_T,
    )
    prediction_transform = PredictionTransform(
        pred_type,
        "transform",
        num_pairs=data_pairs.shape[0] - 1,
        image_points=image_points,
        in_image_coords=True,
        tform_image_to_tool=tform_calib,
        tform_image_mm_to_tool=tform_calib_R_T,
        rotation_rep=rotation_rep,
    )
    return label_transform, prediction_transform
