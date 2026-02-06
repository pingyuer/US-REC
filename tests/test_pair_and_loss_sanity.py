import torch

from trainers.utils.loss import compute_loss
from utils.rec_ops import data_pairs_adjacent
from utils.utils_ori import reference_image_points


def test_data_pairs_adjacent_chain():
    pairs = data_pairs_adjacent(5)
    expected = torch.tensor([[0, 0], [0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long)
    assert torch.equal(pairs, expected)


def test_mse_points_rigid_loss_split():
    labels = torch.zeros(2, 2, 3, 4)
    pred = torch.ones_like(labels)
    frames = torch.zeros(2, 2, 8, 8)

    loss, loss_rec, loss_reg, dist, wrap_dist, extras = compute_loss(
        loss_type="MSE_points",
        labels=labels,
        pred_pts=pred,
        frames=frames,
        step=0,
        criterion=torch.nn.MSELoss(),
        img_loss=torch.nn.MSELoss(),
        regularization=lambda x: x.mean() if x is not None else torch.tensor(0.0),
        reg_loss_weight=1000.0,
        ddf_dirc="Move",
        conv_coords="optimised_coord",
        option="common_volume",
        device=torch.device("cpu"),
        scatter_pts_registration=lambda *args, **kwargs: None,
        scatter_pts_interpolation=lambda *args, **kwargs: None,
        wrapped_pred_dist_fn=lambda *args, **kwargs: None,
        rigid_only=True,
    )

    assert torch.allclose(loss, loss_rec)
    assert float(loss_reg.item()) == 0.0
    assert float(wrap_dist.item()) == 0.0
    assert extras.get("wrap_enabled") is False


def test_reference_image_points_bounds():
    pts = reference_image_points((8, 10), (8, 10))
    assert float(pts[0].min().item()) >= 0.0
    assert float(pts[1].min().item()) >= 0.0
    assert float(pts[0].max().item()) <= 7.0
    assert float(pts[1].max().item()) <= 9.0
