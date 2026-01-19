import torch

from models.losses.dice import DiceCELoss, DiceLoss


def test_dice_loss_perfect_prediction_is_near_zero():
    # 2-class, perfect prediction for a 2x2 mask
    logits = torch.tensor(
        [
            [
                [[10.0, -10.0], [10.0, -10.0]],  # class 0 logits
                [[-10.0, 10.0], [-10.0, 10.0]],  # class 1 logits
            ]
        ]
    )
    target = torch.tensor([[[0, 1], [0, 1]]])
    loss = DiceLoss()
    out = loss(logits, target)
    assert out.item() < 1e-3


def test_dice_loss_worst_prediction_is_high():
    # Flip the logits so argmax is always wrong.
    logits = torch.tensor(
        [
            [
                [[-10.0, 10.0], [-10.0, 10.0]],  # predicts class 1
                [[10.0, -10.0], [10.0, -10.0]],  # predicts class 0
            ]
        ]
    )
    target = torch.tensor([[[0, 1], [0, 1]]])
    loss = DiceLoss()
    out = loss(logits, target)
    assert out.item() > 0.8


def test_dice_loss_ignore_index_excludes_pixels():
    ignore = 255
    logits = torch.tensor(
        [
            [
                [[10.0, -10.0], [10.0, -10.0]],  # class 0 logits
                [[-10.0, 10.0], [-10.0, 10.0]],  # class 1 logits
            ]
        ]
    )
    # one ignored pixel
    target = torch.tensor([[[0, 1], [ignore, 1]]])
    loss = DiceLoss(ignore_index=ignore)
    out = loss(logits, target)
    assert torch.isfinite(out)
    # still near zero because non-ignored pixels are perfectly predicted
    assert out.item() < 1e-3


def test_dice_ce_loss_backward_has_gradients():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.randint(0, 3, (2, 16, 16))
    loss_fn = DiceCELoss(dice_weight=1.0, ce_weight=1.0, ignore_index=255)
    out = loss_fn(logits, target)
    out.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

