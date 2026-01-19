import torch

from trainers.metrics import ConfusionMatrix
from trainers.metrics.functional import iou_score
from models import build_loss


def test_build_loss_dice_ce_forward():
    loss = build_loss(
        {
            "type": "DiceCELoss",
            "dice_weight": 1.0,
            "ce_weight": 1.0,
            "ignore_index": 255,
        }
    )
    logits = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 3, (2, 32, 32))
    target[0, 0, 0] = 255
    out = loss(logits, target)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_confusion_matrix_metrics_simple_case():
    # 2-class example
    conf = ConfusionMatrix(num_classes=2, ignore_index=None)
    preds = torch.tensor([[[0, 1], [1, 0]]])
    target = torch.tensor([[[0, 1], [0, 0]]])
    conf.update(preds, target)
    metrics = conf.compute(["mIoU", "Dice", "Accuracy"])

    # Manually:
    # class0: tp=2 (positions (0,0) and (1,1)), fp=0, fn=1 -> iou=2/3, dice=4/5
    # class1: tp=1, fp=1, fn=0 -> iou=1/2, dice=2/3
    # mean iou = (2/3 + 1/2)/2 = 7/12
    assert abs(metrics["mIoU"] - (7 / 12)) < 1e-6
    assert 0.0 <= metrics["Dice"] <= 1.0
    assert abs(metrics["Accuracy"] - (3 / 4)) < 1e-6


def test_confusion_matrix_accepts_logits_input():
    conf = ConfusionMatrix(num_classes=3, ignore_index=None)
    # logits (B,C,H,W) where argmax gives class map:
    # [[0,2],
    #  [1,1]]
    logits = torch.tensor(
        [
            [
                [[10.0, -1.0], [-1.0, -1.0]],  # class0
                [[-1.0, -1.0], [10.0, 10.0]],  # class1
                [[-1.0, 10.0], [-1.0, -1.0]],  # class2
            ]
        ]
    )
    target = torch.tensor([[[0, 2], [1, 0]]])
    conf.update(logits, target)
    metrics = conf.compute(["Accuracy"])
    # 3 correct out of 4
    assert abs(metrics["Accuracy"] - 0.75) < 1e-6


def test_confusion_matrix_ignore_index_excludes_pixels():
    ignore = 255
    preds = torch.tensor([[[0, 1], [1, 0]]])
    target = torch.tensor([[[0, 1], [0, ignore]]])
    conf = ConfusionMatrix(num_classes=2, ignore_index=ignore)
    conf.update(preds, target)
    metrics = conf.compute(["Accuracy"])
    # ignored pixel should not count; among remaining 3 pixels, 2 are correct
    assert abs(metrics["Accuracy"] - (2 / 3)) < 1e-6


def test_iou_score_binary_is_hand_calculable():
    # output logits -> sigmoid -> threshold at 0.5
    # predicted: [[1,0],[1,0]]; target: [[1,1],[0,0]]
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    iou, dice = iou_score(logits, target)
    # iou_score uses a smoothing constant (1e-5)
    smooth = 1e-5
    expected_iou = (1 + smooth) / (3 + smooth)
    expected_dice = (2 * expected_iou) / (expected_iou + 1)
    assert abs(iou - expected_iou) < 1e-9
    assert abs(dice - expected_dice) < 1e-9
