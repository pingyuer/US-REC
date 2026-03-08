import random

import torch

from trainers.utils.vq_memory import (
    valid_length,
    pad_sequence_list,
    build_batched_scan_cache,
    build_consistency_views,
    build_masked_geom_targets,
)


class _FakeEncoder:
    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        return x.float()


class _FakeVQ:
    def __call__(self, z_e: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "z_q": z_e + 1.0,
            "z_q_detached": (z_e + 1.0).detach(),
            "commit_loss": z_e.mean(),
            "indices": torch.arange(z_e.shape[1], device=z_e.device).unsqueeze(0).expand(z_e.shape[0], -1),
        }


class _FakeModel:
    def __init__(self):
        self.encoder = _FakeEncoder()
        self.vq = _FakeVQ()

    def proj_vq(self, h: torch.Tensor) -> torch.Tensor:
        return h

    def contextualize_memory(self, z_q: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return z_q + 2.0
        weights = mask.unsqueeze(-1).float()
        return z_q + 2.0 * weights

    def summary_pool(self, z_q: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return z_q.mean(dim=1)
        weights = mask.unsqueeze(-1).float()
        return (z_q * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def encode_scan_anchors(self, scan_frames: torch.Tensor) -> dict[str, torch.Tensor]:
        h_all = self.encoder.encode_sequence(scan_frames)
        anchors = h_all[:, ::2]
        z_q = anchors + 10.0
        z_ctx = self.contextualize_memory(z_q)
        return {
            "z_q": z_q,
            "z_ctx": z_ctx,
            "z_q_detached": z_q.detach(),
            "g": self.summary_pool(z_ctx),
            "commit_loss": anchors.mean(),
            "vq_indices": torch.arange(anchors.shape[1], device=anchors.device).unsqueeze(0),
            "h_all": h_all,
        }


def test_valid_length_uses_mask_sum():
    mask = torch.tensor([True, True, False, False])
    assert valid_length(mask, 99) == 2
    assert valid_length(None, 7) == 7


def test_pad_sequence_list_returns_mask():
    padded, mask = pad_sequence_list([
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[3.0]]),
    ])
    assert padded.shape == (2, 2, 1)
    assert mask.tolist() == [[True, True], [True, False]]
    assert padded[1, 1, 0].item() == 0.0


def test_build_batched_scan_cache_respects_scan_mask():
    model = _FakeModel()
    scan_frames = torch.tensor([
        [[1.0], [2.0], [3.0], [4.0]],
        [[5.0], [6.0], [0.0], [0.0]],
    ])
    scan_mask = torch.tensor([
        [True, True, True, True],
        [True, True, False, False],
    ])

    cache = build_batched_scan_cache(model, scan_frames, scan_mask)
    assert cache["z_q"].shape == (2, 2, 1)
    assert cache["memory_mask"].tolist() == [[True, True], [True, False]]
    assert cache["scan_mask"].tolist() == [[True, True, True, True], [True, True, False, False]]
    assert cache["vq_indices"][1, 1].item() == -1


def test_build_consistency_views_respects_masked_lengths():
    model = _FakeModel()
    h_all = torch.tensor([
        [[1.0], [2.0], [3.0], [4.0]],
        [[5.0], [6.0], [0.0], [0.0]],
    ])
    scan_mask = torch.tensor([
        [True, True, True, True],
        [True, True, False, False],
    ])

    view1, view2 = build_consistency_views(
        model,
        anchor_stride=2,
        scan_frames=torch.zeros(2, 4, 1),
        h_all=h_all,
        scan_mask=scan_mask,
        ratio=1.0,
        rng=random.Random(0),
    )
    assert view1["g"].shape == (2, 1)
    assert view2["g"].shape == (2, 1)
    assert torch.isfinite(view1["g"]).all()
    assert torch.isfinite(view2["g"]).all()


def test_build_masked_geom_targets_ignores_padded_suffix():
    scan_gt = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(2, 4, 1, 1)
    scan_gt[1, 2:, :3, 3] = 999.0
    scan_mask = torch.tensor([
        [True, True, True, True],
        [True, True, False, False],
    ])

    target = build_masked_geom_targets(scan_gt, n_waypoints=3, scan_gt_mask=scan_mask)
    assert target.shape == (2, 3, 6)
    assert target[1].abs().max().item() < 1e-4