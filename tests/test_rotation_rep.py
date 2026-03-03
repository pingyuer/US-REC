"""Unit tests for rotation representations and rotation-aware losses.

Tests cover:
  1. quat_to_rotmat: identity quaternion → I
  2. Quaternion sign invariance: q and -q → same R; loss doesn't jump
  3. Geodesic loss: R_pred == R_gt → 0; monotonicity for small angles
  4. rot6d orthonormality: R^T R ≈ I, det(R) ≈ +1
  5. Round-trip: rotmat → quat → rotmat consistency
  6. PredictionTransform dispatch for each rotation_rep
  7. compute_dimention with rotation_rep
"""

import math
import pytest
import torch

from utils.rotation import (
    normalize_quat,
    quat_sign_align,
    quat_to_rotmat,
    rotmat_to_quat,
    rot6d_to_rotmat,
    make_se3,
    rotation_rep_to_rotmat,
    ROTATION_REP_DIM,
    get_pose_output_dim,
)
from utils.rotation_loss import (
    geodesic_loss,
    quat_inner_loss,
    l1_translation_loss,
    l2_translation_loss,
    pose_loss,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. quat_to_rotmat basics
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestQuatToRotmat:
    def test_identity_quaternion(self):
        """q = [1, 0, 0, 0] → R = I."""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        R = quat_to_rotmat(q)
        assert R.shape == (1, 3, 3)
        torch.testing.assert_close(R[0], torch.eye(3), atol=1e-6, rtol=1e-6)

    def test_180_degree_rotation_around_z(self):
        """q = [0, 0, 0, 1] → 180° around z-axis."""
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        R = quat_to_rotmat(q)
        expected = torch.tensor([[-1.0, 0.0, 0.0],
                                  [0.0, -1.0, 0.0],
                                  [0.0, 0.0, 1.0]])
        torch.testing.assert_close(R[0], expected, atol=1e-5, rtol=1e-5)

    def test_90_degree_rotation_around_x(self):
        """q = [cos(45°), sin(45°), 0, 0] → 90° around x-axis."""
        angle = math.pi / 2
        q = torch.tensor([[math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0]])
        R = quat_to_rotmat(q)
        expected = torch.tensor([[1.0, 0.0, 0.0],
                                  [0.0, 0.0, -1.0],
                                  [0.0, 1.0, 0.0]])
        torch.testing.assert_close(R[0], expected, atol=1e-5, rtol=1e-5)

    def test_batch_processing(self):
        """Multiple quaternions processed in parallel."""
        q = torch.randn(8, 4)
        R = quat_to_rotmat(q)
        assert R.shape == (8, 3, 3)
        # All should be proper rotations
        for i in range(8):
            RtR = R[i].T @ R[i]
            torch.testing.assert_close(RtR, torch.eye(3), atol=1e-5, rtol=1e-5)
            assert torch.det(R[i]).item() > 0


# ═════════════════════════════════════════════════════════════════════════════
# 2. Quaternion sign invariance
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestQuatSignInvariance:
    def test_q_and_neg_q_same_rotmat(self):
        """q and -q must produce the same rotation matrix."""
        q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        R_pos = quat_to_rotmat(q)
        R_neg = quat_to_rotmat(-q)
        torch.testing.assert_close(R_pos, R_neg, atol=1e-6, rtol=1e-6)

    def test_sign_align(self):
        """quat_sign_align should flip q to match q_ref hemisphere."""
        q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        q_neg = -q
        aligned = quat_sign_align(q_neg, q)
        dot = (aligned * q).sum(dim=-1)
        assert dot.item() > 0

    def test_quat_inner_loss_sign_invariant(self):
        """Loss should be identical for q and -q."""
        q_gt = normalize_quat(torch.randn(4, 4))
        q_pred = normalize_quat(torch.randn(4, 4))
        loss_pos = quat_inner_loss(q_pred, q_gt)
        loss_neg = quat_inner_loss(-q_pred, q_gt)
        torch.testing.assert_close(loss_pos, loss_neg, atol=1e-6, rtol=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Geodesic loss
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGeodesicLoss:
    def test_identical_matrices_zero_loss(self):
        """R_pred == R_gt → loss = 0."""
        R = torch.eye(3).unsqueeze(0).expand(4, -1, -1)
        loss = geodesic_loss(R, R)
        assert loss.item() < 1e-6

    def test_random_rotation_zero_loss(self):
        """Same random rotation → loss = 0."""
        q = normalize_quat(torch.randn(8, 4))
        R = quat_to_rotmat(q)
        loss = geodesic_loss(R, R)
        assert loss.item() < 1e-5

    def test_monotonic_with_angle(self):
        """Larger angle → larger geodesic loss."""
        losses = []
        for deg in [1, 5, 10, 30, 60, 90]:
            angle = math.radians(deg)
            q = torch.tensor([[math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0]])
            R_pred = quat_to_rotmat(q)
            R_gt = torch.eye(3).unsqueeze(0)
            loss = geodesic_loss(R_pred, R_gt)
            losses.append(loss.item())
        for i in range(len(losses) - 1):
            assert losses[i] < losses[i + 1], f"Monotonicity violated at {i}: {losses}"

    def test_known_angle(self):
        """90° rotation → loss ≈ π/2."""
        angle = math.pi / 2
        q = torch.tensor([[math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]])
        R_pred = quat_to_rotmat(q)
        R_gt = torch.eye(3).unsqueeze(0)
        loss = geodesic_loss(R_pred, R_gt)
        assert abs(loss.item() - math.pi / 2) < 0.01


# ═════════════════════════════════════════════════════════════════════════════
# 4. rot6d orthonormality
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRot6d:
    def test_orthonormal(self):
        """R^T R ≈ I for random 6D inputs."""
        r6d = torch.randn(16, 6)
        R = rot6d_to_rotmat(r6d)
        assert R.shape == (16, 3, 3)
        for i in range(16):
            RtR = R[i].T @ R[i]
            torch.testing.assert_close(RtR, torch.eye(3), atol=1e-5, rtol=1e-5)

    def test_det_positive(self):
        """det(R) ≈ +1 (proper rotation, no reflection)."""
        r6d = torch.randn(16, 6)
        R = rot6d_to_rotmat(r6d)
        for i in range(16):
            det = torch.det(R[i])
            assert abs(det.item() - 1.0) < 1e-4, f"det = {det.item()}"

    def test_identity_from_canonical_input(self):
        """Canonical e1, e2 → R = I."""
        r6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        R = rot6d_to_rotmat(r6d)
        torch.testing.assert_close(R[0], torch.eye(3), atol=1e-6, rtol=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Round-trip consistency
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRoundTrip:
    def test_rotmat_to_quat_roundtrip(self):
        """rotmat → quat → rotmat should be consistent."""
        q_orig = normalize_quat(torch.randn(8, 4))
        R = quat_to_rotmat(q_orig)
        q_back = rotmat_to_quat(R)
        R_back = quat_to_rotmat(q_back)
        torch.testing.assert_close(R, R_back, atol=1e-5, rtol=1e-5)

    def test_make_se3_shape(self):
        """make_se3 produces (B, 4, 4) with correct last row."""
        R = torch.eye(3).unsqueeze(0).expand(4, -1, -1)
        t = torch.randn(4, 3)
        T = make_se3(R, t)
        assert T.shape == (4, 4, 4)
        for i in range(4):
            torch.testing.assert_close(T[i, 3, :3], torch.zeros(3), atol=1e-7, rtol=1e-7)
            assert T[i, 3, 3].item() == 1.0


# ═════════════════════════════════════════════════════════════════════════════
# 6. PredictionTransform dispatch
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPredictionTransformDispatch:
    """Verify that PredictionTransform produces (B, num_pairs, 4, 4) for each rotation_rep."""

    @staticmethod
    def _make_dummy_transform(rotation_rep, num_pairs=1):
        from utils.transform import PredictionTransform
        image_points = torch.tensor([
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=torch.float32)
        tform_calib = torch.eye(4)
        tform_calib_R_T = torch.eye(4)
        return PredictionTransform(
            pred_type="parameter",
            label_type="transform",
            num_pairs=num_pairs,
            image_points=image_points,
            in_image_coords=True,
            tform_image_to_tool=tform_calib,
            tform_image_mm_to_tool=tform_calib_R_T,
            rotation_rep=rotation_rep,
        )

    def test_se3_expmap(self):
        pt = self._make_dummy_transform("se3_expmap", num_pairs=1)
        B = 2
        outputs = torch.randn(B, 6)  # 6 per pair
        result = pt(outputs)
        assert result.shape == (B, 1, 4, 4)

    def test_quat(self):
        pt = self._make_dummy_transform("quat", num_pairs=1)
        B = 2
        outputs = torch.randn(B, 7)  # 7 per pair
        result = pt(outputs)
        assert result.shape == (B, 1, 4, 4)
        # Check it's a valid transform: last row = [0,0,0,1]
        for i in range(B):
            torch.testing.assert_close(
                result[i, 0, 3, :],
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
                atol=1e-6, rtol=1e-6,
            )

    def test_rot6d(self):
        pt = self._make_dummy_transform("rot6d", num_pairs=1)
        B = 2
        outputs = torch.randn(B, 9)  # 9 per pair
        result = pt(outputs)
        assert result.shape == (B, 1, 4, 4)
        # Check orthonormality of rotation block
        for i in range(B):
            R = result[i, 0, :3, :3]
            RtR = R.T @ R
            torch.testing.assert_close(RtR, torch.eye(3), atol=1e-4, rtol=1e-4)

    def test_multi_pair_quat(self):
        pt = self._make_dummy_transform("quat", num_pairs=3)
        B = 4
        outputs = torch.randn(B, 7 * 3)
        result = pt(outputs)
        assert result.shape == (B, 3, 4, 4)


# ═════════════════════════════════════════════════════════════════════════════
# 7. compute_dimention with rotation_rep
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestComputeDimention:
    def test_default_se3(self):
        from trainers.utils.rec_ops import compute_dimention
        dim = compute_dimention("parameter", num_frames=2, type_option="pred")
        assert dim == 6  # 1 pair × 6

    def test_quat(self):
        from trainers.utils.rec_ops import compute_dimention
        dim = compute_dimention("parameter", num_frames=2, type_option="pred", rotation_rep="quat")
        assert dim == 7  # 1 pair × 7

    def test_rot6d(self):
        from trainers.utils.rec_ops import compute_dimention
        dim = compute_dimention("parameter", num_frames=2, type_option="pred", rotation_rep="rot6d")
        assert dim == 9  # 1 pair × 9

    def test_non_parameter_unaffected(self):
        from trainers.utils.rec_ops import compute_dimention
        dim = compute_dimention("transform", num_frames=2, type_option="pred")
        assert dim == 12  # 1 pair × 12


# ═════════════════════════════════════════════════════════════════════════════
# 8. Combined pose loss
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPoseLoss:
    def test_zero_when_identical(self):
        """Pose loss = 0 when pred == gt."""
        R = torch.eye(3).unsqueeze(0).expand(4, -1, -1)
        t = torch.randn(4, 3)
        total, breakdown = pose_loss(
            R_pred=R, R_gt=R, t_pred=t, t_gt=t,
            rot_loss_type="geodesic", trans_loss_type="l2",
        )
        assert total.item() < 1e-6
        assert breakdown["rot_loss"] < 1e-6
        assert breakdown["trans_loss"] < 1e-6

    def test_quat_inner_requires_quats(self):
        R = torch.eye(3).unsqueeze(0)
        t = torch.zeros(1, 3)
        with pytest.raises(ValueError, match="quat_inner"):
            pose_loss(
                R_pred=R, R_gt=R, t_pred=t, t_gt=t,
                rot_loss_type="quat_inner", trans_loss_type="l2",
            )

    def test_quat_inner_with_quats(self):
        q = normalize_quat(torch.randn(4, 4))
        R = quat_to_rotmat(q)
        t = torch.randn(4, 3)
        total, breakdown = pose_loss(
            R_pred=R, R_gt=R, t_pred=t, t_gt=t,
            rot_loss_type="quat_inner", trans_loss_type="l1",
            q_pred=q, q_gt=q,
        )
        assert total.item() < 1e-5

    def test_gradient_flows(self):
        """Ensure gradient flows through the loss."""
        q = torch.randn(4, 4, requires_grad=True)
        t = torch.randn(4, 3, requires_grad=True)
        R = quat_to_rotmat(normalize_quat(q))
        R_gt = torch.eye(3).unsqueeze(0).expand(4, -1, -1)
        t_gt = torch.zeros(4, 3)
        total, _ = pose_loss(
            R_pred=R, R_gt=R_gt, t_pred=t, t_gt=t_gt,
            rot_loss_type="geodesic", trans_loss_type="l2",
        )
        total.backward()
        assert q.grad is not None
        assert t.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(t.grad).any()


# ═════════════════════════════════════════════════════════════════════════════
# 9. rotation_rep_to_rotmat dispatcher
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRotationRepToRotmat:
    def test_quat_path(self):
        q = normalize_quat(torch.randn(4, 4))
        R = rotation_rep_to_rotmat(q, "quat")
        assert R.shape == (4, 3, 3)

    def test_rot6d_path(self):
        r6d = torch.randn(4, 6)
        R = rotation_rep_to_rotmat(r6d, "rot6d")
        assert R.shape == (4, 3, 3)

    def test_se3_expmap_path(self):
        euler = torch.randn(4, 3)
        R = rotation_rep_to_rotmat(euler, "se3_expmap")
        assert R.shape == (4, 3, 3)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            rotation_rep_to_rotmat(torch.randn(4, 3), "unknown_rep")
