import cv2
import numpy as np
from vo.pose import estimate_relative_pose, accumulate_pose


def test_essential_matrix_recovers_known_translation_direction():
    rng = np.random.default_rng(12)
    K = np.array([[520.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]])
    angle = 0.025
    R, _ = cv2.Rodrigues(np.array([0.0, angle, 0.0]))
    t = np.array([0.35, 0.02, 0.0])
    xyz = np.column_stack([rng.uniform(-2, 2, 250), rng.uniform(-1, 1, 250), rng.uniform(4, 9, 250)])
    p1h = (K @ xyz.T).T
    p1 = p1h[:, :2] / p1h[:, 2:3]
    xyz2 = (R @ xyz.T + t[:, None]).T
    p2h = (K @ xyz2.T).T
    p2 = p2h[:, :2] / p2h[:, 2:3]
    p1 += rng.normal(0, 0.15, p1.shape)
    p2 += rng.normal(0, 0.15, p2.shape)
    R_est, t_est, inliers = estimate_relative_pose(p1, p2, K, ransac_threshold=1.0)
    assert inliers.sum() > 180
    assert np.trace(R_est.T @ R) > 2.9
    assert abs(float(np.dot(t_est / np.linalg.norm(t_est), t / np.linalg.norm(t)))) > 0.98


def test_trajectory_accumulation_identity_translation():
    pose = np.eye(4)
    R = np.eye(3)
    t = np.array([-1.0, 0.0, 0.0])
    updated = accumulate_pose(pose, R, t, scale=2.5)
    np.testing.assert_allclose(updated[:3, 3], [2.5, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(updated[:3, :3], np.eye(3), atol=1e-9)
