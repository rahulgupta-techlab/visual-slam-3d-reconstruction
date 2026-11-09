"""Relative camera-pose estimation and trajectory integration."""
from __future__ import annotations

import cv2
import numpy as np


def estimate_relative_pose(
    points1: np.ndarray,
    points2: np.ndarray,
    camera_matrix: np.ndarray,
    ransac_threshold: float = 1.0,
    confidence: float = 0.999,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate R and unit translation direction from corresponding pixels.

    Returns (R, t_unit, inlier_mask). Translation magnitude is not observable
    from monocular two-view geometry and is deliberately left at unit scale.
    """
    p1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)
    if len(p1) != len(p2) or len(p1) < 5:
        raise ValueError("at least five paired correspondences are required")
    essential, mask = cv2.findEssentialMat(
        p1, p2, camera_matrix, method=cv2.RANSAC,
        prob=confidence, threshold=ransac_threshold,
    )
    if essential is None or mask is None:
        raise RuntimeError("essential matrix estimation failed")
    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential, p1, p2, camera_matrix, mask=mask
    )
    inliers = pose_mask.reshape(-1).astype(bool)
    return rotation, translation.reshape(3), inliers


def accumulate_pose(
    world_from_camera: np.ndarray,
    relative_rotation: np.ndarray,
    relative_translation: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Integrate a camera-to-camera transform into a 4x4 world pose.

    The relative translation is interpreted in the previous camera frame.
    The returned pose maps current-camera coordinates into the world frame.
    """
    pose = np.asarray(world_from_camera, dtype=np.float64)
    rotation = np.asarray(relative_rotation, dtype=np.float64)
    translation = np.asarray(relative_translation, dtype=np.float64).reshape(3)
    if pose.shape != (4, 4) or rotation.shape != (3, 3):
        raise ValueError("pose and rotation must have shapes (4,4) and (3,3)")
    if scale < 0:
        raise ValueError("scale must be non-negative")
    updated = pose.copy()
    updated[:3, :3] = pose[:3, :3] @ rotation.T
    updated[:3, 3] = pose[:3, 3] - scale * (pose[:3, :3] @ rotation.T @ translation)
    return updated
