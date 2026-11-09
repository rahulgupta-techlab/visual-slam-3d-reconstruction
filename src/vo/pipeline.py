"""TUM RGB-D image reader and monocular ORB visual-odometry pipeline."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .features import detect_orb, match_descriptors
from .pose import accumulate_pose, estimate_relative_pose

TUM_FX, TUM_FY, TUM_CX, TUM_CY = 517.3, 516.5, 318.6, 255.3


def read_associations(sequence_dir: Path) -> list[tuple[float, Path]]:
    """Read TUM rgb.txt (timestamp and image path), ignoring comments."""
    association_file = sequence_dir / "rgb.txt"
    if not association_file.exists():
        raise FileNotFoundError(f"Missing {association_file}; download the TUM sequence first")
    frames: list[tuple[float, Path]] = []
    with association_file.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 2:
                frames.append((float(fields[0]), sequence_dir / fields[1]))
    return frames


def run_monocular_vo(
    sequence_dir: Path,
    output_file: Path,
    max_frames: int | None = None,
    min_matches: int = 8,
) -> int:
    """Run monocular VO and save TUM-format estimated poses.

    The first pose is identity. Each recovered translation has unit length;
    no ground-truth scale is used during estimation.
    """
    frames = read_associations(sequence_dir)
    if max_frames:
        frames = frames[:max_frames]
    if len(frames) < 2:
        raise ValueError("sequence must contain at least two frames")
    K = np.array([[TUM_FX, 0.0, TUM_CX], [0.0, TUM_FY, TUM_CY], [0.0, 0.0, 1.0]])
    poses: list[tuple[float, np.ndarray]] = []
    current_pose = np.eye(4, dtype=np.float64)
    previous_image = cv2.imread(str(frames[0][1]), cv2.IMREAD_GRAYSCALE)
    if previous_image is None:
        raise FileNotFoundError(frames[0][1])
    previous_keypoints, previous_descriptors = detect_orb(previous_image)
    poses.append((frames[0][0], current_pose.copy()))

    for timestamp, image_path in frames[1:]:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        keypoints, descriptors = detect_orb(image)
        matches = match_descriptors(previous_descriptors, descriptors)
        if len(matches) >= min_matches:
            p1 = np.float64([previous_keypoints[m.queryIdx].pt for m in matches])
            p2 = np.float64([keypoints[m.trainIdx].pt for m in matches])
            try:
                R, t, inliers = estimate_relative_pose(p1, p2, K)
                if int(inliers.sum()) >= min_matches:
                    current_pose = accumulate_pose(current_pose, R, t, scale=1.0)
            except (RuntimeError, ValueError, cv2.error):
                pass
        poses.append((timestamp, current_pose.copy()))
        previous_image, previous_keypoints, previous_descriptors = image, keypoints, descriptors

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=" ")
        for timestamp, pose in poses:
            q = rotation_matrix_to_quaternion(pose[:3, :3])
            x, y, z = pose[:3, 3]
            writer.writerow([f"{timestamp:.9f}", f"{x:.9f}", f"{y:.9f}", f"{z:.9f}",
                             f"{q[0]:.9f}", f"{q[1]:.9f}", f"{q[2]:.9f}", f"{q[3]:.9f}"])
    return len(poses)


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a rotation matrix to TUM quaternion order (qx, qy, qz, qw)."""
    q = cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].reshape(3)
    angle = float(np.linalg.norm(q))
    if angle < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    axis = q / angle
    half = angle / 2.0
    return float(axis[0] * np.sin(half)), float(axis[1] * np.sin(half)), float(axis[2] * np.sin(half)), float(np.cos(half))
