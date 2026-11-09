"""KITTI odometry stereo loader and metric-scale visual odometry."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .features import detect_orb, match_descriptors
from .pipeline import rotation_matrix_to_quaternion


def load_kitti_sequence(sequence_dir: Path) -> tuple[list[tuple[float, Path, Path]], np.ndarray, float]:
    """Load rectified KITTI grayscale stereo frames, intrinsics and baseline."""
    sequence_dir = Path(sequence_dir)
    calib_path = sequence_dir / "calib.txt"
    times_path = sequence_dir / "times.txt"
    left_dir, right_dir = sequence_dir / "image_0", sequence_dir / "image_1"
    for required in (calib_path, times_path, left_dir, right_dir):
        if not required.exists():
            raise FileNotFoundError(f"Missing KITTI input: {required}")
    projections: dict[str, np.ndarray] = {}
    for line in calib_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        if key.strip() in {"P0", "P1"}:
            projections[key.strip()] = np.fromstring(values, sep=" ").reshape(3, 4)
    if not {"P0", "P1"}.issubset(projections):
        raise ValueError("calib.txt must contain P0 and P1 projection matrices")
    P0, P1 = projections["P0"], projections["P1"]
    K = P0[:, :3]
    baseline = abs(float(P1[0, 3] / P1[0, 0] - P0[0, 3] / P0[0, 0]))
    if baseline <= 0:
        raise ValueError("Could not derive a positive stereo baseline from P0/P1")
    timestamps = [float(x.strip()) for x in times_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    left = sorted(left_dir.glob("*.png"))
    right = sorted(right_dir.glob("*.png"))
    if not timestamps or len(left) != len(right) or len(left) != len(timestamps):
        raise ValueError("KITTI image_0, image_1 and times.txt must have matching frame counts")
    return [(timestamps[i], left[i], right[i]) for i in range(len(left))], K, baseline


def stereo_disparity(left: np.ndarray, right: np.ndarray, num_disparities: int = 128) -> np.ndarray:
    """Compute positive disparity (pixels) for a rectified grayscale stereo pair."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("stereo images must be same-sized grayscale arrays")
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disparities, blockSize=5,
        P1=8 * 5 * 5, P2=32 * 5 * 5, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(left, right).astype(np.float32) / 16.0


def pixels_to_3d(points: np.ndarray, disparity: np.ndarray, K: np.ndarray, baseline: float) -> tuple[np.ndarray, np.ndarray]:
    """Back-project pixel coordinates with stereo disparity; returns points and valid mask."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    u = np.clip(np.rint(pts[:, 0]).astype(int), 0, disparity.shape[1] - 1)
    v = np.clip(np.rint(pts[:, 1]).astype(int), 0, disparity.shape[0] - 1)
    d = disparity[v, u]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    valid = np.isfinite(d) & (d > 0.5)
    z = np.zeros_like(d, dtype=np.float32)
    z[valid] = float(fx * baseline) / d[valid]
    valid &= (z > 0.1) & (z < 100.0)
    xyz = np.zeros((len(pts), 3), dtype=np.float32)
    xyz[:, 0] = (pts[:, 0] - cx) * z / fx
    xyz[:, 1] = (pts[:, 1] - cy) * z / fy
    xyz[:, 2] = z
    return xyz, valid


def _sample_cloud(disparity: np.ndarray, K: np.ndarray, baseline: float, stride: int = 12) -> np.ndarray:
    h, w = disparity.shape
    yy, xx = np.mgrid[10:h:stride, 10:w:stride]
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    xyz, valid = pixels_to_3d(coords, disparity, K, baseline)
    return xyz[valid]


def run_kitti_stereo_vo(sequence_dir: Path, output_prefix: Path, max_frames: int | None = None) -> int:
    """Run stereo metric-scale VO and export TUM trajectory, ego CSV and sparse PLY."""
    frames, K, baseline = load_kitti_sequence(sequence_dir)
    if max_frames is not None:
        frames = frames[:max_frames]
    if len(frames) < 2:
        raise ValueError("KITTI sequence must contain at least two stereo frames")

    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
        P1=200, P2=800, disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    poses: list[tuple[float, np.ndarray]] = []
    cloud_world: list[np.ndarray] = []
    pose = np.eye(4, dtype=np.float64)
    prev_left = cv2.imread(str(frames[0][1]), cv2.IMREAD_GRAYSCALE)
    prev_right = cv2.imread(str(frames[0][2]), cv2.IMREAD_GRAYSCALE)
    if prev_left is None or prev_right is None:
        raise FileNotFoundError("Could not read first KITTI stereo image pair")
    prev_disp = stereo.compute(prev_left, prev_right).astype(np.float32) / 16.0
    prev_kp, prev_desc = detect_orb(prev_left)
    poses.append((frames[0][0], pose.copy()))
    cloud = _sample_cloud(prev_disp, K, baseline)
    cloud_world.append(cloud)

    for timestamp, left_path, right_path in frames[1:]:
        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise FileNotFoundError(f"Could not read KITTI stereo pair at {timestamp}")
        disparity = stereo.compute(left, right).astype(np.float32) / 16.0
        kp, desc = detect_orb(left)
        matches = match_descriptors(prev_desc, desc)
        if len(matches) >= 6:
            p_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            p_curr = np.float32([kp[m.trainIdx].pt for m in matches])
            xyz, valid = pixels_to_3d(p_prev, prev_disp, K, baseline)
            object_points, image_points = xyz[valid], p_curr[valid]
            if len(object_points) >= 6:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points, image_points, K, None, iterationsCount=150,
                    reprojectionError=2.0, confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok and inliers is not None and len(inliers) >= 6:
                    R, _ = cv2.Rodrigues(rvec)
                    # PnP maps previous-camera 3D points into the current camera.
                    pose[:3, :3] = pose[:3, :3] @ R.T
                    pose[:3, 3] = pose[:3, 3] - pose[:3, :3] @ tvec.reshape(3)
        poses.append((timestamp, pose.copy()))
        points = _sample_cloud(disparity, K, baseline)
        if len(points):
            world = (pose[:3, :3] @ points.T).T + pose[:3, 3]
            cloud_world.append(world)
        prev_left, prev_right, prev_disp, prev_kp, prev_desc = left, right, disparity, kp, desc

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tum_path = output_prefix.with_name(output_prefix.name + "_trajectory.tum")
    csv_path = output_prefix.with_name(output_prefix.name + "_ego_motion.csv")
    ply_path = output_prefix.with_name(output_prefix.name + "_sparse_map.ply")
    with tum_path.open("w", encoding="utf-8") as f_tum, csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["timestamp", "x", "y", "z", "yaw"])
        for timestamp, p in poses:
            q = rotation_matrix_to_quaternion(p[:3, :3])
            x, y, z = p[:3, 3]
            f_tum.write(f"{timestamp:.9f} {x:.9f} {y:.9f} {z:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
            yaw = float(np.arctan2(p[1, 0], p[0, 0]))
            writer.writerow([f"{timestamp:.9f}", f"{x:.9f}", f"{y:.9f}", f"{z:.9f}", f"{yaw:.9f}"])
    points = np.vstack(cloud_world) if cloud_world else np.empty((0, 3))
    with ply_path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, points, fmt="%.5f")
    return len(poses)
