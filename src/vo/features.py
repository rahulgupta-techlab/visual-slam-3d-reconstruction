"""ORB feature extraction and descriptor matching."""
from __future__ import annotations

import cv2
import numpy as np


def detect_orb(image: np.ndarray, nfeatures: int = 2500) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Detect ORB keypoints and compute descriptors for a grayscale image."""
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty array")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=nfeatures)
    return detector.detectAndCompute(gray, None)


def match_descriptors(
    desc1: np.ndarray | None,
    desc2: np.ndarray | None,
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    """Match ORB descriptors using Hamming KNN and Lowe's ratio test."""
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be between 0 and 1")
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(desc1, desc2, k=2)
    return [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
