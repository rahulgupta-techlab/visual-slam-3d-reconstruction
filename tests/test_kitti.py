from pathlib import Path

import numpy as np
import pytest

from vo.kitti import load_kitti_sequence, pixels_to_3d


def test_kitti_loader_reads_rectified_stereo_calibration_and_times(tmp_path: Path):
    seq = tmp_path / "00"
    (seq / "image_0").mkdir(parents=True)
    (seq / "image_1").mkdir()
    (seq / "image_0" / "000000.png").write_bytes(b"")
    (seq / "image_1" / "000000.png").write_bytes(b"")
    (seq / "times.txt").write_text("0.0\n", encoding="utf-8")
    (seq / "calib.txt").write_text(
        "P0: 700 0 600 0 0 700 180 0 0 0 1 0\n"
        "P1: 700 0 600 -378 0 700 180 0 0 0 1 0\n",
        encoding="utf-8",
    )
    frames, K, baseline = load_kitti_sequence(seq)
    assert len(frames) == 1
    assert frames[0][0] == 0.0
    assert K[0, 0] == 700
    assert baseline == pytest.approx(0.54)


def test_stereo_disparity_backprojection_uses_metric_baseline():
    K = np.array([[700.0, 0, 600.0], [0, 700.0, 180.0], [0, 0, 1.0]])
    disparity = np.full((20, 30), 10.0, dtype=np.float32)
    xyz, valid = pixels_to_3d(np.array([[600.0, 180.0], [601.0, 180.0]]), disparity, K, 0.54)
    assert valid.all()
    np.testing.assert_allclose(xyz[0], [0.0, 0.0, 37.8], atol=1e-5)
    assert xyz[1, 0] > 0
