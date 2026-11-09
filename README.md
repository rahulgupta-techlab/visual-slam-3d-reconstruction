# visual-odometry-driving

This project estimates vehicle ego-motion and a sparse 3D map from camera images. It is designed as a perception and localization module for my autonomous-driving work in [predictive-world-model-driving](https://github.com/rahulgupta-techlab/predictive-world-model-driving), which consumes an ego-motion estimate as an input to its predictive world-model and planning pipeline.

The primary benchmark is the **KITTI Odometry dataset**, using rectified stereo image pairs from sequences 00–02. Stereo disparity provides metric depth and metric translation estimates; sparse feature correspondences are tracked between consecutive left-camera frames. TUM RGB-D remains available as an automatically downloadable monocular fallback baseline.

## Project status

- [x] Existing ORB feature extraction, matching, essential-matrix pose estimation and monocular trajectory integration retained.
- [x] KITTI odometry sequence loader for sequences 00–02.
- [x] Stereo depth from rectified KITTI image pairs and metric PnP motion estimation.
- [x] Sparse point-cloud export (PLY) and ego-motion CSV export.
- [x] TUM RGB-D automatic fallback loader and monocular VO path retained.
- [ ] KITTI sequences 00–02 benchmark runs and evo metrics: **Not run yet** — KITTI files and ground-truth pose files must be downloaded locally first.
- [ ] TUM fallback benchmark and evo metrics: **Not run yet** — not executed in the current environment.
- [ ] Dense reconstruction, loop closure and global bundle adjustment: **Not implemented**.

## Datasets

### Primary: KITTI Odometry (manual download)

Download the following from the [KITTI Vision Benchmark Suite – Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php):

1. **Odometry data set (grayscale, 22 GB)** — extract so that sequence folders are under data/kitti/sequences/.
2. **Odometry ground truth poses (4 MB)** — extract so that pose files are under data/kitti/poses/.

Expected layout:

    data/kitti/
    ├── sequences/
    │   ├── 00/
    │   │   ├── image_0/
    │   │   ├── image_1/
    │   │   ├── calib.txt
    │   │   └── times.txt
    │   ├── 01/
    │   ├── 02/
    │   └── ...
    └── poses/
        ├── 00.txt
        ├── 01.txt
        └── 02.txt

The loader reads grayscale stereo pairs, calibration and timestamps from the local sequence directory. It does not download KITTI automatically because the dataset requires manual acceptance of its terms.

### Fallback: TUM RGB-D (automatic download)

If KITTI is not available, download the two TUM RGB-D Freiburg 1 sequences:

    bash scripts/download_data.sh

The fallback uses the retained monocular ORB/essential-matrix implementation. Its translation scale is arbitrary; it is **not** metric-scale odometry.

## Installation

Python 3.10–3.12:

    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    python -m pip install -e '.[dev]'

## Run KITTI stereo visual odometry

    python scripts/run_vo.py --dataset kitti --seq 00 --data-root data/kitti

Available sequences: 00, 01, 02. The run writes:

- results/kitti_00_ego_motion.csv — timestamp,x,y,z,yaw in the world frame.
- results/kitti_00_trajectory.tum — TUM trajectory format for evo.
- results/kitti_00_sparse_map.ply — accumulated sparse 3D landmarks in the world frame.

Run the evaluation after generating the trajectory:

    python scripts/evaluate.py --dataset kitti --seq 00 --data-root data/kitti

KITTI ground truth is expected at data/kitti/poses/00.txt. Evaluation uses the sequence ground-truth poses and the generated trajectory. The code does not manufacture missing metrics: if data, trajectories or dependencies are absent, the result is recorded as **Not run yet** with the reason.

## Run the TUM fallback

    python scripts/run_vo.py --dataset tum --seq fr1_xyz
    python scripts/evaluate.py --dataset tum --seq fr1_xyz

Use fr1_desk for the second fallback sequence. TUM runs export the same ego-motion CSV interface, but estimated translation is up to arbitrary monocular scale.

## How this connects to predictive-world-model-driving

[visual-odometry-driving](https://github.com/rahulgupta-techlab/visual-odometry-driving) is responsible for estimating camera ego-motion and producing a sparse geometric map from stereo images. It exports a stable per-frame CSV with timestamp, world-frame position (x, y, z) and heading (yaw).

[predictive-world-model-driving](https://github.com/rahulgupta-techlab/predictive-world-model-driving) can consume this CSV as its ego-motion input rather than estimating vehicle motion internally. Keep timestamps in seconds and ensure both projects use the same coordinate-frame convention before combining trajectories, object tracks or predicted future states. The CSV export uses KITTI camera/world coordinates as defined by the odometry pose integration; axis conversion, if needed by the integration, should be explicit at the boundary.

## Method

1. Detect ORB features in consecutive left-camera frames.
2. Match descriptors using Hamming KNN and Lowe's ratio test.
3. Compute stereo disparity with Semi-Global Block Matching.
4. Back-project matched previous-frame pixels into 3D using the calibrated stereo baseline.
5. Estimate metric camera motion with RANSAC PnP from previous-frame 3D points to current-frame 2D observations.
6. Integrate camera poses and transform sampled stereo landmarks into the world frame.
7. Export ego-motion CSV, TUM trajectory and PLY point cloud.

The monocular TUM fallback continues to use an essential matrix and recoverPose, with unit translation direction and scale-aligned evo evaluation only.

## Evaluation policy

All reported values must come from a completed run of the checked-in code on the named dataset. KITTI ATE/RPE metrics and plots are **Not run yet** until the local KITTI sequences and ground-truth poses are present. No benchmark scores are inferred from synthetic unit tests. See results/metrics.md for the current run status.

## Limitations

- Stereo depth quality depends on calibration, texture, disparity range and occlusion.
- Sparse ORB landmarks are not a dense reconstruction.
- No loop closure, global bundle adjustment, relocalisation or dynamic-object filtering.
- KITTI evaluation covers only the sequences actually downloaded and run.
- The TUM fallback is monocular and therefore does not provide metric scale.

## License

MIT. KITTI and TUM datasets remain subject to their respective terms of use.
