# Visual odometry evaluation

No dataset benchmark has been run in this repository environment. KITTI requires manual download of the stereo odometry data and ground-truth pose archive; TUM is available through the download script but has not been downloaded and evaluated here. No metric values or plots are fabricated.

| Dataset / sequence | ATE RMSE (m) | RPE RMSE (m) | Status |
|---|---:|---:|---|
| KITTI 00 | — | — | Not run yet — KITTI grayscale stereo sequence and ground-truth pose archive not downloaded |
| KITTI 01 | — | — | Not run yet — KITTI grayscale stereo sequence and ground-truth pose archive not downloaded |
| KITTI 02 | — | — | Not run yet — KITTI grayscale stereo sequence and ground-truth pose archive not downloaded |
| TUM fr1_xyz | — | — | Not run yet — TUM sequence not downloaded and VO/evo not executed |
| TUM fr1_desk | — | — | Not run yet — TUM sequence not downloaded and VO/evo not executed |

Run the KITTI evaluator after downloading data and generating estimates:

    python scripts/run_vo.py --dataset kitti --seq 00 --data-root data/kitti
    python scripts/evaluate.py --dataset kitti --seq 00 --data-root data/kitti

Successful runs generate ATE/RPE logs and plots under results/ and docs/. KITTI evaluation does not apply scale correction; TUM monocular evaluation uses global Sim(3) alignment.
