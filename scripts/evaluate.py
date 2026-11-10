#!/usr/bin/env python3
"""Run evo evaluation and record only metrics produced by successful executions."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def run_metric(command: list[str]) -> tuple[float, str]:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    output = completed.stdout + "\n" + completed.stderr
    # evo prints the statistics table as "      rmse\t0.148051" (no unit).
    match = re.search(r"^\s*rmse\s+([0-9.eE+-]+)\s*$", output, re.IGNORECASE | re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse RMSE from evo output:\n{output}")
    return float(match.group(1)), output


def evaluate_kitti(sequence: str, data_root: Path) -> tuple[float, float]:
    gt = data_root / "poses" / f"{sequence}.txt"
    estimate = Path("results") / f"kitti_{sequence}_trajectory.kitti"
    if not gt.exists():
        raise FileNotFoundError(f"Missing KITTI ground truth poses: {gt} (download the KITTI odometry ground-truth archive)")
    if not estimate.exists():
        raise FileNotFoundError(f"Missing estimated trajectory: {estimate}; run scripts/run_vo.py --dataset kitti first")
    Path("docs").mkdir(exist_ok=True)
    common = ["--align", "--no_warnings"]
    ate, ate_log = run_metric(["evo_ape", "kitti", str(gt), str(estimate), *common,
        "--plot_mode", "xy", "--save_plot", f"docs/kitti_{sequence}_ate.png"])
    rpe, rpe_log = run_metric(["evo_rpe", "kitti", str(gt), str(estimate), *common,
        "--delta", "1", "--delta_unit", "f", "--plot_mode", "xy",
        "--save_plot", f"docs/kitti_{sequence}_rpe.png"])
    Path("results").mkdir(exist_ok=True)
    Path(f"results/kitti_{sequence}_ate.log").write_text(ate_log, encoding="utf-8")
    Path(f"results/kitti_{sequence}_rpe.log").write_text(rpe_log, encoding="utf-8")
    return ate, rpe


def evaluate_tum(sequence: str, data_root: Path) -> tuple[float, float]:
    sequence_dir = data_root / sequence
    gt, estimate = sequence_dir / "groundtruth.txt", Path("results") / f"{sequence}_estimated.txt"
    if not gt.exists():
        raise FileNotFoundError(f"Missing TUM ground truth: {gt}; run scripts/download_data.sh")
    if not estimate.exists():
        raise FileNotFoundError(f"Missing TUM estimate: {estimate}; run scripts/run_vo.py --dataset tum first")
    Path("docs").mkdir(exist_ok=True)
    common = ["--align", "--correct_scale", "--t_max_diff", "0.02", "--no_warnings"]
    ate, ate_log = run_metric(["evo_ape", "tum", str(gt), str(estimate), *common,
        "--plot_mode", "xy", "--save_plot", f"docs/{sequence}_ate.png"])
    rpe, rpe_log = run_metric(["evo_rpe", "tum", str(gt), str(estimate), *common,
        "--delta", "1", "--delta_unit", "f", "--plot_mode", "xy",
        "--save_plot", f"docs/{sequence}_rpe.png"])
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{sequence}_ate.log").write_text(ate_log, encoding="utf-8")
    Path(f"results/{sequence}_rpe.log").write_text(rpe_log, encoding="utf-8")
    return ate, rpe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["kitti", "tum"], default="kitti")
    parser.add_argument("--seq", choices=["00", "01", "02", "fr1_xyz", "fr1_desk", "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    is_kitti = args.dataset == "kitti"
    sequences = (["00", "01", "02"] if is_kitti else ["fr1_xyz", "fr1_desk"]) if args.seq == "all" else [args.seq]
    data_root = args.data_root or Path("data/kitti" if is_kitti else "data/tum")
    rows: list[tuple[str, float | None, float | None, str]] = []
    for sequence in sequences:
        try:
            if shutil.which("evo_ape") is None or shutil.which("evo_rpe") is None:
                raise FileNotFoundError("evo_ape/evo_rpe executables not found; install project dependencies with python -m pip install -e .")
            ate, rpe = evaluate_kitti(sequence, data_root) if is_kitti else evaluate_tum(sequence, data_root)
            rows.append((sequence, ate, rpe, "Completed"))
            print(f"{sequence}: ATE RMSE={ate:.6f} m, RPE RMSE={rpe:.6f} m")
        except (FileNotFoundError, subprocess.CalledProcessError, RuntimeError) as exc:
            reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            rows.append((sequence, None, None, f"Not run yet — {reason}"))
            print(f"{sequence}: Not run yet — {exc}")
    lines = ["# Visual odometry evaluation", "",
        f"Dataset: {'KITTI stereo (metric scale)' if is_kitti else 'TUM monocular fallback'}.",
        "KITTI uses metric stereo scale and no scale correction in evo. TUM uses one global Sim(3) alignment because monocular scale is unobservable.",
        "Plots and raw logs are written only for sequences whose evo runs complete successfully.", "",
        "| Sequence | ATE RMSE (m) | RPE RMSE (m) | Status |", "|---|---:|---:|---|"]
    for seq, ate, rpe, status in rows:
        lines.append(f"| {seq} | {ate:.6f} | {rpe:.6f} | {status} |" if ate is not None and rpe is not None
                     else f"| {seq} | — | — | {status} |")
    # One file per dataset, so evaluating TUM never overwrites KITTI results (and vice versa).
    Path("results").mkdir(exist_ok=True)
    output = Path("results") / f"metrics_{args.dataset}.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
