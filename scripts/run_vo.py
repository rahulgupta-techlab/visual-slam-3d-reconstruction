#!/usr/bin/env python3
"""Run KITTI stereo metric VO or the retained TUM monocular fallback."""
from __future__ import annotations

import argparse
from pathlib import Path

from vo.kitti import run_kitti_stereo_vo
from vo.pipeline import run_monocular_vo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["kitti", "tum"], default="kitti")
    parser.add_argument("--seq", required=True, help="KITTI: 00, 01, 02; TUM: fr1_xyz or fr1_desk")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if args.dataset == "kitti":
        if args.seq not in {"00", "01", "02"}:
            parser.error("KITTI sequence must be 00, 01 or 02")
        root = args.data_root or Path("data/kitti")
        sequence_dir = root / "sequences" / args.seq
        prefix = Path("results") / f"kitti_{args.seq}"
        count = run_kitti_stereo_vo(sequence_dir, prefix, args.max_frames)
        print(f"Wrote {count} KITTI poses and ego-motion/sparse-map exports with prefix {prefix}")
    else:
        if args.seq not in {"fr1_xyz", "fr1_desk"}:
            parser.error("TUM sequence must be fr1_xyz or fr1_desk")
        root = args.data_root or Path("data/tum")
        sequence_dir = root / args.seq
        output = Path("results") / f"{args.seq}_estimated.txt"
        count = run_monocular_vo(sequence_dir, output, args.max_frames)
        print(f"Wrote {count} monocular poses to {output}")


if __name__ == "__main__":
    main()
