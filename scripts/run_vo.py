#!/usr/bin/env python3
"""Run monocular visual odometry on a downloaded TUM sequence."""
from __future__ import annotations
import argparse
from pathlib import Path
from vo.pipeline import run_monocular_vo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", choices=["fr1_xyz", "fr1_desk"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/tum"))
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    sequence_dir = args.data_root / args.seq
    output = Path("results") / f"{args.seq}_estimated.txt"
    count = run_monocular_vo(sequence_dir, output, args.max_frames)
    print(f"Wrote {count} estimated poses to {output}")

if __name__ == "__main__":
    main()
