#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-data/tum}"
mkdir -p "$ROOT"
base="https://vision.in.tum.de/rgbd/dataset/freiburg1"
download_sequence() {
  local seq="$1" archive="$2" dest="$ROOT/$1"
  if [[ -d "$dest" && -f "$dest/rgb.txt" ]]; then
    echo "[skip] $seq already present"
    return
  fi
  local tmp="$ROOT/${seq}.tgz"
  echo "[download] $seq from $base/$archive"
  curl -fL --retry 3 --retry-delay 2 "$base/$archive" -o "$tmp"
  mkdir -p "$dest"
  tar -xzf "$tmp" -C "$dest" --strip-components=1
  rm "$tmp"
  test -f "$dest/rgb.txt"
}
download_sequence fr1_xyz rgbd_dataset_freiburg1_xyz.tgz
download_sequence fr1_desk rgbd_dataset_freiburg1_desk.tgz
echo "TUM sequences are available under $ROOT"
