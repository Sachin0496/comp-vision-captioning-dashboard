#!/usr/bin/env bash
# Download the COCO 2017 validation images (5k) and annotations.
# The project deliberately uses val2017 so the full experiment fits on a laptop.
# https is avoided because images.cocodataset.org currently serves a certificate
# that does not match the hostname; the official HTTP mirror is used instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/data/raw"

download() {
  local url="$1"
  local out="$2"
  if [[ -s "$out" ]]; then
    echo "Already exists: $out"
    return
  fi
  echo "Downloading $url"
  curl -L --fail --retry 3 --retry-delay 5 -C - -o "$out" "$url"
}

download "http://images.cocodataset.org/zips/val2017.zip" "$ROOT/data/raw/val2017.zip"
download "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$ROOT/data/raw/annotations_trainval2017.zip"

echo "Downloads complete. Run scripts/01_prepare_data.py next."
