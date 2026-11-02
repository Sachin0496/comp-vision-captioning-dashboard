#!/usr/bin/env bash
# End-to-end reproduction. Trains three small models on CPU; safe for a 16 GB Mac.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TORCH_HOME="$ROOT/.cache/torch"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
mkdir -p "$ROOT/logs"

if [[ ! -f data/raw/val2017.zip || ! -f data/raw/annotations_trainval2017.zip ]]; then
  bash scripts/00_download_data.sh
fi

"$PYTHON" scripts/01_prepare_data.py
"$PYTHON" scripts/02_extract_features.py
"$PYTHON" scripts/03_detect_objects.py

for model in global attention object; do
  "$PYTHON" scripts/04_train.py --model "$model"
done

"$PYTHON" scripts/05_evaluate.py
"$PYTHON" scripts/06_visualize.py
"$PYTHON" scripts/07_make_report.py

echo "Done. See report/report.pdf and outputs/metrics/."
