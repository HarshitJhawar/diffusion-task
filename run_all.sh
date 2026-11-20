#!/usr/bin/env bash
#   bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

python -m pytest -q

echo "Part A"
python part_a.py

echo "Part B: sweep"
python part_b.py sweep

echo "Part B: design choice"
python part_b.py design

echo "Part C"
python part_c.py

echo "Every table and figure is in outputs/, and all of them together in outputs/RESULTS.md."
