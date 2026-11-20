#!/usr/bin/env bash
# Regenerate every result and figure, in order.
#   bash run_all.sh
# Write and commit PREDICTIONS.md (and HYPOTHESIS_B.md) before the first run
# of Part C (and of the design choice): see the README.
set -euo pipefail
cd "$(dirname "$0")"

echo "== Known-answer tests =="
python -m pytest -q

echo "== Part A =="
python part_a.py

echo "== Part B: sweep =="
python part_b.py sweep

echo "== Part B: design choice (gradient through the denoiser or not) =="
python part_b.py design

echo "== Part C =="
python part_c.py all

echo "Every table and figure is in outputs/, and all of them together in outputs/RESULTS.md."
