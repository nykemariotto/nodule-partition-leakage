#!/usr/bin/env bash
# STAGE 8 — ROC, confusion matrices, train/val curves, real gap/forest plot.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.figures --config config.yaml
