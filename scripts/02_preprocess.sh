#!/usr/bin/env bash
# STAGE 2 — deterministic preprocessing (HU window, consensus ROI, resize 256²).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.preprocess --config config.yaml
