#!/usr/bin/env bash
# STAGE 7 — gap±CI, Wilcoxon (A vs B across seeds), McNemar (within-arm).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.stats --config config.yaml
