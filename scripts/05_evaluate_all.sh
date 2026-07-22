#!/usr/bin/env bash
# STAGE 6 — metrics + AUC + bootstrap CIs + confusion matrices from saved probs; ensemble.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.evaluate --config config.yaml
python -m src.ensemble --config config.yaml
