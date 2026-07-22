#!/usr/bin/env bash
# STAGE 1 — data completeness check + master metadata table.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/verify_download.py --config config.yaml
python -m src.metadata --config config.yaml
