#!/usr/bin/env bash
# STAGE 3 — seeded patient- and nodule-level splits; exports split CSVs + leakage assertion.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.splits --config config.yaml
