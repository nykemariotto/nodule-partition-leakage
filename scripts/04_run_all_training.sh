#!/usr/bin/env bash
# STAGE 4 — main paired grid: arch × seed × arm (dataset S2 first). SPEC §2.5.
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET=lidc_binary
ARCHS=(resnet50 densenet121 efficientnet_b0 swin_tiny)
SEEDS=(42 123 2024 7 99)
ARMS=(patient nodule)
for arch in "${ARCHS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
      python -m src.train --dataset "$DATASET" --arch "$arch" --arm "$arm" --seed "$seed" --config config.yaml
    done
  done
done
