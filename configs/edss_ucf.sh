#!/usr/bin/env bash
# Reproduce the single-seed EDSS UCF-Crime recipe reported in the paper.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT"
exec "$PYTHON_BIN" -u src/ucf_train.py \
  --seed 234 \
  --tag edss_ucf \
  --log-mode w \
  --ebh-weight1 1.0 --ebh-weight2 0.0 \
  --ebh-alpha 0.30 --bet-eta 1.5 \
  --ebh-max-frac 0.15 --ebh-neg-frac 0.10 --ebh-normal-weight 0.20 \
  --ebh-warmup-epochs 1 \
  --smooth-weight 0.0008 --sparse-weight 0.0001 \
  --max-epoch 10 --test-every 10 \
  --early-stop-frac 0.20 --early-stop-patience-frac 0.40 \
  "$@"
