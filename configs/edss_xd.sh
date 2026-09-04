#!/usr/bin/env bash
# Reproduce the single-seed EDSS XD-Violence recipe reported in the paper.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT"
exec "$PYTHON_BIN" -u src/xd_train.py \
  --seed 234 \
  --tag edss_xd \
  --log-mode w \
  --ebh-weight1 0.0 --ebh-weight2 0.5 \
  --ebh-alpha 0.05 --bet-eta 1.5 \
  --ebh-max-frac 0.25 --ebh-neg-frac 0.0 --ebh-normal-weight 0.50 \
  --ebh-warmup-epochs 1 \
  --max-epoch 5 --test-every 0 \
  --early-stop-frac 0.20 --early-stop-patience-frac 0.40 \
  "$@"
