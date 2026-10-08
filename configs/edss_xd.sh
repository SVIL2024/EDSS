#!/usr/bin/env bash
# Reproduce the single-seed EDSS XD-Violence recipe reported in the paper.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

cd "$ROOT"
exec "$PYTHON_BIN" -u src/xd_train.py \
  --seed 234 \
  --tag edss_xd \
  --log-mode w \
  --lr 1e-5 \
  --ebh-weight1 0.0 --ebh-weight2 0.5 \
  --ebh-alpha 0.075 --bet-eta 1.5 \
  --ebh-max-frac 0.25 --ebh-neg-frac 0.0 \
  --ebh-normal-weight 0.75 --ebh-context-weight 0.0 \
  --ebh-warmup-epochs 0 \
  --max-epoch 10 --test-every 50 \
  --early-stop-patience 0 --early-stop-patience-frac 0 \
  --dump-run-best model/runbest_xd.pth \
  "$@"
