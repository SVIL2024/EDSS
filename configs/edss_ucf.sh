#!/usr/bin/env bash
# Reproduce the single-seed EDSS UCF-Crime recipe reported in the paper.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

cd "$ROOT"
exec "$PYTHON_BIN" -u src/ucf_train.py \
  --seed 234 \
  --tag edss_ucf \
  --log-mode w \
  --lr 2e-5 \
  --ebh-weight1 1.0 --ebh-weight2 0.0 \
  --ebh-alpha 0.10 --bet-eta 1.125 \
  --ebh-max-frac 0.20 --ebh-neg-frac 0.10 \
  --ebh-normal-weight 0.20 --ebh-context-weight 0.20 \
  --ebh-warmup-epochs 2 \
  --max-epoch 10 --test-every 10 \
  --early-stop-patience 0 --early-stop-patience-frac 0 \
  --dump-run-best model/runbest_ucf.pth \
  "$@"
