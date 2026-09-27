#!/usr/bin/env bash
set -euo pipefail
PROBLEMS=${1:-hse10_failures_blind.jsonl}
CKPT=${2:-ode2.pth}
SAMPLES=${SAMPLES:-4096}
WORKERS=${WORKERS:-32}
BATCH=${BATCH:-256}
for T in 0.7 1.0 1.3 1.6; do
  TAG=${T/./p}
  python ttrl_ode_benchmark/run_coverage.py \
    --checkpoint "$CKPT" \
    --problems "$PROBLEMS" \
    --max-problems 10 \
    --samples "$SAMPLES" \
    --batch-size "$BATCH" \
    --temperature "$T" \
    --candidate-timeout 4 \
    --verifier-workers "$WORKERS" \
    --output "hard10_frozen_T${TAG}.jsonl" \
    --rare-output "hard10_frozen_T${TAG}_rare.jsonl"
done
