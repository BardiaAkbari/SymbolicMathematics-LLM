#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:-ode2.pth}"
DATASET="${2:-hse_ode300k}"

python ttrl_ode_benchmark/prepare_hse300k.py \
  --dataset-dir "$DATASET" \
  --output hse300k_ode2_verified.jsonl \
  --reject-log hse300k_ode2_rejects.jsonl \
  --max-accepted 1000 \
  --reference-timeout 4

python ttrl_ode_benchmark/run_coverage.py \
  --checkpoint "$CKPT" \
  --problems hse300k_ode2_verified.jsonl \
  --max-problems 100 --samples 128 --batch-size 64 --temperature 1.0 --verifier-workers 32

NRARE=$(wc -l < hse300k_ode2_rare_support.jsonl 2>/dev/null || echo 0)
echo "rare_support_count=$NRARE"
if [[ "$NRARE" -gt 0 ]]; then
  python ttrl_ode_benchmark/run_ttrl.py \
    --checkpoint "$CKPT" \
    --problems hse300k_ode2_rare_support.jsonl \
    --max-problems 10 --warmup-samples 512 --warmup-batch-size 64 --warmup-temperature 1.0 \
    --steps 15 --rollouts 64 --temperature 1.0 --eval-rollouts 256 --eval-every 5 \
    --scope last_layer --lr 3e-6 --replay-updates-new-exact 20 --replay-updates 2 --verifier-workers 32
else
  echo "No rare-support tasks found in this pilot. Do NOT run TTRL blindly."
  echo "Next: create representation-shift variants and rerun coverage (see README_HSE300K_TTRL.md)."
fi
