#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:?usage: bash run_grpo4_v31.sh /real/path/to/ode2.pth}"
python ttrl_ode_benchmark/run_grpo_v31.py \
  --checkpoint "$CKPT" \
  --problems grpo4_failures_blind.jsonl \
  --preexact-mode grpo \
  --warmup-samples 256 \
  --steps 30 \
  --rollouts 64 \
  --base-rollout-fraction 0.25 \
  --temperature 1.0 \
  --scope last_layer \
  --grpo-lr 1e-6 \
  --grpo-epochs 1 \
  --grpo-clip 0.20 \
  --kl-coef 0.02 \
  --entropy-coef 0.002 \
  --max-old-kl 0.08 \
  --reward-workers 16 \
  --verifier-workers 16 \
  --eval-every 5 \
  --eval-rollouts 256 \
  --seed 0 \
  --output grpo4_v31_results.jsonl \
  --summary grpo4_v31_summary.json | tee grpo4_v31_terminal.txt
