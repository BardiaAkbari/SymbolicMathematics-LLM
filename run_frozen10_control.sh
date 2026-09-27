#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:?usage: bash run_frozen10_control.sh /real/path/to/ode2.pth}"
python ttrl_ode_benchmark/run_grpo_v31.py \
  --checkpoint "$CKPT" \
  --problems hse10_failures_blind.jsonl \
  --preexact-mode frozen \
  --warmup-samples 256 \
  --steps 30 \
  --rollouts 64 \
  --temperature 1.0 \
  --scope last_layer \
  --verifier-workers 16 \
  --reward-workers 1 \
  --eval-every 5 \
  --eval-rollouts 256 \
  --seed 0 \
  --output frozen10_control_results.jsonl \
  --summary frozen10_control_summary.json | tee frozen10_control_terminal.txt
