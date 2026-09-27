#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:-ode2.pth}"
python rank_reward_validation50_v3.py \
  --checkpoint "$CKPT" \
  --problems validation50_balanced_blind.jsonl \
  --samples 256 \
  --batch-size 128 \
  --temperature 1.0 \
  --top-k 10 \
  --verifier-workers 32 \
  --candidate-timeout 4 \
  --seed 0 \
  --output validation50_reward_v3_ranking.jsonl
python analyze_reward50.py validation50_reward_v3_ranking.jsonl
