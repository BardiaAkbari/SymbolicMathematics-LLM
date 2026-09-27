#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:?usage: bash run_validation50_v31.sh /path/to/ode2.pth}"
python test_reward_v31_safety.py
python test_reward_v31_logged_regressions.py
python rank_reward_validation50_v31.py \
  --checkpoint "$CKPT" \
  --problems validation50_balanced_blind.jsonl \
  --samples 256 --batch-size 128 --temperature 1.0 --top-k 10 \
  --verifier-workers 32 --candidate-timeout 4 --reward-timeout 6 \
  --seed 0 --probe-seed 314159 \
  --output validation50_reward_v31_ranking.jsonl \
  | tee validation50_reward_v31_terminal.txt
python analyze_reward_v31.py validation50_reward_v31_ranking.jsonl
