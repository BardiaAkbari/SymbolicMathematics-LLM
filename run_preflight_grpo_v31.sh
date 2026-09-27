#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python test_grpo_utils.py
python test_reward_v31_safety.py
python test_reward_v31_logged_regressions.py
echo "ALL GRPO/REWARD V3.1 PREFLIGHT TESTS PASSED"
