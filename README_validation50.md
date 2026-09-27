# Reward v3 — 50 new strict-failure validation

Reward v3 is frozen and unchanged from the prior Hard-10 and validation-20 experiments.

## Selection
- Seed: `2026092250`
- Excluded: all Hard-10 IDs and all prior balanced validation-20 IDs.
- Strict failure: baseline exact=0, warmup exact=0, no exact in any train step, final held-out exact=0, final greedy wrong.
- 50 new cases: 20 homogeneous, 15 trig-forced, 15 polynomial-forced.
- No new strict exponential-forced cases remain after exclusions; the prior validation-20 already used all 3 remaining strict exp failures.

This is a reward stress test, not a population-weighted accuracy estimate.

## Run
Place this directory in the same repo/environment used for the previous reward-v3 run, where `ttrl_lane_emden`, `ttrl_ode_benchmark.common`, `ttrl_ode_benchmark.exact_verifier`, and `ode2.pth` are available.

```bash
python rank_reward_validation50_v3.py \
  --checkpoint ode2.pth \
  --problems validation50_balanced_blind.jsonl \
  --samples 256 \
  --batch-size 128 \
  --temperature 1.0 \
  --top-k 10 \
  --verifier-workers 32 \
  --candidate-timeout 4 \
  --seed 0 \
  --output validation50_reward_v3_ranking.jsonl
```

Then run:

```bash
python analyze_reward50.py validation50_reward_v3_ranking.jsonl
```

Send back **the complete `validation50_reward_v3_ranking.jsonl`** (preferred) or the complete terminal output. We will inspect every one of the 50 equations and their V3 top candidates manually.

## What this test is trying to falsify
1. Rank-collapse / duplicated-mode exploit gets high v3 reward.
2. Amplitude hiding lets a bad homogeneous direction disappear in the total residual.
3. Fixed positive-domain residual rewards finite-domain impostors.
4. Bad parameter directions remain positive after penalties.
5. Homogeneous root search is pulled toward a wrong local basin.
6. Top-k reward plateaus are too flat for useful GRPO advantages.
7. The same failure pattern recurs across several unrelated ODEs.

Do not change Reward v3 until this 50-set has been evaluated; otherwise the set is no longer a clean out-of-sample stress test.
