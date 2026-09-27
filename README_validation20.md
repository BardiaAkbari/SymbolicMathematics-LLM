# Reward v3 — unseen 20-failure validation before GRPO

This is an out-of-sample reward stress test. Reward v3 is unchanged from the Hard-10 experiment.

## Strict failure definition
Every selected problem satisfies all of the following in the original 600-run:
- frozen baseline exact = 0
- warmup exact = 0
- `first_exact_by = null`
- every training step has `train_exact = 0` and `new_exact = 0`
- final held-out exact = 0
- final greedy is wrong
- original Hard-10 IDs are excluded

So these are genuinely unseen failures where no exact trajectory was observed anywhere in the logged run/evaluations.

Seed: `20260922`.

## Primary set: structure-balanced random 20
`validation20_balanced_blind.jsonl`

The strict remaining pool contains:
- 259 homogeneous
- 50 trig-forced
- 33 polynomial-forced
- only 3 exponential-forced

To stress-test every part of Reward v3, the primary set randomly samples:
- 6 homogeneous
- 6 trig
- 5 polynomial
- all 3 remaining exponential strict failures

This is for reward validation, not population accuracy estimation.

## Secondary set: literal uniform-random 20
`validation20_uniform_blind.jsonl`

Use this optionally after the primary set. It reflects the benchmark's strong homogeneous skew.

## Run primary validation
```bash
python rank_reward_validation_v3.py \
  --checkpoint ode2.pth \
  --problems validation20_balanced_blind.jsonl \
  --samples 256 \
  --batch-size 128 \
  --temperature 1.0 \
  --top-k 10 \
  --verifier-workers 32 \
  --candidate-timeout 4 \
  --output validation20_reward_v3_ranking.jsonl
```

Send the complete terminal output back. We should inspect every equation and every V3 top candidate manually before implementing GRPO.

## Acceptance rubric before GRPO
1. Large-amplitude/normalization exploits are demoted.
2. In inhomogeneous ODEs, pure homogeneous solutions do not receive large positive reward.
3. Better particular-solution structure generally outranks worse particular structure when homogeneous directions are valid.
4. Bad parameter directions receive `Pd`; degenerate constant families receive `Pg`.
5. Legitimate stiff/two-mode solutions are not systematically destroyed by the gates.
6. No new recurring exploit dominates V3 TOP.
7. For homogeneous equations, the ranking should meaningfully favor candidates whose two solution directions better satisfy the operator.

If this holds on the unseen 20, proceed to pre-exact GRPO + Reward v3, switching to verified replay immediately after first exact discovery.
