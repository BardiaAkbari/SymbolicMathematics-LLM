# Hard-10 rich reward ranking diagnostic

This is a **diagnostic only**. It does not update the policy.

It compares the old verifier reward against a new pre-exact mathematical shaping score on exactly the same sampled candidates.

New pre-exact score:

```
R_pre = Q_equation + 0.35 * Q_constant_direction + 2.0 * G_independence - 5.0 * (1 - validity)
```

- `Q_equation`: unsaturated `-log10(error)` (cap 12, not 6).
  - Inhomogeneous linear ODE `L[y]=g`: `mean |L[h]-g|^2 / mean |g|^2`.
    Candidate amplitude/homogeneous terms cannot inflate the denominator.
  - Homogeneous ODE: scale-invariant operator residual.
- `Q_constant_direction`: for every used integration constant `c`, check whether `d h / d c` lies in the nullspace of the homogeneous operator. This rewards correct homogeneous modes even when the particular solution is still wrong.
- `G_independence`: continuous normalized determinant of `(h_c, h'_c)` for the best pair of constants, in `[0,1]`.
- `validity`: fraction of finite-real numerical probes.

The existing exact symbolic verifier is untouched and remains the correctness gate.

## Install over the current v3 multiprocess code

```bash
unzip -o hse_hard10_reward_v2.zip
```

## Run

Start with 256 samples per problem:

```bash
python rank_reward_hard10.py \
  --checkpoint ode2.pth \
  --problems hse10_failures_blind.jsonl \
  --samples 256 \
  --batch-size 128 \
  --temperature 1.0 \
  --top-k 10 \
  --verifier-workers 32 \
  --candidate-timeout 4 \
  --output hard10_reward_ranking.jsonl
```

For stronger statistics use `--samples 512`.

The output prints the top-10 candidates under **OLD TOP** and **NEW TOP** for every ODE, with:

- old reward
- new pre-exact reward
- equation-fit score
- constant-direction score
- continuous independence
- validity
- exact flag

Do not run GRPO yet. First inspect whether NEW TOP contains more mathematically meaningful partial solutions than OLD TOP.

## Smoke test first

```bash
python smoke_test_rich_reward.py
```

Expected ordering:

```text
exact > homogeneous_only > wrong_family
RICH REWARD SMOKE TEST PASSED
```
