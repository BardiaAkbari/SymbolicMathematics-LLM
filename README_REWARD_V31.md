# Reward v3.1 — 50-failure validation

This is a **diagnostic pre-exact reward**, not the exact correctness verifier.
The exact symbolic residual + generality verifier remains unchanged.

## Why v3.1 exists

The 50-failure audit of Reward v3 exposed two repeatable GRPO-unsafe behaviors:

1. **Asymptotic hiding:** on positive x, one exact growing mode could numerically hide a wrong decaying mode. Wrong candidates such as roots `(22,-28)` for an exact `(22,-33)` problem could score almost 15.
2. **Duplicate-mode collapse:** two constants controlling the same exact mode could still score 11 because the numerical independence penalty was only additive.

Polynomial/trig cases also showed finite-domain approximants, so v3.1 probes negative, central, and positive regions.

## Reward v3.1

Three domains are used:

- `D- = [-4,-0.25]`
- `D0 = [-1,1]`
- `D+ = [0.25,4]`

Each domain contains fixed anchors plus randomized interior points. `probe_seed` makes a diagnostic run reproducible. **A future RL loop should change `probe_seed` every optimization step.**

For each domain score `q_j = clip(-log10(error_j + eps), -6, 12)`, v3.1 uses a soft worst-case aggregator:

`A(q) = 0.60 * min(q) + 0.40 * median(q)`

A hard min caught asymptotic hacks but created an artificial valley in a legitimate polynomial-completion path. The soft worst-case retains pressure from the bad domain while preserving useful progress.

### Inhomogeneous `L[y]=g`

`Q_force = A(domain forcing-normalized scores)`

For every parameter direction `d_i = dh/dc_i`, compute its homogeneous operator score across all domains. Bad directions are a penalty-only constraint:

`P_dir = max(0, 4 - min_i Q_dir_i)`

Ignoring validity/missing-constant penalties for notation:

`R31 = Q_force - P_dir`

### Homogeneous `L[y]=0`

There is **no global candidate-level q_equation term**.

For every parameter direction:

`Q_i = A(domain scores of L[d_i]=0)`

Also evaluate the parameter-independent base component

`b(x) = h(x,c=0)`

separately, so perfect parameter directions cannot hide a wrong additive offset.

`R31_hom = min(Q_1, Q_2, ..., Q_base)`

### Hard symbolic generality gate

For two parameter directions:

`W = d1*d2' - d2*d1'`

If all available direction pairs have `W == 0` symbolically, or fewer than two integration constants are present, reward is hard-capped at `-4`.

Numerical `G` is retained **for diagnostics only** and does not affect Reward v3.1.

## Built-in tests

Run:

```bash
python test_reward_v31_safety.py
python test_reward_v31_logged_regressions.py
```

The second test contains the seven strongest Reward-v3 failures found in the real 50-problem log. Expected examples:

- v3 ~15 asymptotic-hiding case -> v3.1 low positive (~1–3)
- v3 11 duplicate-mode cases -> v3.1 = -4 hard gate

## Full 50-problem comparison

Run from the same SymbolicMathematics repository/environment used for the previous validation:

```bash
bash run_validation50_v31.sh /path/to/ode2.pth
```

or directly:

```bash
python rank_reward_validation50_v31.py \
  --checkpoint /path/to/ode2.pth \
  --problems validation50_balanced_blind.jsonl \
  --samples 256 \
  --batch-size 128 \
  --temperature 1.0 \
  --top-k 10 \
  --verifier-workers 32 \
  --candidate-timeout 4 \
  --reward-timeout 6 \
  --seed 0 \
  --probe-seed 314159 \
  --output validation50_reward_v31_ranking.jsonl
```

The script samples each candidate set **once**, then scores the exact same candidates with Reward v3 and Reward v3.1. This makes ranking differences attributable to the reward rather than generation noise.

Send back:

- `validation50_reward_v31_ranking.jsonl`
- preferably `validation50_reward_v31_terminal.txt`

for the next manual mathematical audit.
