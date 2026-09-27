# HSE Hard-10 Reward v3 diagnostic

Reward v3 is still **diagnostic only**. Do not enable GRPO yet.

## Why v3

Reward v2 improved forcing-aware scoring, but manual inspection found three issues:

1. Correct homogeneous directions received a huge positive bonus even when the particular solution was missing.
2. The median local independence metric falsely treated stiff hyperbolic two-mode solutions as degenerate.
3. For homogeneous ODEs, one good mode could partially hide one bad mode.

## Reward v3

### Inhomogeneous `L[y]=g`

```text
R3 = Q_force
     - P_bad_direction
     - P_degenerate_or_missing_constants
     - P_invalid
```

`Q_force = -log10(mean|L[h]-g|^2 / mean|g|^2)` (cap 12).

Correct homogeneous directions are now a **constraint**, not a positive bonus. If the worst coefficient direction has `Q_dir >= 4`, it receives no extra reward; if it is worse, it is penalized.

### Homogeneous `L[y]=0`

```text
R3 = min_i Q( L[dh/dc_i] ) + 0.25*Q(L[h])
     - P_degenerate_or_missing_constants
     - P_invalid
```

The worst coefficient direction controls the score. This prevents one exact mode from hiding a bad second mode.

### Independence

Independence is now a **penalty-only gate**, evaluated on small/near-origin probes and using a high quantile of the normalized Wronskian. It is not a positive reward. This avoids the v2 failure on stiff `exp(-6x) cosh(a + kx)` parameterizations.

Exact symbolic verification remains unchanged and separate.

## Install

```bash
unzip -o hse_hard10_reward_v3.zip
```

## First run the mathematical path test

```bash
python test_reward_v3_paths.py
```

Expected: all four paths increase monotonically and end with:

```text
REWARD V3 PATH TESTS PASSED
```

The tested paths are:

- H60: second exponent `-35 -> -34 -> -33` (exact)
- H367: hyperbolic frequency `36 -> 35 -> 34` (exact)
- H1264: progressively complete the cubic particular solution
- H1516: progressively complete the trig particular solution
- degeneracy gate: one disguised mode must rank below a true two-mode solution

## Then rank the same Hard-10 model samples

```bash
python rank_reward_hard10_v3.py \
  --checkpoint ode2.pth \
  --problems hse10_failures_blind.jsonl \
  --samples 256 \
  --batch-size 128 \
  --temperature 1.0 \
  --top-k 10 \
  --verifier-workers 32 \
  --candidate-timeout 4 \
  --output hard10_reward_v3_ranking.jsonl
```

Send the complete terminal output back for manual inspection before we put GRPO on top.
