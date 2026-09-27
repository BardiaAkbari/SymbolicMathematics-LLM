# Reward-v3.1 Pre-Exact GRPO for Symbolic ODE TTRL

This package adds the first **learning-before-exact-discovery** stage to the existing verifier-guided ODE TTRL pipeline.

## Scientific design

The exact symbolic verifier is unchanged. No ground-truth/reference solution is used for training.

The runner has three phases:

1. **Frozen warmup search** (optional, default 256 samples).
2. **Pre-exact discovery:** Reward-v3.1 + GRPO.
3. **Post-exact consolidation:** as soon as any exact trajectory is found, dense Reward-v3.1 is stopped and the runner switches to the existing exact-verifier RL + persistent exact replay mechanism.

The critical design rule is:

> A batch containing any exact candidate receives **no dense-reward GRPO update**. Exact discovery immediately triggers verified replay.

## Pre-exact rollout mixture

Default total rollouts per step: 64.

- 48 (`75%`) are sampled from the currently adapted policy.
- 16 (`25%`) are sampled from the original frozen pretrained policy.

The frozen-base rollouts are used for **search / exact discovery only**. They are intentionally **not inserted into the GRPO policy-gradient loss**, because doing so would be an off-policy update without proper importance correction.

If a frozen-base sample discovers an exact solution, that exact trajectory is immediately placed in verified replay and can train the current policy safely.

## GRPO objective

Reward-v3.1 produces one scalar dense reward `R_i` for each **policy-generated** trajectory in a pre-exact group.

Group-normalized advantages:

```
A_i = (R_i - mean(R)) / (std(R) + eps)
```

The policy update is token-level clipped GRPO/PPO:

```
r_it = exp(log pi_theta(a_it|s_it) - log pi_old(a_it|s_it))
L_pg = -mean_i mean_t min(r_it A_i, clip(r_it,1-eps,1+eps) A_i)
```

Default clipping epsilon: `0.20`.

A token-level sampled k3 KL regularizer anchors the adapted policy to the **original pretrained policy**, and entropy is kept in the graph:

```
L = L_pg + beta * KL_k3(pi_theta || pi_base) - eta * H(pi_theta)
```

Defaults:

- `beta = 0.02`
- `eta = 0.002`
- one GRPO epoch per fresh rollout group
- `grpo_lr = 1e-6`
- `scope = last_layer`

Using one epoch is deliberate for the first experiment: it keeps the method close to on-policy and avoids over-optimizing one noisy symbolic group.

## Reward-v3.1

The package contains the exact tested Reward-v3.1 implementation.

Key properties:

- negative / central / positive-domain probes;
- randomized probe interiors, changed every GRPO step;
- soft worst-domain aggregation;
- independent parameter-direction scoring;
- no global candidate residual in homogeneous ODE reward;
- homogeneous base-offset check;
- symbolic Wronskian generality gate;
- numerically conditioned independence is diagnostic only, not a reward penalty.

## Preflight tests

From the root of the SymbolicMathematics repository, after extracting this zip:

```bash
bash run_preflight_grpo_v31.sh
```

It checks:

- policy gradient actually changes gradients;
- entropy contributes gradients independently;
- KL contributes gradients independently;
- Reward-v3.1 progression paths;
- seven previously observed Reward-v3 exploits remain fixed.

## First experiment: four diverse hard failures

The recommended first experiment uses:

1. `test:367` — homogeneous hyperbolic near-root case;
2. `test:60` — homogeneous nearby exponential-root case;
3. `test:1264` — polynomial particular-solution case;
4. `test:1516` — trig particular-solution case.

Run the compute-matched frozen control first:

```bash
bash run_frozen4_control.sh /REAL/PATH/TO/ode2.pth
```

Then GRPO:

```bash
bash run_grpo4_v31.sh /REAL/PATH/TO/ode2.pth
```

Compare:

```bash
python compare_grpo_frozen.py \
  --frozen frozen4_control_summary.json \
  --grpo grpo4_v31_summary.json
```

Both use exactly:

- 256 frozen warmup samples;
- 30 x 64 = 1920 search samples afterward;
- total search budget 2176 candidates/problem;
- identical exact verifier and held-out evaluation settings.

The primary endpoint is **first exact discovery**, not final held-out accuracy.

Important metrics:

- `first_exact_by_search_calls`
- `first_exact_source` (`policy_pre`, `base_pre`, or warmup)
- `n_discovered`
- `discovery_rate`
- `final_greedy_exact`

## Then scale to Hard-10

```bash
bash run_frozen10_control.sh /REAL/PATH/TO/ode2.pth
bash run_grpo10_v31.sh /REAL/PATH/TO/ode2.pth
```

Use the same comparison script with the corresponding summary files.

## Important log fields

For every pre-exact GRPO step the JSONL stores:

- policy and frozen-base rollout counts;
- exact discoveries by source;
- Reward-v3.1 mean/std/min/max;
- fresh `probe_seed`;
- number of symbolic generality-gate failures;
- token-level policy loss;
- base-policy KL;
- entropy;
- PPO/GRPO clip fraction;
- gradient norm;
- sampled old-policy KL after update;
- top three Reward-v3.1 candidates.

This makes reward collapse, policy collapse, or excessively large updates visible rather than implicit.

## Recommended initial interpretation

A successful first result is **not** merely higher Reward-v3.1. The meaningful event is:

```
no exact under matched frozen search
        -> exact discovered under pre-exact GRPO
        -> verified replay amplifies it
        -> greedy becomes exact
```

If Reward-v3.1 rises but exact discovery does not improve, inspect whether the model is trapped in a symbolic family / basin rather than immediately increasing GRPO strength.
