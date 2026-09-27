# Hard-timeout patch for TTRL GRPO v3.1

This overlay fixes indefinite stalls caused by pathological SymPy/NumPy candidates.

## What changed

- `ParallelVerifierPool` no longer uses blocking `Pool.map()` without a parent deadline.
- Exact-verifier work is dispatched in waves of at most `workers` candidates.
- Worker-local SIGALRM remains in place.
- A parent hard watchdog waits at most `candidate_timeout + 3s` per wave.
- If any task remains stuck, unfinished candidates are marked as timeout, the worker pool is terminated/recreated, and the run continues.
- `ParallelRewardV31Pool` gets the same protection for GRPO dense-reward scoring.
- RuntimeWarning spam from pathological generated expressions is suppressed inside workers.

## Install

From the SymbolicMathematics repository root:

```bash
unzip -o ttrl_grpo_hard_timeout_patch.zip
```

Then rerun:

```bash
bash run_frozen4_control.sh ./ode2.pth
```

and afterward:

```bash
bash run_grpo4_v31.sh ./ode2.pth
```

The mathematical verifier/reward definitions are unchanged; this patch changes only process scheduling, hard timeout enforcement, and warning verbosity.
