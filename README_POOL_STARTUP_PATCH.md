# GRPO CPU-pool startup patch

Fixes a hang immediately after `trainable_parameters=...` caused by constructing a 16-worker `spawn` pool after the model/CUDA stack was initialized.

Changes:
- Linux CPU-only verifier/reward pools use `fork` (fallback: `spawn`).
- Pools are created **before any CUDA call/model loading**.
- Parent hard watchdog from the previous timeout patch is retained unchanged.
- Reward math, exact-verifier math, rollout budgets, and GRPO objective are unchanged.

Expected startup log:
```
starting_cpu_pools verifier_workers=16 reward_workers=1
cpu_pools_ready
device=cuda
trainable_parameters=...
```

Apply from repository root:
```
unzip -o ttrl_grpo_pool_startup_patch.zip
bash run_frozen4_control.sh ./ode2.pth
```
