# HSE 300k ODE × Verifier-Guided TTRL Pilot

This package plugs into the SymbolicMathematics repository after the Lane–Emden v7 code is installed.
It tests whether the same **search → exact mathematical verification → test-time adaptation → verified replay**
mechanism generalizes beyond Lane–Emden.

## Why ODE2 first

The HSE 300k corpus contains generated first-, second-, and third-order ODEs. The first pilot intentionally
uses the second-order subset because it maps directly to the public Lample–Charton `ode2.pth` checkpoint.
The HSE paper reports 9,405 second-order homogeneous and 15,751 second-order inhomogeneous generated equations.

The HSE reference answer is used **only during preprocessing** to reject malformed / incorrect dataset rows.
During coverage and TTRL, the reference answer is never used. A model output succeeds only when:

1. substitution into the ODE gives an exact symbolic zero residual; and
2. the candidate has two independent integration constants (generic jet-Jacobian rank two plus exact check).

This is deliberately stronger than BLEU/TeXBLEU string similarity.

## 0. Install

From the root of your existing `SymbolicMathematics` checkout:

```bash
unzip -o hse300k_ttrl_ode2_pilot.zip
pip install -r requirements_hse300k.txt
python ttrl_ode_benchmark/smoke_test_hse_adapter.py
```

Expected final line:

```text
HSE ADAPTER SMOKE TEST PASSED
```

The smoke test contains real-format examples from the authors' earlier public corpus and verifies that an
incorrect published reference pair is rejected by substitution.

## 1. Download the HSE 300k repository

```bash
bash download_hse300k.sh hse_ode300k
```

Equivalent manual command:

```bash
git clone --depth 1 https://github.com/hse-scila/dif_equations_fine_tune.git hse_ode300k
```

No dataset bytes are redistributed in this package.

## 2. Build a verified ODE2 pilot subset

Start with 1,000 verified compatible equations rather than parsing all 300k:

```bash
python ttrl_ode_benchmark/prepare_hse300k.py \
  --dataset-dir hse_ode300k \
  --output hse300k_ode2_verified.jsonl \
  --reject-log hse300k_ode2_rejects.jsonl \
  --max-accepted 1000 \
  --reference-timeout 4
```

The loader recursively discovers **XLSX/CSV/JSON/JSONL** files and recognizes common `equation` / `answer` schemas. The current public HSE repository ships `data/train.xlsx` and `data/test.xlsx`; XLSX files are streamed with `openpyxl` so the full corpus is not loaded into memory. On startup the loader prints the detected Excel headers and the selected equation/answer/category columns.
Rows are rejected if they are not scalar ODE2, contain IC/BC constraints, cannot be represented in the
`ode2.pth` vocabulary, cannot be parsed, or their reference solution fails exact mathematical verification.

## 3. Frozen coverage scan

This identifies the scientifically interesting **rare-support** regime: greedy is wrong, but at least one exact
verified general solution appears in stochastic frozen sampling.

```bash
python ttrl_ode_benchmark/run_coverage.py \
  --checkpoint ode2.pth \
  --problems hse300k_ode2_verified.jsonl \
  --max-problems 100 \
  --samples 128 \
  --batch-size 64 \
  --temperature 1.0 \
  --candidate-timeout 4 \
  --verifier-workers 32
```

Outputs:

- `hse300k_ode2_coverage.jsonl` — all problems and frozen statistics
- `hse300k_ode2_rare_support.jsonl` — only greedy-fail / stochastic-exact problems

The exact trajectory found during coverage is deliberately removed from the rare-support input file. The HSE answer/reference fields are also physically stripped. TTRL therefore receives only the ODE plus coverage metadata and must rediscover a correct trajectory under its own counted search budget.

## 4. Run the first multi-equation TTRL pilot

```bash
python ttrl_ode_benchmark/run_ttrl.py \
  --checkpoint ode2.pth \
  --problems hse300k_ode2_rare_support.jsonl \
  --max-problems 10 \
  --warmup-samples 512 \
  --warmup-batch-size 64 \
  --warmup-temperature 1.0 \
  --steps 15 \
  --rollouts 64 \
  --temperature 1.0 \
  --eval-rollouts 256 \
  --eval-every 5 \
  --scope last_layer \
  --lr 3e-6 \
  --replay-updates-new-exact 20 \
  --replay-updates 2 \
  --candidate-timeout 4 \
  --verifier-workers 32
```

Per problem the adaptation/search candidate budget is:

```text
512 + 15*64 = 1472 generated candidates
```

Held-out evaluation generations are tracked separately. The JSON logs both generated candidate samples and unique verifier evaluations (cache misses), so compute-matched baselines can be defined unambiguously. Each problem starts from exactly the original pretrained
trainable weights; adaptation from one equation never leaks into the next equation.

The algorithm is intentionally **search-then-adapt**. Until the first exact verified trajectory is found, the
policy stays frozen instead of reinforcing mediocre approximations. Once an exact trajectory is found, group-relative
REINFORCE plus exact verified replay is enabled. Exact replay candidates are capped (default 32) to avoid the large
replay-buffer slowdown seen in the Lane–Emden n=2 experiment.

Outputs:

- `hse300k_ttrl_results.jsonl`
- `hse300k_ttrl_summary.json`

Primary pilot metric: mean held-out exact-general success probability before vs after adaptation, plus greedy-exact rate.


## Parallel CPU verification

The exact verifier can now score generated candidates concurrently across CPU **processes**. This is the only algorithmic/runtime change in this patch; reward functions, exact checks, replay, search budgets, and model inference are unchanged.

- `--verifier-workers 0` (default): auto-select up to 32 logical CPUs.
- `--verifier-workers 1`: original serial verifier.
- `--verifier-workers 32`: explicitly use 32 verifier processes.

The pool uses the `spawn` start method so it is safe to create after CUDA initialization, and one pool is reused across all equations. Only uncached unique candidates are dispatched to workers.

## 5. If the raw HSE ODE2 subset is too easy: representation-shift benchmark

Create equivalent real-domain equations without changing the solution set:

```bash
python ttrl_ode_benchmark/make_representation_shift.py \
  --problems hse300k_ode2_verified.jsonl \
  --output hse300k_ode2_representation_shift.jsonl \
  --max-problems 500 \
  --variants original,neg,scale7,mul_exp,mul_quad,div_quad
```

Then simply point `run_coverage.py` at `hse300k_ode2_representation_shift.jsonl`.

The non-scalar transformations use factors that are nonzero for all real `x`:

- `exp(x)`
- `x^2 + 1`
- `1/(x^2 + 1)`

so they preserve the real solution set and avoid the singularity caveat of multiplying by `x`.

## Recommended paper-quality progression

1. Pilot: 100 frozen coverage problems, then 10 rare-support TTRL tasks.
2. Debug parser/filter failure categories using `hse300k_ode2_rejects.jsonl`.
3. Scale coverage to 1,000+ verified ODE2 problems.
4. Evaluate 50–200 rare-support tasks over multiple random seeds.
5. Compare matched candidate/verifier budgets against frozen Best-of-N, residual reranking, replay-only, RL-only,
   and the full search-then-adapt method.
6. Repeat on representation-shift variants and report robustness by transformation family.

## Important methodological note

Unlike the exploratory Lane–Emden n=2 IVP code, this HSE exact-ODE2 benchmark has **no numerical-reference reward**.
After preprocessing, training/adaptation is driven only by the differential equation itself and the mathematical
completeness check. This is the cleaner setting for the main paper.
