# HSE ODE2 Hard-10 Failure Benchmark

All 10 were selected from the 600-problem run with: baseline exact=0, warmup exact=0, no train-step exact discovery, final held-out exact=0, and final greedy wrong. The runnable JSONL intentionally contains no reference solution.

| # | Family | ID | ODE | Old best reward (baseline→final) | Old final greedy | Failure pattern |
|---:|---|---|---|---:|---|---|
| 1 | trig | `data/test.xlsx#test:35` | `3y^{\prime\prime} -5y^{\prime} -2y = \cos(2 * x)` | 4.639→6.000 | `a8*exp(-x/3) + exp(a9 + 2*x) - cos(x)**2` | correct homogeneous modes; wrong particular forcing response |
| 2 | trig | `data/test.xlsx#test:1516` | `4y^{\prime\prime} + 4y^{\prime} + 0y = \cos(2 * x)` | 1.379→1.998 | `a8 + a9*exp(-x) + sin(2*x)/8` | correct homogeneous modes; incomplete/wrong trig particular |
| 3 | trig | `data/test.xlsx#test:5454` | `2y^{\prime\prime} -3y^{\prime} -2y = \sin(x)` | 6.000→6.000 | `a8*exp(-x/2) + a9*exp(2*x) - sin(x)` | correct homogeneous modes; wrong sin/cos particular |
| 4 | exp | `data/test.xlsx#test:12167` | `y^{\prime\prime} -3y^{\prime} -4y = e^{x}` | 6.000→6.000 | `a8*exp(4*x) + (a9 - x/2)*exp(x)` | forcing-response confusion; misses one homogeneous mode |
| 5 | exp | `data/test.xlsx#test:5276` | `-2y^{\prime\prime} -2y^{\prime} + 4y = e^{2 * x}` | 1.183→0.870 | `a8*exp(-2*x) + (a9 - x)*exp(2*x)/4` | forcing exponent confused with solution mode; misses e^x homogeneous mode |
| 6 | poly | `data/test.xlsx#test:1264` | `4y^{\prime\prime} -5y^{\prime} + 0y = x^{2}` | 6.000→6.000 | `a8*exp(5*x/4) + a9 - x**3/25 - x` | correct homogeneous modes; polynomial particular close but coefficients wrong |
| 7 | poly | `data/test.xlsx#test:6492` | `y^{\prime\prime} -4y^{\prime} + 3y = x^{3}` | 6.000→6.000 | `a8*exp(x) + a9*exp(3*x) + x + 1` | correct homogeneous modes; polynomial particular severely underfit |
| 8 | homogeneous | `data/test.xlsx#test:60` | `y^{\prime\prime}+48y^{\prime}+495y=0` | 6.000→6.000 | `(a8 + exp(a9 + 3*x))*exp(-25*x)` | pure homogeneous; both characteristic exponents wrong |
| 9 | homogeneous | `data/test.xlsx#test:945` | `y^{\prime\prime}+44y^{\prime}+403y=0` | 2.366→2.469 | `a9*exp(-22*x)*cos(a8 + x)` | pure homogeneous; model uses damped oscillatory family instead of two real modes |
| 10 | homogeneous | `data/test.xlsx#test:367` | `y^{\prime\prime}+12y^{\prime}-1120y=0` | 2.049→2.049 | `a9*exp(-6*x)*sinh(a8 + 12*x)` | pure homogeneous; widely separated true roots not recovered |

## Offline analytic references (do not feed to TTRL)

1. `data/test.xlsx#test:35`: `C1*exp(-x/3) + C2*exp(2*x) - 5*sin(2*x)/148 - 7*cos(2*x)/148`
2. `data/test.xlsx#test:1516`: `C1 + C2*exp(-x) + sin(2*x)/40 - cos(2*x)/20`
3. `data/test.xlsx#test:5454`: `C1*exp(-x/2) + C2*exp(2*x) - 4*sin(x)/25 + 3*cos(x)/25`
4. `data/test.xlsx#test:12167`: `C1*exp(-x) + C2*exp(4*x) - exp(x)/6`
5. `data/test.xlsx#test:5276`: `C1*exp(-2*x) + C2*exp(x) - exp(2*x)/8`
6. `data/test.xlsx#test:1264`: `C1 + C2*exp(5*x/4) - x**3/15 - 4*x**2/25 - 32*x/125`
7. `data/test.xlsx#test:6492`: `C1*exp(x) + C2*exp(3*x) + x**3/3 + 4*x**2/3 + 26*x/9 + 80/27`
8. `data/test.xlsx#test:60`: `C1*exp(-15*x) + C2*exp(-33*x)`
9. `data/test.xlsx#test:945`: `C1*exp(-13*x) + C2*exp(-31*x)`
10. `data/test.xlsx#test:367`: `C1*exp(-40*x) + C2*exp(28*x)`

## First experimental sequence

1. **Frozen support probe:** 10k samples per problem at temperatures 0.7, 1.0, 1.3, 1.6. This asks whether the exact solution is merely ultra-rare before changing learning.
2. **Reward diagnostic:** compare current monolithic relative residual to a forcing-aware / structure-aware reward on the same generated candidates.
3. **Pre-exact RL:** only after reward diagnostics, compare REINFORCE vs GRPO under the same verifier-call budget.
4. Primary metric: first-exact discovery rate and verifier calls to first exact; final greedy exact is secondary because post-discovery amplification is already strong.

## Important reward pathology exposed by this set

For inhomogeneous linear ODEs, the current denominator `sum_j |T_j[h]|^2` can be dominated by very large homogeneous terms that cancel in the numerator. A candidate that gets the homogeneous solution right but misses the fixed particular solution can therefore receive an artificially tiny relative residual and saturate at reward 6. Several selected trig/polynomial cases exhibit exactly this behavior.