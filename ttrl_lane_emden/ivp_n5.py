"""Reference-free Lane-Emden n=5 verifier (fast path).

ODE (cleared):  x*y'' + 2*y' + x*y^5 = 0
IVP:            y(0)=1, y'(0)=0

Dense reward is pure numerical (no least_squares, no heavy SymPy per rollout).
Exact symbolic certification is coefficient-free and optional / gated.
The known closed form is NEVER used as a reward target.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from ttrl_lane_emden.core import generated_to_candidates, ids_to_sympy

BAD_REWARD = -20.0

# Fixed coefficient draws for a8/a9-style symbols (deterministic, small).
_COEFF_DRAWS = [
    (),  # placeholder; real draws built per candidate
]


@dataclass
class N5Info:
    reward: float
    valid_parse: bool
    finite: bool
    certified_ivp: bool
    elite_eligible: bool
    ode_residual: float
    anchor_error: float
    expression: Optional[sp.Expr]
    fitted_expression: Optional[sp.Expr]   # kept for API compat; = expression or best draw
    fitted_coefficients: Dict[str, float]
    exact: bool
    error: Optional[str] = None


class LaneEmdenN5Verifier:
    def __init__(
        self,
        env,
        x_anchor: float = 0.1,
        x_max: float = 4.0,
        n_ode_points: int = 32,
        ode_score_floor: float = -6.0,
        ode_score_cap: float = 8.0,
        anchor_score_floor: float = -6.0,
        anchor_score_cap: float = 8.0,
        length_penalty: float = 0.01,
        certify_ode: float = 1e-8,
        certify_anchor: float = 1e-6,
        elite_ode: float = 1e-2,
        elite_anchor: float = 1e-3,
        max_coeffs: int = 2,
        n_coeff_draws: int = 4,
        exact_check: bool = True,          # set False for pure speed micro-bench
    ):
        self.env = env
        self.x = env.local_dict["x"]
        self.f = env.local_dict["f"]
        self.x_anchor = float(x_anchor)
        self.x_max = float(x_max)
        self.ode_score_floor = float(ode_score_floor)
        self.ode_score_cap = float(ode_score_cap)
        self.anchor_score_floor = float(anchor_score_floor)
        self.anchor_score_cap = float(anchor_score_cap)
        self.length_penalty = float(length_penalty)
        self.certify_ode = float(certify_ode)
        self.certify_anchor = float(certify_anchor)
        self.elite_ode = float(elite_ode)
        self.elite_anchor = float(elite_anchor)
        self.max_coeffs = int(max_coeffs)
        self.n_coeff_draws = int(n_coeff_draws)
        self.exact_check = bool(exact_check)

        self.ode_x = np.linspace(max(self.x_anchor, 0.05), self.x_max, n_ode_points)

        # ODE + IVP Taylor anchor (derived from the DE, NOT from the closed form)
        xa = self.x_anchor
        self.anchor_y = 1.0 - xa**2 / 6.0 + xa**4 / 24.0 - 5.0 * xa**6 / 432.0
        self.anchor_dy = -xa / 3.0 + xa**3 / 6.0 - 5.0 * xa**5 / 72.0

        # deterministic coefficient draws
        rng = np.random.RandomState(0)
        self._draw_table = [rng.uniform(-1.5, 1.5, size=self.max_coeffs)
                            for _ in range(max(1, self.n_coeff_draws))]

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _candidate_coefficients(hyp: sp.Expr) -> List[sp.Symbol]:
        coeffs = [
            s for s in hyp.free_symbols
            if s.is_Symbol and s.name.startswith("a") and s.name[1:].isdigit()
        ]
        return sorted(coeffs, key=lambda s: int(s.name[1:]))

    @staticmethod
    def _finite_array(v, shape) -> Optional[np.ndarray]:
        arr = np.asarray(v, dtype=np.complex128)
        if arr.ndim == 0:
            arr = np.full(shape, arr, dtype=np.complex128)
        arr = np.broadcast_to(arr, shape)
        if not np.all(np.isfinite(arr)) or np.max(np.abs(arr.imag)) > 1e-7:
            return None
        return arr.real.astype(np.float64)

    def _coeff_draw_list(self, n: int) -> List[np.ndarray]:
        if n == 0:
            return [np.array([], dtype=np.float64)]
        draws = []
        for base in self._draw_table:
            draws.append(base[:n].copy())
        # also try zeros
        draws.append(np.zeros(n, dtype=np.float64))
        return draws

    # ----------------------------------------------------------- dense reward
    def _relative_ode_and_anchor(
        self, hyp: sp.Expr
    ) -> Tuple[float, float, Dict[str, float], sp.Expr]:
        """Return (ode_err, anchor_err, best_coeff_map, best_expr).

        No nonlinear fitting. For coefficient-bearing expressions we evaluate a
        handful of fixed draws and keep the best relative residual.
        """
        coeffs = self._candidate_coefficients(hyp)
        if len(coeffs) > self.max_coeffs:
            raise ValueError(f"too many free coefficient symbols: {len(coeffs)}")

        yp = sp.diff(hyp, self.x)
        ypp = sp.diff(hyp, self.x, 2)
        terms = (self.x * ypp, 2 * yp, self.x * hyp**5)

        # lambdify once
        free = [self.x] + coeffs
        term_fns = [sp.lambdify(free, t, modules=["numpy"]) for t in terms]
        y_fn = sp.lambdify(free, hyp, modules=["numpy"])
        yp_fn = sp.lambdify(free, yp, modules=["numpy"])

        best_ode = float("inf")
        best_anchor = float("inf")
        best_map: Dict[str, float] = {}
        best_expr = hyp

        for cvals in self._coeff_draw_list(len(coeffs)):
            try:
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    term_vals = []
                    ok = True
                    for fn in term_fns:
                        v = self._finite_array(fn(self.ode_x, *cvals), self.ode_x.shape)
                        if v is None:
                            ok = False
                            break
                        term_vals.append(np.clip(v, -1e50, 1e50))
                    if not ok:
                        continue

                    arr = np.stack(term_vals, axis=0)
                    residual = arr.sum(axis=0)
                    denom = np.sum(arr * arr, axis=0)
                    valid = denom > 1e-24
                    if not np.any(valid):
                        continue
                    ode_err = float(np.mean(
                        residual[valid] ** 2 / np.maximum(denom[valid], 1e-30)
                    ))
                    if not math.isfinite(ode_err):
                        continue

                    yv = self._finite_array(y_fn(self.x_anchor, *cvals), (1,))
                    dv = self._finite_array(yp_fn(self.x_anchor, *cvals), (1,))
                    if yv is None or dv is None:
                        continue
                    anchor_err = float(np.sqrt(
                        0.5 * ((yv[0] - self.anchor_y) ** 2 + (dv[0] - self.anchor_dy) ** 2)
                    ))
                    if not math.isfinite(anchor_err):
                        continue

                    # primary key: ODE residual; secondary: anchor
                    if (ode_err < best_ode) or (
                        abs(ode_err - best_ode) < 1e-15 and anchor_err < best_anchor
                    ):
                        best_ode = ode_err
                        best_anchor = anchor_err
                        best_map = {str(c): float(v) for c, v in zip(coeffs, cvals)}
                        if coeffs:
                            best_expr = hyp.subs({c: float(v) for c, v in zip(coeffs, cvals)})
                        else:
                            best_expr = hyp
            except Exception:
                continue

        return best_ode, best_anchor, best_map, best_expr

    # ---------------------------------------------------- exact certification
    def _exact_ivp(self, hyp: sp.Expr) -> bool:
        """Strict: coefficient-free + residual simplifies to 0 + IVP limits."""
        if self._candidate_coefficients(hyp):
            return False
        try:
            yp = sp.diff(hyp, self.x)
            residual = sp.simplify(
                self.x * sp.diff(hyp, self.x, 2) + 2 * yp + self.x * hyp**5
            )
            if residual != 0:
                # cheap extra try
                residual = sp.cancel(sp.together(residual))
                if residual != 0:
                    return False
            y0 = sp.limit(hyp, self.x, 0, dir="+")
            dy0 = sp.limit(yp, self.x, 0, dir="+")
            return bool(sp.simplify(y0 - 1) == 0 and sp.simplify(dy0) == 0)
        except Exception:
            return False

    # --------------------------------------------------------------- public API
    def score_expr(self, hyp: sp.Expr, sequence_length: int = 0) -> N5Info:
        try:
            ode_err, anchor_err, coeff_map, best_expr = self._relative_ode_and_anchor(hyp)

            if not (math.isfinite(ode_err) and math.isfinite(anchor_err)):
                return N5Info(
                    reward=BAD_REWARD,
                    valid_parse=True,
                    finite=False,
                    certified_ivp=False,
                    elite_eligible=False,
                    ode_residual=1e6,
                    anchor_error=1e6,
                    expression=hyp,
                    fitted_expression=best_expr,
                    fitted_coefficients=coeff_map,
                    exact=False,
                    error="non-finite numerical evaluation",
                )

            ode_score = float(np.clip(
                -math.log10(ode_err + 1e-12),
                self.ode_score_floor, self.ode_score_cap,
            ))
            anchor_score = float(np.clip(
                -math.log10(anchor_err + 1e-12),
                self.anchor_score_floor, self.anchor_score_cap,
            ))
            reward = ode_score + 0.5 * anchor_score - self.length_penalty * sequence_length

            exact = False
            if self.exact_check and not coeff_map:
                # only attempt expensive symbolic check when numerically promising
                if ode_err <= 1e-6 and anchor_err <= 1e-5:
                    exact = self._exact_ivp(hyp)
            if exact:
                reward += 30.0

            elite = (
                not coeff_map
                and anchor_err <= self.elite_anchor
                and ode_err <= self.elite_ode
            )
            certified = (
                anchor_err <= self.certify_anchor and ode_err <= self.certify_ode
            )
            if exact:
                elite = True
                certified = True

            return N5Info(
                reward=float(reward),
                valid_parse=True,
                finite=True,
                certified_ivp=bool(certified),
                elite_eligible=bool(elite),
                ode_residual=float(ode_err),
                anchor_error=float(anchor_err),
                expression=hyp,
                fitted_expression=best_expr,
                fitted_coefficients=coeff_map,
                exact=bool(exact),
            )
        except Exception as e:
            return N5Info(
                reward=BAD_REWARD,
                valid_parse=False,
                finite=False,
                certified_ivp=False,
                elite_eligible=False,
                ode_residual=1e6,
                anchor_error=1e6,
                expression=hyp,
                fitted_expression=None,
                fitted_coefficients={},
                exact=False,
                error=f"{type(e).__name__}: {e}",
            )

    def score_ids(self, token_ids: Sequence[int]) -> N5Info:
        try:
            return self.score_expr(ids_to_sympy(self.env, token_ids), len(token_ids))
        except Exception as e:
            return N5Info(
                reward=BAD_REWARD,
                valid_parse=False,
                finite=False,
                certified_ivp=False,
                elite_eligible=False,
                ode_residual=1e6,
                anchor_error=1e6,
                expression=None,
                fitted_expression=None,
                fitted_coefficients={},
                exact=False,
                error=f"{type(e).__name__}: {e}",
            )

    def evaluate_generated(self, generated, gen_len, cache=None):
        candidates = generated_to_candidates(self.env, generated, gen_len)
        infos = []
        hits = misses = 0
        for ids, _ in candidates:
            key = tuple(int(v) for v in ids)
            if cache is not None and key in cache:
                info = cache[key]
                hits += 1
            else:
                info = self.score_ids(ids)
                if cache is not None:
                    cache[key] = info
                misses += 1
            infos.append(info)
        return candidates, infos, hits, misses
