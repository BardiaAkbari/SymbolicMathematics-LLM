"""Reference-free Lane-Emden n=5 verifier.

ODE (cleared):
    x*y'' + 2*y' + x*y^5 = 0
IVP:
    y(0)=1, y'(0)=0

The known closed-form solution is NOT used by the reward.

Important design choices:
- Expressions containing model coefficient symbols a8/a9 are allowed.
- a8/a9 are fitted only to the local IVP Taylor anchor so they can receive
  dense reward, but coefficient-bearing expressions are NEVER put into the
  MLE replay buffer. This prevents replay from teaching arbitrary free-
  parameter hacks.
- Truly invalid/non-finite candidates receive a reward strictly below any
  finite mathematical candidate (-20), instead of the old ~-10 plateau.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

from ttrl_lane_emden.core import generated_to_candidates, ids_to_sympy


BAD_REWARD = -20.0


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
    fitted_expression: Optional[sp.Expr]
    fitted_coefficients: Dict[str, float]
    exact: bool
    error: Optional[str] = None


class LaneEmdenN5Verifier:
    def __init__(
        self,
        env,
        x_anchor: float = 0.1,
        x_max: float = 4.0,
        n_ode_points: int = 48,
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
        self.ode_x = np.linspace(max(self.x_anchor, 0.05), self.x_max, n_ode_points)

        # Derived directly from the n=5 ODE + IVP, not from the known closed
        # form.  y = 1 - x^2/6 + x^4/24 - 5*x^6/432 + O(x^8)
        xa = self.x_anchor
        self.anchor_y = 1.0 - xa**2 / 6.0 + xa**4 / 24.0 - 5.0 * xa**6 / 432.0
        self.anchor_dy = -xa / 3.0 + xa**3 / 6.0 - 5.0 * xa**5 / 72.0

    @staticmethod
    def _candidate_coefficients(hyp: sp.Expr):
        coeffs = [
            s for s in hyp.free_symbols
            if s.is_Symbol and s.name.startswith("a") and s.name[1:].isdigit()
        ]
        return sorted(coeffs, key=lambda s: int(s.name[1:]))

    @staticmethod
    def _finite_scalar(value):
        arr = np.asarray(value, dtype=np.complex128)
        if arr.size != 1:
            return None
        z = complex(arr.reshape(-1)[0])
        if not (math.isfinite(z.real) and math.isfinite(z.imag)):
            return None
        if abs(z.imag) > 1e-7:
            return None
        return float(z.real)

    def _anchor_targets(self):
        return self.anchor_y, self.anchor_dy

    def _fit_coefficients(self, hyp: sp.Expr):
        coeffs = self._candidate_coefficients(hyp)
        if len(coeffs) > self.max_coeffs:
            raise ValueError(f"too many free coefficient symbols: {len(coeffs)}")

        if not coeffs:
            return hyp, {}, self._anchor_error(hyp)

        hy = sp.lambdify([self.x] + coeffs, hyp, modules=["numpy"])
        hdy = sp.lambdify([self.x] + coeffs, sp.diff(hyp, self.x), modules=["numpy"])
        target_y, target_dy = self._anchor_targets()

        def residual(cvals):
            try:
                with np.errstate(all="ignore"):
                    yv = self._finite_scalar(hy(self.x_anchor, *cvals))
                    dv = self._finite_scalar(hdy(self.x_anchor, *cvals))
                if yv is None or dv is None:
                    return np.array([100.0, 100.0], dtype=np.float64)
                return np.array([yv - target_y, dv - target_dy], dtype=np.float64)
            except Exception:
                return np.array([100.0, 100.0], dtype=np.float64)

        # A small number of starts keeps the verifier fast while still giving
        # simple two-parameter candidates a chance to fit the anchor.
        starts = [np.zeros(len(coeffs), dtype=np.float64)]
        if len(coeffs) == 1:
            starts += [np.ones(1), -np.ones(1)]
        else:
            starts += [np.ones(2), np.array([1.0, -1.0])]

        best_err = float("inf")
        best_x = None
        for start in starts:
            try:
                out = least_squares(
                    residual,
                    start,
                    bounds=(-20.0, 20.0),
                    max_nfev=40,
                    ftol=1e-7,
                    xtol=1e-7,
                    gtol=1e-7,
                )
                err = float(np.sqrt(np.mean(residual(out.x) ** 2)))
                if err < best_err:
                    best_err = err
                    best_x = out.x.copy()
            except Exception:
                continue

        if best_x is None or not math.isfinite(best_err):
            raise ValueError("coefficient fitting failed")

        coeff_map = {c: float(v) for c, v in zip(coeffs, best_x)}
        fitted = hyp.subs(coeff_map)
        return fitted, {str(c): float(v) for c, v in coeff_map.items()}, best_err

    def _anchor_error(self, hyp: sp.Expr) -> float:
        try:
            fy = sp.lambdify(self.x, hyp, modules=["numpy"])
            fyp = sp.lambdify(self.x, sp.diff(hyp, self.x), modules=["numpy"])
            with np.errstate(all="ignore"):
                yv = self._finite_scalar(fy(self.x_anchor))
                dv = self._finite_scalar(fyp(self.x_anchor))
            if yv is None or dv is None:
                return float("inf")
            return float(np.sqrt(0.5 * ((yv - self.anchor_y) ** 2 + (dv - self.anchor_dy) ** 2)))
        except Exception:
            return float("inf")

    def _ode_relative_error(self, fitted: sp.Expr) -> float:
        try:
            yp = sp.diff(fitted, self.x)
            ypp = sp.diff(fitted, self.x, 2)
            terms = (self.x * ypp, 2 * yp, self.x * fitted**5)
            values = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for term in terms:
                    fn = sp.lambdify(self.x, term, modules=["numpy"])
                    v = np.asarray(fn(self.ode_x), dtype=np.float64)
                    if v.ndim == 0:
                        v = np.full(self.ode_x.shape, v, dtype=np.float64)
                    v = np.broadcast_to(v, self.ode_x.shape)
                    if not np.all(np.isfinite(v)):
                        return float("inf")
                    values.append(np.clip(v, -1e50, 1e50))

            arr = np.stack(values, axis=0)
            residual = arr.sum(axis=0)
            denom = np.sum(arr * arr, axis=0)
            valid = denom > 1e-24
            if not np.any(valid):
                return float("inf")
            err = np.mean(
                residual[valid] ** 2 /
                np.maximum(denom[valid], 1e-30)
            )
            return float(err) if math.isfinite(float(err)) else float("inf")
        except Exception:
            return float("inf")

    def _exact_ivp(self, hyp: sp.Expr) -> bool:
        # Exact certification is symbolic and coefficient-free.
        if self._candidate_coefficients(hyp):
            return False
        try:
            yp = sp.diff(hyp, self.x)
            residual = sp.simplify(
                self.x * sp.diff(hyp, self.x, 2)
                + 2 * yp
                + self.x * hyp**5
            )
            if residual != 0:
                return False
            y0 = sp.limit(hyp, self.x, 0, dir="+")
            dy0 = sp.limit(yp, self.x, 0, dir="+")
            return sp.simplify(y0 - 1) == 0 and sp.simplify(dy0) == 0
        except Exception:
            return False

    def score_expr(self, hyp: sp.Expr, sequence_length: int = 0) -> N5Info:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fitted, coeffs, anchor_error = self._fit_coefficients(hyp)
                ode_error = self._ode_relative_error(fitted)

            # Any non-finite candidate is genuinely bad.  This is the key fix
            # over the previous implementation, which mapped such expressions
            # to a misleading ~-10 reward plateau.
            if not (math.isfinite(anchor_error) and math.isfinite(ode_error)):
                return N5Info(
                    reward=BAD_REWARD,
                    valid_parse=True,
                    finite=False,
                    certified_ivp=False,
                    elite_eligible=False,
                    ode_residual=1e6,
                    anchor_error=1e6,
                    expression=hyp,
                    fitted_expression=fitted,
                    fitted_coefficients=coeffs,
                    exact=False,
                    error="non-finite numerical evaluation",
                )

            ode_score = float(np.clip(
                -math.log10(ode_error + 1e-12),
                self.ode_score_floor,
                self.ode_score_cap,
            ))
            anchor_score = float(np.clip(
                -math.log10(anchor_error + 1e-12),
                self.anchor_score_floor,
                self.anchor_score_cap,
            ))

            reward = ode_score + 0.5 * anchor_score - self.length_penalty * sequence_length
            exact = self._exact_ivp(hyp)
            if exact:
                reward += 30.0

            # Replay is reserved for coefficient-free candidates only.  A
            # fitted a8/a9 expression can guide policy gradients, but should
            # not become a hard supervised target.
            elite = (
                not coeffs
                and anchor_error <= self.elite_anchor
                and ode_error <= self.elite_ode
            )
            certified = anchor_error <= self.certify_anchor and ode_error <= self.certify_ode
            if exact:
                elite = True
                certified = True

            return N5Info(
                reward=float(reward),
                valid_parse=True,
                finite=True,
                certified_ivp=bool(certified),
                elite_eligible=bool(elite),
                ode_residual=float(ode_error),
                anchor_error=float(anchor_error),
                expression=hyp,
                fitted_expression=fitted,
                fitted_coefficients=coeffs,
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
