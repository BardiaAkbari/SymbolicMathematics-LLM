"""Reference-free verifier for the Lane-Emden n=5 IVP.

Problem:
    x*y'' + 2*y' + x*y^5 = 0
    y(0) = 1, y'(0) = 0

The verifier does NOT use the known closed-form solution as a reward target.
Model-generated a8/a9 constants are allowed and are fitted only to local
IVP conditions obtained from the Lane-Emden Taylor series near x=0.
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


@dataclass
class N5Info:
    reward: float
    valid_parse: bool
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
    """Reference-free mathematical verifier for the n=5 physical IVP."""

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
        candidate_timeout_s: float = 0.0,
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
        self.candidate_timeout_s = float(candidate_timeout_s)

        self.ode_x = np.linspace(max(self.x_anchor, 0.05), self.x_max, n_ode_points)

        # Local series for the n=5 Lane-Emden IVP, derived from the ODE itself:
        # y = 1 - x^2/6 + x^4/24 - 5*x^6/432 + O(x^8)
        # y' = -x/3 + x^3/6 - 5*x^5/72 + O(x^7)
        # This is NOT the known closed-form solution.
        x = self.x_anchor
        self.anchor_y = 1.0 - x**2 / 6.0 + x**4 / 24.0 - 5.0 * x**6 / 432.0
        self.anchor_dy = -x / 3.0 + x**3 / 6.0 - 5.0 * x**5 / 72.0

    @staticmethod
    def _candidate_coefficients(hyp: sp.Expr):
        coeffs = [
            s for s in hyp.free_symbols
            if s.is_Symbol and s.name.startswith("a") and s.name[1:].isdigit()
        ]
        return sorted(coeffs, key=lambda s: int(s.name[1:]))

    def _fit_coefficients(self, hyp: sp.Expr):
        coeffs = self._candidate_coefficients(hyp)

        if len(coeffs) > 2:
            raise ValueError(
                f"candidate contains {len(coeffs)} free coefficients; max supported is 2"
            )

        if not coeffs:
            return hyp, {}, self._anchor_error(hyp)

        hy = sp.lambdify([self.x] + coeffs, hyp, modules=["numpy"])
        hdy = sp.lambdify([self.x] + coeffs, sp.diff(hyp, self.x), modules=["numpy"])

        def residual(cvals):
            try:
                with np.errstate(all="ignore"):
                    yv = np.asarray(hy(self.x_anchor, *cvals), dtype=np.complex128)
                    dv = np.asarray(hdy(self.x_anchor, *cvals), dtype=np.complex128)
                yv = complex(yv.reshape(-1)[0])
                dv = complex(dv.reshape(-1)[0])
                if not (
                    math.isfinite(yv.real) and math.isfinite(dv.real)
                    and abs(yv.imag) <= 1e-7 and abs(dv.imag) <= 1e-7
                ):
                    return np.array([1e3, 1e3], dtype=np.float64)
                return np.array([
                    yv.real - self.anchor_y,
                    dv.real - self.anchor_dy,
                ], dtype=np.float64)
            except Exception:
                return np.array([1e3, 1e3], dtype=np.float64)

        starts = [
            np.zeros(len(coeffs), dtype=np.float64),
            np.ones(len(coeffs), dtype=np.float64),
            -np.ones(len(coeffs), dtype=np.float64),
        ]
        if len(coeffs) == 2:
            starts.extend([
                np.array([1.0, -1.0]),
                np.array([-1.0, 1.0]),
            ])

        best = None
        for start in starts:
            try:
                out = least_squares(
                    residual,
                    start,
                    bounds=(-20.0, 20.0),
                    max_nfev=100,
                    xtol=1e-9,
                    ftol=1e-9,
                    gtol=1e-9,
                )
                err = float(np.sqrt(np.mean(residual(out.x) ** 2)))
                if best is None or err < best[0]:
                    best = (err, out.x.copy())
            except Exception:
                continue

        if best is None:
            raise ValueError("coefficient fitting failed")

        coeff_map = {c: float(v) for c, v in zip(coeffs, best[1])}
        fitted = hyp.subs(coeff_map)
        return fitted, {str(c): float(v) for c, v in coeff_map.items()}, float(best[0])

    def _anchor_error(self, hyp: sp.Expr) -> float:
        try:
            fy = sp.lambdify(self.x, hyp, modules=["numpy"])
            fyp = sp.lambdify(self.x, sp.diff(hyp, self.x), modules=["numpy"])
            with np.errstate(all="ignore"):
                yv = float(np.asarray(fy(self.x_anchor), dtype=np.float64))
                dv = float(np.asarray(fyp(self.x_anchor), dtype=np.float64))
            if not (math.isfinite(yv) and math.isfinite(dv)):
                return 1e6
            return float(np.sqrt(0.5 * ((yv - self.anchor_y) ** 2 + (dv - self.anchor_dy) ** 2)))
        except Exception:
            return 1e6

    def _ode_relative_error(self, fitted: sp.Expr) -> float:
        try:
            yp = sp.diff(fitted, self.x)
            ypp = sp.diff(fitted, self.x, 2)
            terms = (
                self.x * ypp,
                2 * yp,
                self.x * fitted**5,
            )
            fns = [sp.lambdify(self.x, t, modules=["numpy"]) for t in terms]

            with np.errstate(all="ignore"):
                values = []
                for fn in fns:
                    v = np.asarray(fn(self.ode_x), dtype=np.float64)
                    if v.ndim == 0:
                        v = np.full(self.ode_x.shape, v, dtype=np.float64)
                    v = np.broadcast_to(v, self.ode_x.shape)
                    if not np.all(np.isfinite(v)):
                        return 1e6
                    values.append(np.clip(v, -1e50, 1e50))

            arr = np.stack(values, axis=0)
            residual = arr.sum(axis=0)
            denominator = np.sum(arr * arr, axis=0)
            valid = denominator > 1e-24
            if not np.any(valid):
                return 1e6

            score = float(
                np.mean(
                    residual[valid] ** 2
                    / np.maximum(denominator[valid], 1e-30)
                )
            )
            return score if math.isfinite(score) else 1e6
        except Exception:
            return 1e6

    def _exact_ivp(self, hyp: sp.Expr) -> bool:
        # Exact success is reserved for a coefficient-free expression.
        # Numerical fitting of a8/a9 is useful for dense reward, but is not
        # itself allowed to manufacture an exact symbolic certificate.
        if self._candidate_coefficients(hyp):
            return False
        try:
            yp = sp.diff(hyp, self.x)
            residual = sp.together(
                self.x * sp.diff(hyp, self.x, 2)
                + 2 * yp
                + self.x * hyp**5
            )
            if sp.simplify(residual) != 0:
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

            if not all(math.isfinite(v) for v in (anchor_error, ode_error)):
                raise ValueError("non-finite verifier metric")

            # Dense reward uses only the ODE and IVP anchor constraints.
            # Scores are capped so pathological numerical expressions cannot
            # create arbitrarily large rewards.
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

            reward = (
                ode_score
                + 0.5 * anchor_score
                - self.length_penalty * sequence_length
            )

            exact = self._exact_ivp(hyp)
            certified = (
                anchor_error <= self.certify_anchor
                and ode_error <= self.certify_ode
            )
            elite = (
                anchor_error <= self.elite_anchor
                and ode_error <= self.elite_ode
            )

            if exact:
                reward += 20.0
                certified = True
                elite = True

            return N5Info(
                reward=float(reward),
                valid_parse=True,
                certified_ivp=bool(certified),
                elite_eligible=bool(elite),
                ode_residual=float(ode_error),
                anchor_error=float(anchor_error),
                expression=hyp,
                fitted_expression=fitted,
                fitted_coefficients=coeffs,
                exact=bool(exact),
            )
        except BaseException as e:
            return N5Info(
                reward=-20.0,
                valid_parse=False,
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
            hyp = ids_to_sympy(self.env, token_ids)
            return self.score_expr(hyp, sequence_length=len(token_ids))
        except Exception as e:
            return N5Info(
                reward=-20.0,
                valid_parse=False,
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
            key = tuple(int(x) for x in ids)
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
