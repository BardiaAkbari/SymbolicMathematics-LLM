from dataclasses import dataclass
from typing import Optional, Sequence
import math

import numpy as np
import sympy as sp

from ttrl_lane_emden.core import (
    _coefficient_symbols,
    generated_to_candidates,
    ids_to_sympy,
)


@dataclass
class N5Info:
    reward: float
    ode_residual: float
    reference_error: float
    anchor_error: float
    exact: bool
    expression: Optional[sp.Expr]
    fitted_expression: Optional[sp.Expr]
    error: Optional[str] = None


class LaneEmdenN5Verifier:
    """
    Lane-Emden n=5 IVP:

        y'' + 2/x y' + y^5 = 0
        y(0) = 1
        y'(0) = 0

    Exact solution:

        y(x) = 1 / sqrt(1 + x^2 / 3)
    """

    def __init__(self, env, x_anchor=0.1, x_max=4.0):
        self.env = env
        self.x = env.local_dict["x"]
        self.f = env.local_dict["f"]

        self.x_anchor = x_anchor
        self.x_max = x_max

        self.target = 1 / sp.sqrt(1 + self.x**2 / 3)

        self.ode_x = np.linspace(0.1, x_max, 48)
        self.ref_x = np.linspace(0.13, x_max - 0.05, 64)

        target_fn = sp.lambdify(self.x, self.target, "numpy")
        target_d_fn = sp.lambdify(self.x, sp.diff(self.target, self.x), "numpy")

        self.ref_y = np.asarray(target_fn(self.ref_x), dtype=float)
        self.anchor_y = float(target_fn(self.x_anchor))
        self.anchor_dy = float(target_d_fn(self.x_anchor))

    def _fit_constants(self, hyp):
        coeffs = _coefficient_symbols(self.env, hyp)

        if len(coeffs) == 0:
            return hyp, {}, self._anchor_error(hyp)

        if len(coeffs) > 2:
            return hyp, {}, 1e6

        hy = sp.lambdify([self.x] + coeffs, hyp, "numpy")
        hdy = sp.lambdify([self.x] + coeffs, sp.diff(hyp, self.x), "numpy")

        def err(c):
            try:
                yv = float(np.real(hy(self.x_anchor, *c)))
                dyv = float(np.real(hdy(self.x_anchor, *c)))
                return np.array([
                    yv - self.anchor_y,
                    dyv - self.anchor_dy
                ])
            except Exception:
                return np.array([1e3, 1e3])

        starts = [
            np.zeros(len(coeffs)),
            np.ones(len(coeffs)),
            -np.ones(len(coeffs)),
        ]

        best = None

        from scipy.optimize import least_squares

        for s in starts:
            try:
                out = least_squares(
                    err,
                    s,
                    bounds=(-20, 20),
                    max_nfev=100,
                )
                score = float(np.sqrt(np.mean(err(out.x) ** 2)))
                if best is None or score < best[0]:
                    best = (score, out.x)
            except Exception:
                pass

        if best is None:
            return hyp, {}, 1e6

        coeff_map = {
            c: float(v)
            for c, v in zip(coeffs, best[1])
        }

        fitted = hyp.subs(coeff_map)

        return (
            fitted,
            {str(c): float(v) for c, v in coeff_map.items()},
            best[0],
        )

    def _anchor_error(self, expr):
        try:
            fy = sp.lambdify(self.x, expr, "numpy")
            fdy = sp.lambdify(self.x, sp.diff(expr, self.x), "numpy")

            vals = np.array([
                float(fy(self.x_anchor)),
                float(fdy(self.x_anchor)),
            ])

            target = np.array([
                self.anchor_y,
                self.anchor_dy,
            ])

            return float(np.sqrt(np.mean((vals - target) ** 2)))
        except Exception:
            return 1e6

    def score_expr(self, hyp):

        try:
            fitted, coeffs, anchor_error = self._fit_constants(hyp)

            y = fitted
            yp = sp.diff(y, self.x)
            ypp = sp.diff(y, self.x, 2)

            # Cleared Lane-Emden equation:
            #
            # x*y'' + 2*y' + x*y^5 = 0
            residual = self.x * ypp + 2 * yp + self.x * y**5

            residual_fn = sp.lambdify(self.x, residual, "numpy")
            y_fn = sp.lambdify(self.x, y, "numpy")

            r = np.asarray(residual_fn(self.ode_x), dtype=float)
            pred = np.asarray(y_fn(self.ref_x), dtype=float)

            if not np.all(np.isfinite(r)):
                return N5Info(
                    -25, 1e6, 1e6, 1e6,
                    False, hyp, fitted,
                    "non-finite residual"
                )

            if not np.all(np.isfinite(pred)):
                return N5Info(
                    -25, 1e6, 1e6, 1e6,
                    False, hyp, fitted,
                    "non-finite prediction"
                )

            ode_error = float(np.mean(r**2))

            scale = np.sqrt(np.mean(self.ref_y**2)) + 1e-12
            ref_error = float(
                np.sqrt(np.mean((pred - self.ref_y)**2)) / scale
            )

            # Exact symbolic check.
            exact = sp.simplify(
                sp.together(
                    fitted - self.target
                )
            ) == 0

            # Reward: prioritize matching the physical solution.
            reward = (
                -math.log10(ode_error + 1e-12)
                -2.0 * math.log10(ref_error + 1e-12)
                -1.0 * math.log10(anchor_error + 1e-12)
            )

            if exact:
                reward += 30.0

            return N5Info(
                reward=float(reward),
                ode_residual=ode_error,
                reference_error=ref_error,
                anchor_error=anchor_error,
                exact=bool(exact),
                expression=hyp,
                fitted_expression=fitted,
                error=None,
            )

        except Exception as e:
            return N5Info(
                reward=-25,
                ode_residual=1e6,
                reference_error=1e6,
                anchor_error=1e6,
                exact=False,
                expression=hyp,
                fitted_expression=None,
                error=str(e),
            )

    def score_ids(self, token_ids: Sequence[int]):
        try:
            hyp = ids_to_sympy(self.env, token_ids)
            return self.score_expr(hyp)
        except Exception as e:
            return N5Info(
                -25, 1e6, 1e6, 1e6,
                False, None, None, str(e)
            )

    def evaluate_generated(self, generated, gen_len):
        candidates = generated_to_candidates(
            self.env, generated, gen_len
        )

        infos = [
            self.score_ids(ids)
            for ids, _ in candidates
        ]

        return candidates, infos
