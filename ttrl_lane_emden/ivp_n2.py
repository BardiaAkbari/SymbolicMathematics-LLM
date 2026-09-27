"""Lane--Emden n=2 IVP verifier and reward shaping.

Unlike n=1, the physical n=2 Lane--Emden solution has no simple elementary
closed form.  We therefore certify *approximate IVP solutions* using only
mathematical information derived from the ODE and its initial conditions:

    y'' + 2/x y' + y^2 = 0,   y(0)=1, y'(0)=0.

The neural model may emit two symbolic integration constants.  We fit those
constants to the unique physical IVP using two near-origin anchor conditions
obtained from a high-accuracy numerical integration, then score the resulting
expression on disjoint collocation/reference grids.
"""
from __future__ import annotations

import math
import signal
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

class CandidateVerificationTimeout(BaseException):
    pass


@contextmanager
def _candidate_time_limit(seconds: float):
    """Hard wall-clock timeout for one CPU-side candidate verification."""
    seconds = float(seconds or 0.0)
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handler(signum, frame):
        raise CandidateVerificationTimeout(
            f"candidate verification exceeded {seconds:g}s"
        )

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


from ttrl_lane_emden.core import (
    _coefficient_symbols,
    generated_to_candidates,
    ids_to_sympy,
)


@dataclass
class LaneEmdenN2Reference:
    x_min: float = 0.05
    x_max: float = 4.0
    rtol: float = 1e-11
    atol: float = 1e-13

    def __post_init__(self):
        # Start very close to the regular singular point using the analytic
        # Lane--Emden n=2 Taylor series.
        eps = 1e-5
        x = eps
        y0 = (
            1.0
            - x**2 / 6.0
            + x**4 / 60.0
            - 11.0 * x**6 / 7560.0
            + x**8 / 8505.0
            - 97.0 * x**10 / 10692000.0
        )
        dy0 = (
            -x / 3.0
            + x**3 / 15.0
            - 11.0 * x**5 / 1260.0
            + 8.0 * x**7 / 8505.0
            - 97.0 * x**9 / 1069200.0
        )

        def rhs(t, z):
            y, yp = z
            return [yp, -2.0 * yp / t - y * y]

        self.sol = solve_ivp(
            rhs,
            (eps, self.x_max),
            [y0, dy0],
            method="DOP853",
            dense_output=True,
            rtol=self.rtol,
            atol=self.atol,
            max_step=0.05,
        )
        if not self.sol.success:
            raise RuntimeError(f"Lane--Emden n=2 reference integration failed: {self.sol.message}")

    def y(self, xs):
        xs = np.asarray(xs, dtype=np.float64)
        return np.asarray(self.sol.sol(xs)[0], dtype=np.float64)

    def yp(self, xs):
        xs = np.asarray(xs, dtype=np.float64)
        return np.asarray(self.sol.sol(xs)[1], dtype=np.float64)


@dataclass
class IVPRewardInfo:
    reward: float
    valid_parse: bool
    certified_ivp: bool
    elite_eligible: bool
    ode_rel_mse: float
    ref_nrmse: float
    anchor_rmse: float
    expression: Optional[sp.Expr]
    fitted_expression: Optional[sp.Expr]
    fitted_coefficients: Dict[str, float]
    error: Optional[str] = None

    # Compatibility aliases used by some generic logging helpers.
    @property
    def numerical_mse(self):
        return self.ode_rel_mse

    @property
    def exact_residual_zero(self):
        return False

    @property
    def generality_rank(self):
        return 0

    @property
    def verified_general(self):
        return False


class LaneEmdenN2Verifier:
    def __init__(
        self,
        env,
        x_anchor: float = 0.10,
        x_max: float = 4.0,
        n_ode_points: int = 36,
        n_ref_points: int = 48,
        certify_ode_rel: float = 5e-2,
        certify_ref_nrmse: float = 3e-2,
        certify_anchor_rmse: float = 2e-3,
        elite_ode_rel: float = 1.2e-1,
        elite_ref_nrmse: float = 1.2e-1,
        elite_anchor_rmse: float = 1e-2,
        max_coeffs: int = 2,
        candidate_timeout_s: float = 3.0,
    ):
        self.env = env
        self.x = env.local_dict["x"]
        self.f = env.local_dict["f"]
        self.x_anchor = float(x_anchor)
        self.x_max = float(x_max)
        self.max_coeffs = int(max_coeffs)
        self.certify_ode_rel = float(certify_ode_rel)
        self.certify_ref_nrmse = float(certify_ref_nrmse)
        self.certify_anchor_rmse = float(certify_anchor_rmse)
        self.elite_ode_rel = float(elite_ode_rel)
        self.elite_ref_nrmse = float(elite_ref_nrmse)
        self.elite_anchor_rmse = float(elite_anchor_rmse)
        self.candidate_timeout_s = float(candidate_timeout_s)
        self.reference = LaneEmdenN2Reference(x_min=x_anchor, x_max=x_max)

        # Use the denominator-cleared equation for both model input and numeric
        # residual scoring.  This removes the explicit singular 2/x coefficient.
        y = self.f(self.x)
        yp = sp.diff(y, self.x)
        ypp = sp.diff(y, self.x, 2)
        self.input_terms = (self.x * ypp, 2 * yp, self.x * y**2)
        self.equation = sp.Add(*self.input_terms)

        # Disjoint deterministic grids: odd/even offsets prevent the fitting grid
        # from being identical to the certification/reference grid.
        self.ode_x = np.linspace(self.x_anchor, self.x_max, n_ode_points, dtype=np.float64)
        step = (self.x_max - self.x_anchor) / max(n_ref_points, 1)
        self.ref_x = np.linspace(self.x_anchor + 0.37 * step, self.x_max - 0.19 * step,
                                 n_ref_points, dtype=np.float64)
        self.ref_y = self.reference.y(self.ref_x)
        self.anchor_target = np.array([
            float(self.reference.y([self.x_anchor])[0]),
            float(self.reference.yp([self.x_anchor])[0]),
        ], dtype=np.float64)

    def _fit_coefficients(self, hyp: sp.Expr):
        coeffs = _coefficient_symbols(self.env, hyp)
        if len(coeffs) > self.max_coeffs:
            raise ValueError(f"candidate has {len(coeffs)} coefficients; max supported is {self.max_coeffs}")
        if not coeffs:
            return hyp, {}, self._anchor_rmse(hyp)

        yp = sp.diff(hyp, self.x)
        fn_y = sp.lambdify([self.x] + coeffs, hyp, modules=["numpy"])
        fn_yp = sp.lambdify([self.x] + coeffs, yp, modules=["numpy"])

        def residual(cvals):
            try:
                yv = complex(fn_y(self.x_anchor, *cvals))
                dv = complex(fn_yp(self.x_anchor, *cvals))
                if (not np.isfinite(yv.real) or not np.isfinite(yv.imag) or
                        not np.isfinite(dv.real) or not np.isfinite(dv.imag) or
                        abs(yv.imag) > 1e-7 or abs(dv.imag) > 1e-7):
                    return np.array([1e4, 1e4], dtype=np.float64)
                return np.array([yv.real - self.anchor_target[0], dv.real - self.anchor_target[1]],
                                dtype=np.float64)
            except Exception:
                return np.array([1e4, 1e4], dtype=np.float64)

        starts = [
            np.zeros(len(coeffs)),
            np.ones(len(coeffs)),
            -np.ones(len(coeffs)),
        ]
        if len(coeffs) == 2:
            starts += [np.array([1.0, -1.0]), np.array([-1.0, 1.0])]
        best = None
        for s in starts:
            try:
                out = least_squares(
                    residual,
                    s,
                    bounds=(-20.0, 20.0),
                    max_nfev=80,
                    xtol=1e-9,
                    ftol=1e-9,
                    gtol=1e-9,
                )
                err = float(np.sqrt(np.mean(np.square(residual(out.x)))))
                if best is None or err < best[0]:
                    best = (err, out.x.copy())
            except Exception:
                continue
        if best is None:
            raise ValueError("coefficient fitting failed")

        coeff_map = {c: float(v) for c, v in zip(coeffs, best[1])}
        fitted = hyp.subs(coeff_map)
        return fitted, {str(c): float(v) for c, v in coeff_map.items()}, float(best[0])

    def _anchor_rmse(self, fitted: sp.Expr):
        yp = sp.diff(fitted, self.x)
        try:
            fy = sp.lambdify(self.x, fitted, modules=["numpy"])
            fyp = sp.lambdify(self.x, yp, modules=["numpy"])
            vals = np.array([complex(fy(self.x_anchor)), complex(fyp(self.x_anchor))])
            if not np.all(np.isfinite(vals)) or np.max(np.abs(vals.imag)) > 1e-7:
                return 1e6
            err = vals.real - self.anchor_target
            return float(np.sqrt(np.mean(err * err)))
        except Exception:
            return 1e6

    def _metrics(self, fitted: sp.Expr):
        yp = sp.diff(fitted, self.x)
        ypp = sp.diff(fitted, self.x, 2)
        terms = (self.x * ypp, 2 * yp, self.x * fitted**2)
        try:
            fterms = [sp.lambdify(self.x, t, modules=["numpy"]) for t in terms]
            arrs = []
            for fn in fterms:
                v = np.asarray(fn(self.ode_x), dtype=np.complex128)
                if v.ndim == 0:
                    v = np.full(self.ode_x.shape, v, dtype=np.complex128)
                v = np.broadcast_to(v, self.ode_x.shape)
                if not np.all(np.isfinite(v)) or np.max(np.abs(v.imag)) > 1e-7:
                    return 1e6, 1e6
                arrs.append(np.clip(v.real, -1e50, 1e50))
            arr = np.stack(arrs, axis=0)
            residual = arr.sum(axis=0)
            den = np.sum(arr * arr, axis=0)
            valid = den > 1e-20
            if not np.any(valid):
                ode_rel = 1e6
            else:
                ode_rel = float(np.mean((residual[valid] ** 2) / np.maximum(den[valid], 1e-30)))

            fy = sp.lambdify(self.x, fitted, modules=["numpy"])
            pred = np.asarray(fy(self.ref_x), dtype=np.complex128)
            if pred.ndim == 0:
                pred = np.full(self.ref_x.shape, pred, dtype=np.complex128)
            pred = np.broadcast_to(pred, self.ref_x.shape)
            if not np.all(np.isfinite(pred)) or np.max(np.abs(pred.imag)) > 1e-7:
                return ode_rel, 1e6
            scale = float(np.sqrt(np.mean(self.ref_y * self.ref_y))) + 1e-12
            ref_nrmse = float(np.sqrt(np.mean((pred.real - self.ref_y) ** 2)) / scale)
            return ode_rel, ref_nrmse
        except Exception:
            return 1e6, 1e6

    def score_expr(self, hyp: sp.Expr) -> IVPRewardInfo:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fitted, coeffs, anchor_rmse = self._fit_coefficients(hyp)
                ode_rel, ref_nrmse = self._metrics(fitted)
            if not all(math.isfinite(v) for v in (anchor_rmse, ode_rel, ref_nrmse)):
                raise ValueError("non-finite IVP metric")

            # Dense reward is based on *joint* normalized constraint violation.
            # This prevents a candidate from receiving a high reward merely by
            # driving one metric (especially ODE residual) extremely small while
            # being badly wrong on the physical IVP.  Each term is normalized by
            # the strict certification threshold, then compressed with log1p.
            ode_ratio = max(ode_rel, 0.0) / max(self.certify_ode_rel, 1e-12)
            ref_ratio = max(ref_nrmse, 0.0) / max(self.certify_ref_nrmse, 1e-12)
            anchor_ratio = max(anchor_rmse, 0.0) / max(self.certify_anchor_rmse, 1e-12)
            joint_penalty = (
                math.log1p(ode_ratio)
                + 2.0 * math.log1p(ref_ratio)
                + 0.5 * math.log1p(anchor_ratio)
            )
            reward = -float(joint_penalty)

            certified = (
                anchor_rmse <= self.certify_anchor_rmse
                and ode_rel <= self.certify_ode_rel
                and ref_nrmse <= self.certify_ref_nrmse
            )
            elite_eligible = (
                anchor_rmse <= self.elite_anchor_rmse
                and ode_rel <= self.elite_ode_rel
                and ref_nrmse <= self.elite_ref_nrmse
            )
            if certified:
                reward += 10.0
            elif elite_eligible:
                reward += 3.0

            return IVPRewardInfo(
                reward=float(reward),
                valid_parse=True,
                certified_ivp=bool(certified),
                elite_eligible=bool(elite_eligible),
                ode_rel_mse=float(ode_rel),
                ref_nrmse=float(ref_nrmse),
                anchor_rmse=float(anchor_rmse),
                expression=hyp,
                fitted_expression=fitted,
                fitted_coefficients=coeffs,
            )
        except CandidateVerificationTimeout:
            raise
        except BaseException as e:
            return IVPRewardInfo(
                reward=-25.0,
                valid_parse=False,
                certified_ivp=False,
                elite_eligible=False,
                ode_rel_mse=1e6,
                ref_nrmse=1e6,
                anchor_rmse=1e6,
                expression=hyp,
                fitted_expression=None,
                fitted_coefficients={},
                error=f"{type(e).__name__}: {e}",
            )

    def score_ids(self, token_ids: Sequence[int]) -> IVPRewardInfo:
        hyp = None
        try:
            with _candidate_time_limit(self.candidate_timeout_s):
                hyp = ids_to_sympy(self.env, token_ids)
                return self.score_expr(hyp)
        except CandidateVerificationTimeout as e:
            return IVPRewardInfo(
                reward=-25.0, valid_parse=False, certified_ivp=False, elite_eligible=False,
                ode_rel_mse=1e6, ref_nrmse=1e6, anchor_rmse=1e6,
                expression=hyp, fitted_expression=None, fitted_coefficients={},
                error=f"TIMEOUT: {e}",
            )
        except BaseException as e:
            return IVPRewardInfo(
                reward=-25.0, valid_parse=False, certified_ivp=False, elite_eligible=False,
                ode_rel_mse=1e6, ref_nrmse=1e6, anchor_rmse=1e6,
                expression=hyp, fitted_expression=None, fitted_coefficients={},
                error=f"{type(e).__name__}: {e}",
            )

    def evaluate_generated(self, generated, gen_len, cache: Optional[dict] = None):
        candidates = generated_to_candidates(self.env, generated, gen_len)
        infos: List[IVPRewardInfo] = []
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


def summarize_ivp(candidates, infos, top_k=5):
    order = sorted(range(len(infos)), key=lambda i: infos[i].reward, reverse=True)
    rows = []
    for i in order[:top_k]:
        ids, words = candidates[i]
        z = infos[i]
        rows.append({
            "idx": i,
            "reward": float(z.reward),
            "certified": bool(z.certified_ivp),
            "elite_eligible": bool(z.elite_eligible),
            "ode_rel": float(z.ode_rel_mse),
            "ref_nrmse": float(z.ref_nrmse),
            "anchor_rmse": float(z.anchor_rmse),
            "expr": str(z.expression) if z.expression is not None else "<parse error>",
            "fitted_expr": str(z.fitted_expression) if z.fitted_expression is not None else "<invalid>",
            "coeffs": dict(z.fitted_coefficients),
            "tokens": " ".join(words),
        })
    return rows
