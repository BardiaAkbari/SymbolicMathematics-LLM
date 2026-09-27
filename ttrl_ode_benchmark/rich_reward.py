from __future__ import annotations

import math
import multiprocessing as mp
import os
import signal
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from ttrl_lane_emden.core import ids_to_sympy


@dataclass
class RichRewardInfo:
    # Overall shaping score.  Exact correctness is deliberately NOT decided here.
    pre_reward: float
    equation_score: float
    constant_direction_score: float
    independence_score: float
    validity: float
    equation_error: float
    constant_direction_error: float
    is_linear: bool
    is_inhomogeneous: bool
    n_constants: int
    expression: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class RichRewardTimeout(BaseException):
    pass


@contextmanager
def _time_limit(seconds: float):
    seconds = float(seconds or 0.0)
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handler(signum, frame):
        raise RichRewardTimeout(f"rich reward exceeded {seconds:g}s")

    old = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def _coeff_symbols(env, expr: sp.Expr):
    coeff_set = set(env.coefficients.values())
    return sorted([s for s in expr.free_symbols if s in coeff_set], key=lambda s: s.name)


def linear_ode_components(env, equation: sp.Expr):
    """Return a2,a1,a0,g for a2*y'' + a1*y' + a0*y = g.

    The HSE pilot is linear ODE2.  We still detect linearity explicitly so this
    diagnostic cannot silently apply a forcing-aware metric to a nonlinear ODE.
    """
    x = env.local_dict["x"]
    f = env.local_dict["f"]
    y, yp, ypp = sp.symbols("__Y __YP __YPP")
    mapped = sp.expand(
        equation.xreplace({
            sp.diff(f(x), x, 2): ypp,
            sp.diff(f(x), x): yp,
            f(x): y,
        })
    )
    try:
        poly = sp.Poly(mapped, ypp, yp, y, domain="EX")
    except Exception as e:
        raise ValueError(f"equation is not polynomial/linear in y,y',y'': {e}")
    if poly.total_degree() > 1:
        raise ValueError("equation is nonlinear in y,y',y''")
    a2 = sp.simplify(poly.coeff_monomial(ypp))
    a1 = sp.simplify(poly.coeff_monomial(yp))
    a0 = sp.simplify(poly.coeff_monomial(y))
    c = sp.simplify(poly.coeff_monomial(1))
    # equation is L[y] + c = 0  =>  L[y] = -c = g
    g = sp.simplify(-c)
    if a2 == 0:
        raise ValueError("not genuinely second order after decomposition")
    return a2, a1, a0, g


def _eval_real(fn, xs, cvals, shape):
    try:
        with np.errstate(all="ignore"):
            v = np.asarray(fn(xs, *cvals), dtype=np.complex128)
        if v.ndim == 0:
            v = np.full(shape, v, dtype=np.complex128)
        v = np.broadcast_to(v, shape)
        real = np.isfinite(v.real) & np.isfinite(v.imag) & (np.abs(v.imag) <= 1e-7)
        return v.real, real
    except BaseException:
        return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=bool)


def _score_from_error(err: float, lo: float = -6.0, hi: float = 12.0):
    if not math.isfinite(err) or err < 0:
        return lo
    return float(np.clip(-math.log10(err + 1e-12), lo, hi))


def _relative_operator_error(a2v, a1v, a0v, d0, d1, d2, valid):
    """Scale-invariant test of whether direction d lies in null(L)."""
    t2 = a2v * d2
    t1 = a1v * d1
    t0 = a0v * d0
    res = t2 + t1 + t0
    den = t2 * t2 + t1 * t1 + t0 * t0
    good = valid & np.isfinite(res) & np.isfinite(den) & (den > 1e-24)
    if not np.any(good):
        return 1e12, 0
    ratio = (res[good] * res[good]) / np.maximum(den[good], 1e-300)
    ratio = np.clip(ratio, 0.0, 1e12)
    return float(np.mean(ratio)), int(np.sum(good))


def rich_pre_exact_reward(
    env,
    problem,
    hyp: sp.Expr,
    *,
    x_min: float = 0.25,
    x_max: float = 4.0,
    n_points: int = 24,
    n_coeff_draws: int = 4,
    seed: int = 0,
    equation_cap: float = 12.0,
    direction_cap: float = 12.0,
    direction_weight: float = 0.35,
    independence_weight: float = 2.0,
    invalid_weight: float = 5.0,
) -> RichRewardInfo:
    """Mathematically richer *pre-exact* shaping reward for linear ODE2.

    Components
    ----------
    equation_score:
      Inhomogeneous L[y]=g: mean |L[h]-g|^2 / mean |g|^2.
      Homogeneous L[y]=0: scale-invariant operator residual.
      The -log10 score is NOT clipped at 6; default cap is 12.

    constant_direction_score:
      For each used integration constant c, d=dh/dc should lie in null(L).
      We measure a scale-invariant residual of L[d].  This rewards discovery of
      correct homogeneous directions even when the particular solution is wrong.

    independence_score:
      Continuous 0..1 score based on the normalized determinant of the jet
      columns (h_c, h'_c) for the best pair of constants.  0 means redundant;
      1 means locally orthogonal / strongly independent.

    validity:
      Fraction of numerical probe values that are finite and real.

    Exact symbolic correctness is intentionally handled by the existing verifier.
    """
    x = env.local_dict["x"]
    coeffs = _coeff_symbols(env, hyp)
    xs = np.linspace(x_min, x_max, n_points, dtype=np.float64)
    rng = np.random.RandomState(seed)
    draws = max(1, n_coeff_draws if coeffs else 1)

    try:
        a2, a1, a0, g = linear_ode_components(env, problem.equation)
        is_inhom = bool(sp.simplify(g) != 0)

        hp = sp.diff(hyp, x)
        hpp = sp.diff(hyp, x, 2)
        Lh = sp.expand(a2 * hpp + a1 * hp + a0 * hyp)
        residual = sp.expand(Lh - g)

        args = [x] + coeffs
        fn_h = sp.lambdify(args, hyp, modules=["numpy"])
        fn_hp = sp.lambdify(args, hp, modules=["numpy"])
        fn_hpp = sp.lambdify(args, hpp, modules=["numpy"])
        fn_res = sp.lambdify(args, residual, modules=["numpy"])
        fn_a2 = sp.lambdify([x], a2, modules=["numpy"])
        fn_a1 = sp.lambdify([x], a1, modules=["numpy"])
        fn_a0 = sp.lambdify([x], a0, modules=["numpy"])
        fn_g = sp.lambdify([x], g, modules=["numpy"])

        shape = xs.shape
        a2v, va2 = _eval_real(lambda xx, *unused: fn_a2(xx), xs, [], shape)
        a1v, va1 = _eval_real(lambda xx, *unused: fn_a1(xx), xs, [], shape)
        a0v, va0 = _eval_real(lambda xx, *unused: fn_a0(xx), xs, [], shape)
        gv, vg = _eval_real(lambda xx, *unused: fn_g(xx), xs, [], shape)
        operator_valid = va2 & va1 & va0 & vg

        # Precompile constant tangent functions.
        tangent_fns = []
        jet_fns = []
        for c in coeffs:
            d0 = sp.diff(hyp, c)
            d1 = sp.diff(d0, x)
            d2 = sp.diff(d0, x, 2)
            tangent_fns.append((
                sp.lambdify(args, d0, modules=["numpy"]),
                sp.lambdify(args, d1, modules=["numpy"]),
                sp.lambdify(args, d2, modules=["numpy"]),
            ))
            jet_fns.append((
                sp.lambdify(args, d0, modules=["numpy"]),
                sp.lambdify(args, d1, modules=["numpy"]),
            ))

        eq_num = []
        eq_den = []
        hom_ratios = []
        dir_ratios = []
        indep_values = []
        valid_count = 0
        valid_total = 0

        for _ in range(draws):
            cvals = rng.uniform(-1.7, 1.7, size=len(coeffs)).tolist()
            hv, vh = _eval_real(fn_h, xs, cvals, shape)
            hpv, vhp = _eval_real(fn_hp, xs, cvals, shape)
            hppv, vhpp = _eval_real(fn_hpp, xs, cvals, shape)
            rv, vr = _eval_real(fn_res, xs, cvals, shape)
            base_valid = operator_valid & vh & vhp & vhpp & vr
            valid_count += int(np.sum(base_valid))
            valid_total += int(base_valid.size)

            if is_inhom:
                good = base_valid
                if np.any(good):
                    eq_num.extend(np.square(rv[good]).tolist())
                    eq_den.extend(np.square(gv[good]).tolist())
            else:
                # Homogeneous case must remain amplitude/scale invariant.
                t2 = a2v * hppv
                t1 = a1v * hpv
                t0 = a0v * hv
                den = t2*t2 + t1*t1 + t0*t0
                good = base_valid & np.isfinite(den) & (den > 1e-24)
                if np.any(good):
                    ratio = np.square(rv[good]) / np.maximum(den[good], 1e-300)
                    hom_ratios.extend(np.clip(ratio, 0.0, 1e12).tolist())

            # Does every coefficient direction lie in the homogeneous nullspace?
            jets = []
            for (fd0, fd1, fd2) in tangent_fns:
                d0v, vd0 = _eval_real(fd0, xs, cvals, shape)
                d1v, vd1 = _eval_real(fd1, xs, cvals, shape)
                d2v, vd2 = _eval_real(fd2, xs, cvals, shape)
                dvalid = operator_valid & vd0 & vd1 & vd2
                er, nvalid = _relative_operator_error(a2v, a1v, a0v, d0v, d1v, d2v, dvalid)
                if nvalid:
                    dir_ratios.append(er)
                jets.append((d0v, d1v, dvalid))

            # Continuous 2D independence.  Use the best coefficient pair because
            # exact generality also only requires two independent constants.
            if len(jets) >= 2:
                pair_scores = []
                for i in range(len(jets)):
                    for j in range(i+1, len(jets)):
                        u0, u1, vu = jets[i]
                        v0, v1, vv = jets[j]
                        good = vu & vv
                        if not np.any(good):
                            continue
                        det = np.abs(u0[good]*v1[good] - v0[good]*u1[good])
                        nu = np.sqrt(np.square(u0[good]) + np.square(u1[good]))
                        nv = np.sqrt(np.square(v0[good]) + np.square(v1[good]))
                        denom = nu*nv
                        ggood = denom > 1e-18
                        if np.any(ggood):
                            vals = np.clip(det[ggood]/np.maximum(denom[ggood],1e-300), 0.0, 1.0)
                            pair_scores.extend(vals.tolist())
                if pair_scores:
                    # Median is robust to isolated singular/probe points.
                    indep_values.append(float(np.median(pair_scores)))

        validity = float(valid_count / max(1, valid_total))

        if is_inhom:
            if not eq_num:
                eq_error = 1e12
            else:
                # Global forcing energy denominator avoids explosions at zeros of
                # sin/cos/polynomial forcing and cannot be inflated by h itself.
                num = float(np.mean(np.clip(np.asarray(eq_num), 0.0, 1e200)))
                den = float(np.mean(np.clip(np.asarray(eq_den), 0.0, 1e200))) if eq_den else 0.0
                eq_error = num / max(den, 1e-12)
        else:
            eq_error = float(np.mean(hom_ratios)) if hom_ratios else 1e12

        # Missing constants should not receive a fake perfect direction score.
        if not coeffs:
            dir_error = 1e12
            q_dir = -6.0
        elif dir_ratios:
            dir_error = float(np.mean(np.clip(np.asarray(dir_ratios), 0.0, 1e12)))
            q_dir = _score_from_error(dir_error, -6.0, direction_cap)
        else:
            dir_error = 1e12
            q_dir = -6.0

        q_eq = _score_from_error(eq_error, -6.0, equation_cap)
        indep = float(np.median(indep_values)) if indep_values else 0.0
        indep = float(np.clip(indep, 0.0, 1.0))

        pre = (
            q_eq
            + float(direction_weight) * q_dir
            + float(independence_weight) * indep
            - float(invalid_weight) * (1.0 - validity)
        )

        return RichRewardInfo(
            pre_reward=float(pre),
            equation_score=float(q_eq),
            constant_direction_score=float(q_dir),
            independence_score=indep,
            validity=validity,
            equation_error=float(eq_error),
            constant_direction_error=float(dir_error),
            is_linear=True,
            is_inhomogeneous=is_inhom,
            n_constants=len(coeffs),
            expression=str(hyp),
        )
    except BaseException as e:
        return RichRewardInfo(
            pre_reward=-25.0,
            equation_score=-6.0,
            constant_direction_score=-6.0,
            independence_score=0.0,
            validity=0.0,
            equation_error=1e12,
            constant_direction_error=1e12,
            is_linear=False,
            is_inhomogeneous=False,
            n_constants=0,
            expression=str(hyp) if hyp is not None else None,
            error=f"{type(e).__name__}: {e}",
        )


def score_rich_ids(env, problem, token_ids: Sequence[int], timeout_s: float = 4.0):
    hyp = None
    try:
        with _time_limit(timeout_s):
            hyp = ids_to_sympy(env, token_ids)
            return rich_pre_exact_reward(env, problem, hyp)
    except RichRewardTimeout as e:
        return RichRewardInfo(-25.0,-6.0,-6.0,0.0,0.0,1e12,1e12,False,False,0,
                              str(hyp) if hyp is not None else None,f"TIMEOUT: {e}")
    except BaseException as e:
        return RichRewardInfo(-25.0,-6.0,-6.0,0.0,0.0,1e12,1e12,False,False,0,
                              str(hyp) if hyp is not None else None,f"{type(e).__name__}: {e}")


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _worker_score(payload):
    env, problem, timeout_s, ids = payload
    return score_rich_ids(env, problem, ids, timeout_s)


class ParallelRichRewardPool:
    def __init__(self, workers: int):
        self.workers = max(1, int(workers))
        self._pool = None
        if self.workers > 1:
            self._pool = mp.get_context("spawn").Pool(self.workers, initializer=_worker_init)

    def score_many(self, env, problem, timeout_s: float, id_batches):
        batches = [list(map(int, ids)) for ids in id_batches]
        if not batches:
            return []
        if self._pool is None:
            return [score_rich_ids(env, problem, ids, timeout_s) for ids in batches]
        payloads = [(env, problem, float(timeout_s), ids) for ids in batches]
        return self._pool.map(_worker_score, payloads, chunksize=1)

    def close(self):
        if self._pool is not None:
            self._pool.close(); self._pool.join(); self._pool = None

    def terminate(self):
        if self._pool is not None:
            self._pool.terminate(); self._pool.join(); self._pool = None

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        self.close() if exc_type is None else self.terminate()
