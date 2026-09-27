from __future__ import annotations

import math
import multiprocessing as mp
import os
import signal
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import sympy as sp


@dataclass
class RewardV3Info:
    # Diagnostic shaping score only. Exact symbolic correctness is NOT decided here.
    pre_reward: float
    equation_score: float
    direction_min_score: float
    direction_scores: list[float]
    independence_score: float
    validity: float
    equation_error: float
    direction_errors: list[float]
    direction_penalty: float
    independence_penalty: float
    missing_constant_penalty: float
    invalid_penalty: float
    is_linear: bool
    is_inhomogeneous: bool
    n_constants: int
    expression: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class RewardV3Timeout(BaseException):
    pass


@contextmanager
def _time_limit(seconds: float):
    seconds = float(seconds or 0.0)
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handler(signum, frame):
        raise RewardV3Timeout(f"reward-v3 exceeded {seconds:g}s")

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
    """Return a2,a1,a0,g for a2*y'' + a1*y' + a0*y = g."""
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
    """Scale-invariant residual of a direction d under homogeneous L[d]=0.

    Important zero-operator case: if L annihilates the direction term-by-term
    (e.g. d=1 when a0=0), then numerator=denominator=0.  That is a PERFECT
    homogeneous direction, not an invalid one.
    """
    with np.errstate(all="ignore"):
        t2 = a2v * d2
        t1 = a1v * d1
        t0 = a0v * d0
        res = t2 + t1 + t0
        den = t2 * t2 + t1 * t1 + t0 * t0
        mag = d0*d0 + d1*d1 + d2*d2
    finite = valid & np.isfinite(res) & np.isfinite(den) & np.isfinite(mag) & (mag > 1e-24)
    ordinary = finite & (den > 1e-24)
    annihilated = finite & (den <= 1e-24) & (np.abs(res) <= 1e-12)
    vals = []
    if np.any(ordinary):
        with np.errstate(all="ignore"):
            ratio = (res[ordinary] * res[ordinary]) / np.maximum(den[ordinary], 1e-300)
        vals.extend(np.clip(ratio, 0.0, 1e12).tolist())
    if np.any(annihilated):
        vals.extend([0.0] * int(np.sum(annihilated)))
    if not vals:
        return 1e12, 0
    return float(np.mean(vals)), len(vals)


def _robust_independence(jet_fns, coeffs, rng, *, x_probe=None, n_draws=6):
    """Conditioning-resistant *gate* for two independent integration constants.

    v2 used the median local normalized Wronskian on x in [0.25,4].  For stiff
    modes (e.g. exp(-6x)cosh(a+34x)) both columns become nearly collinear over
    most positive x although they are generically independent.

    v3 deliberately uses small/near-origin probes and the BEST robust local
    Wronskian (90th percentile / max-like statistic).  It is used only as a
    penalty gate, never as a positive reward.
    """
    if len(jet_fns) < 2:
        return 0.0
    if x_probe is None:
        x_probe = np.asarray([-0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 1.0], dtype=np.float64)
    else:
        x_probe = np.asarray(x_probe, dtype=np.float64)
    shape = x_probe.shape
    best_pair = 0.0

    for i in range(len(jet_fns)):
        for j in range(i + 1, len(jet_fns)):
            vals = []
            for _ in range(max(1, n_draws)):
                cvals = rng.uniform(-1.0, 1.0, size=len(coeffs)).tolist()
                u0, vu0 = _eval_real(jet_fns[i][0], x_probe, cvals, shape)
                u1, vu1 = _eval_real(jet_fns[i][1], x_probe, cvals, shape)
                v0, vv0 = _eval_real(jet_fns[j][0], x_probe, cvals, shape)
                v1, vv1 = _eval_real(jet_fns[j][1], x_probe, cvals, shape)
                good = vu0 & vu1 & vv0 & vv1
                if not np.any(good):
                    continue
                with np.errstate(all="ignore"):
                    det = np.abs(u0[good] * v1[good] - v0[good] * u1[good])
                    nu = np.hypot(u0[good], u1[good])
                    nv = np.hypot(v0[good], v1[good])
                    den = nu * nv
                    ggood = np.isfinite(det) & np.isfinite(den) & (den > 1e-18)
                    if np.any(ggood):
                        q = np.clip(det[ggood] / np.maximum(den[ggood], 1e-300), 0.0, 1.0)
                        vals.extend(q.tolist())
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                # We need evidence that the directions are independent SOMEWHERE,
                # not that they are well-conditioned at every stiff probe.
                pair = float(np.quantile(arr, 0.90)) if len(arr) >= 5 else float(np.max(arr))
                best_pair = max(best_pair, pair)
    return float(np.clip(best_pair, 0.0, 1.0))


def rich_pre_exact_reward_v3(
    env,
    problem,
    hyp: sp.Expr,
    *,
    x_min: float = 0.25,
    x_max: float = 4.0,
    n_points: int = 24,
    n_coeff_draws: int = 4,
    seed: int = 0,
    score_cap: float = 12.0,
    direction_gate: float = 4.0,
    direction_penalty_weight: float = 1.0,
    independence_gate: float = 0.01,
    independence_max_penalty: float = 4.0,
    missing_constant_penalty_each: float = 4.0,
    invalid_weight: float = 5.0,
    homogeneous_equation_weight: float = 0.25,
) -> RewardV3Info:
    """Reward v3: equation-first, structural constraints as penalties/gates.

    Inhomogeneous L[y]=g:
        R = Q_force
            - penalty(bad homogeneous parameter directions)
            - penalty(degenerate / missing constants)
            - penalty(invalid domain evaluations)

    Homogeneous L[y]=0:
        R = min_i Q(L[dh/dc_i]) + 0.25 * Q(L[h])
            - penalty(degenerate / missing constants)
            - penalty(invalid evaluations)

    Key differences from v2:
      * no +4.2 bonus merely for already-correct homogeneous directions;
      * worst parameter direction controls homogeneous quality (one good mode
        cannot hide one bad mode);
      * independence is a penalty-only gate using near-origin robust probes;
      * exact symbolic correctness remains entirely outside this shaping reward.
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

        tangent_fns = []
        jet_fns = []
        for c in coeffs:
            d0 = sp.diff(hyp, c)
            d1 = sp.diff(d0, x)
            d2 = sp.diff(d0, x, 2)
            fd0 = sp.lambdify(args, d0, modules=["numpy"])
            fd1 = sp.lambdify(args, d1, modules=["numpy"])
            fd2 = sp.lambdify(args, d2, modules=["numpy"])
            tangent_fns.append((fd0, fd1, fd2))
            jet_fns.append((fd0, fd1))

        eq_num = []
        eq_den = []
        hom_ratios = []
        per_direction_ratios = [[] for _ in tangent_fns]
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
                with np.errstate(all="ignore"):
                    t2 = a2v * hppv
                    t1 = a1v * hpv
                    t0 = a0v * hv
                    den = t2 * t2 + t1 * t1 + t0 * t0
                    rr = t2 + t1 + t0
                good = base_valid & np.isfinite(den) & np.isfinite(rr) & (den > 1e-24)
                if np.any(good):
                    with np.errstate(all="ignore"):
                        ratio = np.square(rr[good]) / np.maximum(den[good], 1e-300)
                    hom_ratios.extend(np.clip(ratio, 0.0, 1e12).tolist())

            for di, (fd0, fd1, fd2) in enumerate(tangent_fns):
                d0v, vd0 = _eval_real(fd0, xs, cvals, shape)
                d1v, vd1 = _eval_real(fd1, xs, cvals, shape)
                d2v, vd2 = _eval_real(fd2, xs, cvals, shape)
                dvalid = operator_valid & vd0 & vd1 & vd2
                er, nvalid = _relative_operator_error(a2v, a1v, a0v, d0v, d1v, d2v, dvalid)
                if nvalid:
                    per_direction_ratios[di].append(er)

        validity = float(valid_count / max(1, valid_total))

        if is_inhom:
            if not eq_num:
                eq_error = 1e12
            else:
                num = float(np.mean(np.clip(np.asarray(eq_num), 0.0, 1e200)))
                den = float(np.mean(np.clip(np.asarray(eq_den), 0.0, 1e200))) if eq_den else 0.0
                eq_error = num / max(den, 1e-12)
        else:
            eq_error = float(np.mean(hom_ratios)) if hom_ratios else 1e12
        q_eq = _score_from_error(eq_error, -6.0, score_cap)

        direction_errors = []
        direction_scores = []
        for vals in per_direction_ratios:
            if vals:
                er = float(np.mean(np.clip(np.asarray(vals), 0.0, 1e12)))
            else:
                er = 1e12
            direction_errors.append(er)
            direction_scores.append(_score_from_error(er, -6.0, score_cap))

        if direction_scores:
            q_dir_min = float(min(direction_scores))
        else:
            q_dir_min = -6.0

        # Robust independence gate. Important: penalty only, never a positive bonus.
        indep_rng = np.random.RandomState(seed + 7919)
        indep = _robust_independence(jet_fns, coeffs, indep_rng)

        missing = max(0, 2 - len(coeffs))
        missing_penalty = float(missing_constant_penalty_each) * missing

        if independence_gate <= 0 or indep >= independence_gate:
            indep_penalty = 0.0
        else:
            indep_penalty = float(independence_max_penalty) * (independence_gate - indep) / independence_gate
            indep_penalty = float(np.clip(indep_penalty, 0.0, independence_max_penalty))

        invalid_penalty = float(invalid_weight) * (1.0 - validity)

        # Correct homogeneous directions are a CONSTRAINT, not a large bonus.
        direction_penalty = 0.0
        if is_inhom:
            direction_penalty = float(direction_penalty_weight) * max(0.0, float(direction_gate) - q_dir_min)
            core = q_eq
        else:
            # For homogeneous equations the worst coefficient direction is the
            # actual task: every integration direction must satisfy L[d]=0.
            # q_eq is retained only as a weak secondary signal for nonlinear
            # parameterizations / coefficient-independent pieces.
            core = q_dir_min + float(homogeneous_equation_weight) * q_eq

        pre = core - direction_penalty - indep_penalty - missing_penalty - invalid_penalty

        return RewardV3Info(
            pre_reward=float(pre),
            equation_score=float(q_eq),
            direction_min_score=float(q_dir_min),
            direction_scores=[float(v) for v in direction_scores],
            independence_score=float(indep),
            validity=float(validity),
            equation_error=float(eq_error),
            direction_errors=[float(v) for v in direction_errors],
            direction_penalty=float(direction_penalty),
            independence_penalty=float(indep_penalty),
            missing_constant_penalty=float(missing_penalty),
            invalid_penalty=float(invalid_penalty),
            is_linear=True,
            is_inhomogeneous=is_inhom,
            n_constants=len(coeffs),
            expression=str(hyp),
        )
    except BaseException as e:
        return RewardV3Info(
            pre_reward=-25.0,
            equation_score=-6.0,
            direction_min_score=-6.0,
            direction_scores=[],
            independence_score=0.0,
            validity=0.0,
            equation_error=1e12,
            direction_errors=[],
            direction_penalty=10.0,
            independence_penalty=4.0,
            missing_constant_penalty=8.0,
            invalid_penalty=5.0,
            is_linear=False,
            is_inhomogeneous=False,
            n_constants=0,
            expression=str(hyp) if hyp is not None else None,
            error=f"{type(e).__name__}: {e}",
        )


def score_reward_v3_ids(env, problem, token_ids: Sequence[int], timeout_s: float = 4.0):
    hyp = None
    try:
        with _time_limit(timeout_s):
            # Lazy import makes the mathematical scorer independently testable.
            from ttrl_lane_emden.core import ids_to_sympy
            hyp = ids_to_sympy(env, token_ids)
            return rich_pre_exact_reward_v3(env, problem, hyp)
    except RewardV3Timeout as e:
        return RewardV3Info(-25.0,-6.0,-6.0,[],0.0,0.0,1e12,[],10.0,4.0,8.0,5.0,False,False,0,
                            str(hyp) if hyp is not None else None,f"TIMEOUT: {e}")
    except BaseException as e:
        return RewardV3Info(-25.0,-6.0,-6.0,[],0.0,0.0,1e12,[],10.0,4.0,8.0,5.0,False,False,0,
                            str(hyp) if hyp is not None else None,f"{type(e).__name__}: {e}")


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _worker_score(payload):
    env, problem, timeout_s, ids = payload
    return score_reward_v3_ids(env, problem, ids, timeout_s)


class ParallelRewardV3Pool:
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
            return [score_reward_v3_ids(env, problem, ids, timeout_s) for ids in batches]
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
