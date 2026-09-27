from __future__ import annotations

import math
import multiprocessing as mp
import os
import signal
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import sympy as sp


@dataclass
class RewardV31Info:
    # Dense PRE-EXACT shaping score only. Exact symbolic correctness is decided
    # by the existing exact verifier, never by this scorer.
    pre_reward: float
    equation_score: float
    equation_domain_scores: list[float]
    direction_min_score: float
    direction_scores: list[float]
    direction_domain_scores: list[list[float]]
    base_score: float
    base_domain_scores: list[float]
    independence_score: float              # diagnostic only in v3.1
    validity: float
    direction_penalty: float
    missing_constant_penalty: float
    invalid_penalty: float
    hard_generality_penalty: float
    generality_gate_pass: bool
    wronskian_zero: bool
    wronskian_expression: Optional[str]
    is_linear: bool
    is_inhomogeneous: bool
    n_constants: int
    expression: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class RewardV31Timeout(BaseException):
    pass


@contextmanager
def _time_limit(seconds: float):
    seconds = float(seconds or 0.0)
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handler(signum, frame):
        raise RewardV31Timeout(f"reward-v3.1 exceeded {seconds:g}s")

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




def _aggregate_domain_scores(scores):
    """Soft worst-case across negative/central/positive domains.

    A hard min caught asymptotic reward hacks but introduced artificial valleys
    for polynomial completion (a correct intermediate term can temporarily be
    worse near a forcing zero).  v3.1 therefore uses 60% worst-domain + 40%
    median-domain score.  A single bad regime still dominates, while progress
    on the other regimes remains visible to RL.
    """
    if not scores:
        return -6.0
    arr=np.asarray(scores,dtype=np.float64)
    return float(0.60*np.min(arr)+0.40*np.median(arr))

def _random_domains(seed: int, n_points_per_domain: int):
    """Three asymptotic regimes. Random interiors + fixed anchors.

    The seed is intentionally explicit: diagnostics are reproducible, while a
    future RL loop should change probe_seed every optimization step.
    """
    specs = [(-4.0, -0.25), (-1.0, 1.0), (0.25, 4.0)]
    out = []
    n = max(6, int(n_points_per_domain))
    for di, (lo, hi) in enumerate(specs):
        rng = np.random.RandomState(int(seed) + 104729 * (di + 1))
        # Include boundaries and midpoint, then randomized interior probes.
        anchors = np.asarray([lo, (lo + hi) / 2.0, hi], dtype=np.float64)
        k = max(0, n - len(anchors))
        rnd = rng.uniform(lo, hi, size=k).astype(np.float64) if k else np.empty(0, dtype=np.float64)
        xs = np.unique(np.concatenate([anchors, rnd]))
        out.append(xs)
    return out


def _stable_relative_operator_error(a2v, a1v, a0v, d0, d1, d2, valid):
    """Scale-invariant L[d]=0 residual without overflow-prone raw squaring."""
    with np.errstate(all="ignore"):
        t2 = a2v * d2
        t1 = a1v * d1
        t0 = a0v * d0
        res = t2 + t1 + t0
        mag = np.maximum.reduce([np.abs(d0), np.abs(d1), np.abs(d2)])
        scale = np.maximum.reduce([np.abs(t2), np.abs(t1), np.abs(t0)])

    finite = valid & np.isfinite(res) & np.isfinite(scale) & np.isfinite(mag) & (mag > 1e-24)
    ordinary = finite & (scale > 1e-300)
    annihilated = finite & (scale <= 1e-300) & (np.abs(res) <= 1e-12)
    vals = []
    if np.any(ordinary):
        s = scale[ordinary]
        with np.errstate(all="ignore"):
            z2 = t2[ordinary] / s
            z1 = t1[ordinary] / s
            z0 = t0[ordinary] / s
            zr = res[ordinary] / s
            den = z2*z2 + z1*z1 + z0*z0
            ratio = (zr*zr) / np.maximum(den, 1e-300)
        vals.extend(np.clip(ratio, 0.0, 1e12).tolist())
    if np.any(annihilated):
        vals.extend([0.0] * int(np.sum(annihilated)))
    if not vals:
        return 1e12, 0
    return float(np.mean(vals)), len(vals)


def _stable_forcing_error(residual, forcing, valid):
    """RMS residual / RMS forcing on one domain, computed with common scaling."""
    good = valid & np.isfinite(residual) & np.isfinite(forcing)
    if not np.any(good):
        return 1e12, 0
    r = residual[good]
    g = forcing[good]
    scale = float(max(np.max(np.abs(r)), np.max(np.abs(g)), 1e-300))
    with np.errstate(all="ignore"):
        rn = r / scale
        gn = g / scale
        num = float(np.mean(rn * rn))
        den = float(np.mean(gn * gn))
    if not math.isfinite(num) or not math.isfinite(den):
        return 1e12, int(np.sum(good))
    if den <= 1e-300:
        # This domain carries no forcing information. Treat exact-zero residual
        # as perfect; otherwise a very poor fit.
        return (0.0 if num <= 1e-24 else 1e12), int(np.sum(good))
    return float(np.clip(num / den, 0.0, 1e12)), int(np.sum(good))


def _symbolic_wronskian_gate(directions, x):
    """Return (pass, all_pairs_zero, representative W expression).

    For order-2 we need at least one pair of parameter directions with a
    Wronskian that is not identically zero.  This is a GENERIC symbolic gate,
    not a conditioning score.
    """
    if len(directions) < 2:
        return False, True, None
    saw_nonzero = False
    representative = None
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            d1 = directions[i]
            d2 = directions[j]
            try:
                W = sp.expand(d1 * sp.diff(d2, x) - d2 * sp.diff(d1, x))
                W = sp.cancel(sp.together(W))
                if W != 0:
                    W2 = sp.simplify(W)
                    W = W2
                if representative is None:
                    representative = W
                if W != 0:
                    saw_nonzero = True
                    representative = W
                    return True, False, representative
            except BaseException:
                # Conservative: if symbolic simplification fails, don't declare
                # dependence. Exact verifier remains final authority.
                return True, False, representative
    return saw_nonzero, not saw_nonzero, representative


def _diagnostic_independence(jet_fns, coeffs, rng, n_draws=4):
    """Numerical independence diagnostic only; NEVER changes v3.1 reward."""
    if len(jet_fns) < 2:
        return 0.0
    x_probe = np.asarray([-1.0,-0.5,-0.25,-0.1,0.0,0.1,0.25,0.5,1.0], dtype=np.float64)
    shape = x_probe.shape
    best = 0.0
    for i in range(len(jet_fns)):
        for j in range(i+1, len(jet_fns)):
            vals=[]
            for _ in range(max(1,n_draws)):
                cvals=rng.uniform(-1.0,1.0,size=len(coeffs)).tolist()
                u0,vu0=_eval_real(jet_fns[i][0],x_probe,cvals,shape)
                u1,vu1=_eval_real(jet_fns[i][1],x_probe,cvals,shape)
                v0,vv0=_eval_real(jet_fns[j][0],x_probe,cvals,shape)
                v1,vv1=_eval_real(jet_fns[j][1],x_probe,cvals,shape)
                good=vu0&vu1&vv0&vv1
                if not np.any(good): continue
                with np.errstate(all='ignore'):
                    det=np.abs(u0[good]*v1[good]-v0[good]*u1[good])
                    den=np.hypot(u0[good],u1[good])*np.hypot(v0[good],v1[good])
                    ok=np.isfinite(det)&np.isfinite(den)&(den>1e-18)
                    if np.any(ok): vals.extend(np.clip(det[ok]/den[ok],0,1).tolist())
            if vals:
                best=max(best,float(np.quantile(np.asarray(vals),0.90)))
    return float(np.clip(best,0,1))


def rich_pre_exact_reward_v31(
    env,
    problem,
    hyp: sp.Expr,
    *,
    n_points_per_domain: int = 16,
    n_coeff_draws: int = 4,
    probe_seed: int = 0,
    score_cap: float = 12.0,
    direction_gate: float = 4.0,
    direction_penalty_weight: float = 1.0,
    hard_generality_cap: float = -4.0,
    missing_constant_penalty_each: float = 4.0,
    invalid_weight: float = 5.0,
) -> RewardV31Info:
    """Reward v3.1, targeted from the 50-failure audit.

    INHOMOGENEOUS L[y]=g:
      Q_force = worst-domain forcing-normalized score over negative/central/positive domains.
      Correct parameter directions are constraints: bad L[dh/dc] gets penalized.
      Symbolically dependent/missing integration directions are hard-capped.

    HOMOGENEOUS L[y]=0:
      Score each parameter direction INDEPENDENTLY on all three domains.
      Q_hom = min(worst-domain direction scores, base-component score).
      No global candidate q_equation enters the reward.
      Symbolic Wronskian dependence is a hard cap, not a soft penalty.

    Numerical independence is logged only for diagnosis. It cannot penalize a
    mathematically independent but stiff pair.
    """
    x = env.local_dict["x"]
    coeffs = _coeff_symbols(env, hyp)
    rng = np.random.RandomState(int(probe_seed) + 17)
    domains = _random_domains(int(probe_seed), n_points_per_domain)
    draws = max(1, n_coeff_draws if coeffs else 1)

    try:
        a2, a1, a0, g = linear_ode_components(env, problem.equation)
        is_inhom = bool(sp.simplify(g) != 0)

        hp = sp.diff(hyp, x)
        hpp = sp.diff(hyp, x, 2)
        Lh = sp.expand(a2*hpp + a1*hp + a0*hyp)
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

        direction_exprs=[]
        tangent_fns=[]
        jet_fns=[]
        for c in coeffs:
            d0=sp.diff(hyp,c); d1=sp.diff(d0,x); d2=sp.diff(d0,x,2)
            direction_exprs.append(d0)
            fd0=sp.lambdify(args,d0,modules=["numpy"])
            fd1=sp.lambdify(args,d1,modules=["numpy"])
            fd2=sp.lambdify(args,d2,modules=["numpy"])
            tangent_fns.append((fd0,fd1,fd2)); jet_fns.append((fd0,fd1))

        gate_pass, wr_zero, W = _symbolic_wronskian_gate(direction_exprs, x)
        missing=max(0,2-len(coeffs))
        if missing:
            gate_pass=False

        # Homogeneous base component catches exact directions + unrelated wrong offset.
        zero_subs={c:0 for c in coeffs}
        base_expr=sp.simplify(hyp.subs(zero_subs)) if coeffs else hyp
        bp=sp.diff(base_expr,x); bpp=sp.diff(base_expr,x,2)
        fn_b=sp.lambdify([x],base_expr,modules=["numpy"])
        fn_bp=sp.lambdify([x],bp,modules=["numpy"])
        fn_bpp=sp.lambdify([x],bpp,modules=["numpy"])

        eq_domain_errors=[[] for _ in domains]
        dir_domain_errors=[[[] for _ in domains] for _ in tangent_fns]
        base_domain_errors=[[] for _ in domains]
        valid_count=0; valid_total=0

        for di,xs in enumerate(domains):
            shape=xs.shape
            a2v,va2=_eval_real(lambda xx,*u: fn_a2(xx),xs,[],shape)
            a1v,va1=_eval_real(lambda xx,*u: fn_a1(xx),xs,[],shape)
            a0v,va0=_eval_real(lambda xx,*u: fn_a0(xx),xs,[],shape)
            gv,vg=_eval_real(lambda xx,*u: fn_g(xx),xs,[],shape)
            operator_valid=va2&va1&va0&vg

            # base component is coefficient independent
            bv,vb=_eval_real(lambda xx,*u: fn_b(xx),xs,[],shape)
            bpv,vbp=_eval_real(lambda xx,*u: fn_bp(xx),xs,[],shape)
            bppv,vbpp=_eval_real(lambda xx,*u: fn_bpp(xx),xs,[],shape)
            berr,bn=_stable_relative_operator_error(a2v,a1v,a0v,bv,bpv,bppv,operator_valid&vb&vbp&vbpp)
            if bn: base_domain_errors[di].append(berr)

            for _ in range(draws):
                cvals=rng.uniform(-1.7,1.7,size=len(coeffs)).tolist()
                hv,vh=_eval_real(fn_h,xs,cvals,shape)
                hpv,vhp=_eval_real(fn_hp,xs,cvals,shape)
                hppv,vhpp=_eval_real(fn_hpp,xs,cvals,shape)
                rv,vr=_eval_real(fn_res,xs,cvals,shape)
                base_valid=operator_valid&vh&vhp&vhpp&vr
                valid_count+=int(np.sum(base_valid)); valid_total+=int(base_valid.size)

                if is_inhom:
                    er,n=_stable_forcing_error(rv,gv,base_valid)
                    if n: eq_domain_errors[di].append(er)

                for ti,(fd0,fd1,fd2) in enumerate(tangent_fns):
                    d0v,vd0=_eval_real(fd0,xs,cvals,shape)
                    d1v,vd1=_eval_real(fd1,xs,cvals,shape)
                    d2v,vd2=_eval_real(fd2,xs,cvals,shape)
                    er,n=_stable_relative_operator_error(a2v,a1v,a0v,d0v,d1v,d2v,operator_valid&vd0&vd1&vd2)
                    if n: dir_domain_errors[ti][di].append(er)

        validity=float(valid_count/max(1,valid_total))
        invalid_penalty=float(invalid_weight)*(1.0-validity)
        missing_penalty=float(missing_constant_penalty_each)*missing

        # Inhom equation score: worst domain, each domain averaged over coefficient draws.
        eq_domain_scores=[]
        for vals in eq_domain_errors:
            er=float(np.mean(np.clip(np.asarray(vals),0,1e12))) if vals else 1e12
            eq_domain_scores.append(_score_from_error(er,-6.0,score_cap))
        q_eq=_aggregate_domain_scores(eq_domain_scores) if eq_domain_scores else (-6.0 if is_inhom else 0.0)

        # Each direction gets its own worst-domain score. Then the worst direction wins.
        direction_scores=[]; direction_domain_scores=[]
        for perdom in dir_domain_errors:
            ds=[]
            for vals in perdom:
                er=float(np.mean(np.clip(np.asarray(vals),0,1e12))) if vals else 1e12
                ds.append(_score_from_error(er,-6.0,score_cap))
            direction_domain_scores.append(ds)
            direction_scores.append(_aggregate_domain_scores(ds) if ds else -6.0)
        q_dir_min=float(min(direction_scores)) if direction_scores else -6.0

        # Base component score for homogeneous equations. A zero base is perfect.
        base_domain_scores=[]
        if sp.simplify(base_expr)==0:
            base_domain_scores=[float(score_cap)]*len(domains)
            q_base=float(score_cap)
        else:
            for vals in base_domain_errors:
                er=float(np.mean(np.clip(np.asarray(vals),0,1e12))) if vals else 1e12
                base_domain_scores.append(_score_from_error(er,-6.0,score_cap))
            q_base=_aggregate_domain_scores(base_domain_scores) if base_domain_scores else -6.0

        indep=_diagnostic_independence(jet_fns,coeffs,np.random.RandomState(int(probe_seed)+7919))

        direction_penalty=0.0
        if is_inhom:
            direction_penalty=float(direction_penalty_weight)*max(0.0,float(direction_gate)-q_dir_min)
            core=q_eq-direction_penalty
        else:
            # NO global candidate q_equation. Every free direction and the base
            # offset must independently satisfy the homogeneous operator.
            core=min(q_dir_min,q_base)
            q_eq=0.0  # intentionally not part of v3.1 homogeneous reward
            eq_domain_scores=[]

        pre=core-missing_penalty-invalid_penalty
        hard_penalty=0.0
        if not gate_pass:
            capped=min(float(pre),float(hard_generality_cap))
            hard_penalty=max(0.0,float(pre)-capped)
            pre=capped

        return RewardV31Info(
            pre_reward=float(pre),
            equation_score=float(q_eq),
            equation_domain_scores=[float(v) for v in eq_domain_scores],
            direction_min_score=float(q_dir_min),
            direction_scores=[float(v) for v in direction_scores],
            direction_domain_scores=[[float(z) for z in row] for row in direction_domain_scores],
            base_score=float(q_base),
            base_domain_scores=[float(v) for v in base_domain_scores],
            independence_score=float(indep),
            validity=float(validity),
            direction_penalty=float(direction_penalty),
            missing_constant_penalty=float(missing_penalty),
            invalid_penalty=float(invalid_penalty),
            hard_generality_penalty=float(hard_penalty),
            generality_gate_pass=bool(gate_pass),
            wronskian_zero=bool(wr_zero),
            wronskian_expression=str(W) if W is not None else None,
            is_linear=True,
            is_inhomogeneous=is_inhom,
            n_constants=len(coeffs),
            expression=str(hyp),
        )
    except BaseException as e:
        return RewardV31Info(
            pre_reward=-25.0,equation_score=-6.0,equation_domain_scores=[],
            direction_min_score=-6.0,direction_scores=[],direction_domain_scores=[],
            base_score=-6.0,base_domain_scores=[],independence_score=0.0,validity=0.0,
            direction_penalty=10.0,missing_constant_penalty=8.0,invalid_penalty=5.0,
            hard_generality_penalty=4.0,generality_gate_pass=False,wronskian_zero=True,
            wronskian_expression=None,is_linear=False,is_inhomogeneous=False,n_constants=0,
            expression=str(hyp) if hyp is not None else None,error=f"{type(e).__name__}: {e}")


def score_reward_v31_ids(env, problem, token_ids: Sequence[int], timeout_s: float = 6.0, probe_seed: int = 0):
    hyp=None
    try:
        with _time_limit(timeout_s):
            from ttrl_lane_emden.core import ids_to_sympy
            hyp=ids_to_sympy(env,token_ids)
            return rich_pre_exact_reward_v31(env,problem,hyp,probe_seed=probe_seed)
    except RewardV31Timeout as e:
        return RewardV31Info(-25.0,-6.0,[],-6.0,[],[],-6.0,[],0.0,0.0,10.0,8.0,5.0,4.0,False,True,None,False,False,0,str(hyp) if hyp is not None else None,f"TIMEOUT: {e}")
    except BaseException as e:
        return RewardV31Info(-25.0,-6.0,[],-6.0,[],[],-6.0,[],0.0,0.0,10.0,8.0,5.0,4.0,False,True,None,False,False,0,str(hyp) if hyp is not None else None,f"{type(e).__name__}: {e}")


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def _worker_score(payload):
    env,problem,timeout_s,ids,probe_seed=payload
    return score_reward_v31_ids(env,problem,ids,timeout_s,probe_seed)


class ParallelRewardV31Pool:
    """Reward-v3.1 pool with worker-local and parent hard timeouts."""
    def __init__(self,workers:int,watchdog_slack_s:float=3.0):
        self.workers=max(1,int(workers))
        self.watchdog_slack_s=float(watchdog_slack_s)
        self._ctx=mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn") if self.workers>1 else None
        self._pool=None
        if self.workers>1:
            self._start_pool()

    def _start_pool(self):
        if self.workers>1:
            self._pool=self._ctx.Pool(self.workers,initializer=_worker_init)

    def _restart_pool(self):
        if self._pool is not None:
            self._pool.terminate(); self._pool.join(); self._pool=None
        self._start_pool()

    @staticmethod
    def _hard_timeout_info(message):
        return RewardV31Info(
            -25.0,-6.0,[],-6.0,[],[],-6.0,[],0.0,0.0,
            10.0,8.0,5.0,4.0,False,True,None,False,False,0,
            None,f"TIMEOUT: {message}"
        )

    def _score_wave(self,payloads,timeout_s:float):
        jobs=[self._pool.apply_async(_worker_score,(payload,)) for payload in payloads]
        out=[None]*len(jobs)
        pending=set(range(len(jobs)))
        hard_s=max(2.0,float(timeout_s)+self.watchdog_slack_s)
        deadline=time.monotonic()+hard_s
        while pending:
            for i in list(pending):
                job=jobs[i]
                if not job.ready():
                    continue
                try:
                    out[i]=job.get(timeout=0)
                except BaseException as e:
                    out[i]=self._hard_timeout_info(f"worker error {type(e).__name__}: {e}")
                pending.remove(i)
            if not pending:
                break
            if time.monotonic()>=deadline:
                for i in pending:
                    out[i]=self._hard_timeout_info(f"parent hard watchdog exceeded {hard_s:g}s")
                self._restart_pool()
                break
            time.sleep(0.01)
        return out

    def score_many(self,env,problem,timeout_s:float,id_batches,probe_seed:int=0):
        batches=[list(map(int,ids)) for ids in id_batches]
        if not batches:return []
        if self._pool is None:
            return [score_reward_v31_ids(env,problem,ids,timeout_s,probe_seed) for ids in batches]
        payloads=[(env,problem,float(timeout_s),ids,int(probe_seed)) for ids in batches]
        results=[]
        for s in range(0,len(payloads),self.workers):
            results.extend(self._score_wave(payloads[s:s+self.workers],timeout_s))
        return results

    def close(self):
        if self._pool is not None:
            self._pool.close();self._pool.join();self._pool=None
    def terminate(self):
        if self._pool is not None:
            self._pool.terminate();self._pool.join();self._pool=None
    def __enter__(self):return self
    def __exit__(self,exc_type,exc,tb): self.close() if exc_type is None else self.terminate()
