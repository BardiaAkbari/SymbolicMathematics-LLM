#!/usr/bin/env python3
"""Compare current HSE ODE2 shaping reward with a forcing-aware residual.

This is a diagnostic only: it does not change TTRL or use dataset reference answers.
For inhomogeneous linear ODEs L[y]=g(x), the proposed dense residual is
    E |L[h]-g|^2 / (E |g|^2 + eps)
so adding/scaling a homogeneous component cannot artificially improve the score.
For homogeneous ODEs, it falls back to the current scale-invariant relative residual.
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np
import sympy as sp

ROOT=os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path: sys.path.insert(0,ROOT)

from ttrl_lane_emden.core import (
    default_env_for_verifier, Problem, relative_residual_mse,
    residual_expression, numeric_generality_rank, exact_zero, exact_generality,
)
from ttrl_ode_benchmark.common import make_problem_from_record


def coeff_symbols(env, expr):
    out=[]
    for i in range(10):
        s=env.local_dict.get(f'a{i}', sp.Symbol(f'a{i}'))
        if expr.has(s): out.append(s)
    return out


def forcing_split(env, problem: Problem):
    """Return dependent operator part and fixed forcing-side residual term.

    problem.equation is stored as F[y,x]=0. Terms independent of f(x) and its
    derivatives are the fixed forcing residual q(x), so F=L[y]+q=0 and g=-q.
    """
    f=env.local_dict['f']; x=env.local_dict['x']
    dep=[]; fixed=[]
    for t in sp.Add.make_args(sp.expand(problem.equation)):
        has_y = t.has(f(x)) or any(isinstance(d, sp.Derivative) and d.expr == f(x) for d in t.atoms(sp.Derivative))
        (dep if has_y else fixed).append(t)
    return sp.Add(*dep), sp.Add(*fixed)


def forcing_aware_mse(env, problem, hyp, x_min=0.25, x_max=4.0, n_points=24, n_coeff_draws=4, seed=0):
    x=env.local_dict['x']; f=env.local_dict['f']
    dep, fixed = forcing_split(env, problem)
    # Homogeneous problem: retain the old scale-invariant metric.
    if sp.simplify(fixed) == 0:
        return relative_residual_mse(env, problem, hyp), 'homogeneous_relative'

    dep_h = dep.subs(f(x), hyp).doit()
    residual = sp.expand(dep_h + fixed)
    coeffs=coeff_symbols(env,hyp)
    xs=np.linspace(x_min,x_max,n_points,dtype=np.float64)
    rng=np.random.RandomState(seed)
    try:
        rfn=sp.lambdify([x]+coeffs,residual,modules=['numpy'])
        gfn=sp.lambdify([x],-fixed,modules=['numpy'])
        gv=np.asarray(gfn(xs),dtype=np.complex128)
        if gv.ndim==0: gv=np.full(xs.shape,gv,dtype=np.complex128)
        gv=np.broadcast_to(gv,xs.shape)
        if not np.all(np.isfinite(gv)) or np.max(np.abs(gv.imag))>1e-7: return 1e12,'forcing_aware'
        gscale=float(np.mean(np.square(gv.real)))
        if not math.isfinite(gscale): return 1e12,'forcing_aware'
        vals=[]
        draws=max(1,n_coeff_draws if coeffs else 1)
        for _ in range(draws):
            cv=rng.uniform(-1.7,1.7,size=len(coeffs)).tolist()
            rv=np.asarray(rfn(xs,*cv),dtype=np.complex128)
            if rv.ndim==0: rv=np.full(xs.shape,rv,dtype=np.complex128)
            rv=np.broadcast_to(rv,xs.shape)
            if not np.all(np.isfinite(rv)) or np.max(np.abs(rv.imag))>1e-7: return 1e12,'forcing_aware'
            vals.append(rv.real)
        arr=np.concatenate(vals)
        mse=float(np.mean(np.square(np.clip(arr,-1e100,1e100))) / max(gscale,1e-12))
        return (mse if math.isfinite(mse) else 1e12),'forcing_aware'
    except Exception:
        return 1e12,'forcing_aware'


def score_from_mse(mse, cap=15.0):
    return float(np.clip(-math.log10(float(mse)+1e-15),-10.0,cap))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--problems',default='hse10_failures_blind.jsonl')
    a=ap.parse_args()
    env=default_env_for_verifier()
    print('id\tfamily\tcurrent_mse\tcurrent_score_cap6\tnew_mse\tnew_score_cap15\trank\told_greedy')
    for line in open(a.problems,encoding='utf-8'):
        rec=json.loads(line); problem=make_problem_from_record(env,rec)
        hyp=sp.sympify(rec['old_final_greedy_expr'],locals=env.local_dict)
        cur=relative_residual_mse(env,problem,hyp)
        new,mode=forcing_aware_mse(env,problem,hyp)
        rank=numeric_generality_rank(env,hyp,2)
        print(f"{rec['id']}\t{rec.get('failure_family','')}\t{cur:.3e}\t{min(6.0,max(-6.0,-math.log10(cur+1e-12))):.3f}\t{new:.3e}\t{score_from_mse(new):.3f}\t{rank}\t{hyp}")

if __name__=='__main__': main()
