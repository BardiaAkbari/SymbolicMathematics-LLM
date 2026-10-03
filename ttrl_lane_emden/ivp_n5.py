"""Reference-free Lane-Emden n=5 IVP verifier."""
from __future__ import annotations
import math, warnings
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import sympy as sp
from ttrl_lane_emden.core import _coefficient_symbols, generated_to_candidates, ids_to_sympy

@dataclass
class N5Info:
    reward: float
    valid_parse: bool
    certified_ivp: bool
    elite_eligible: bool
    ode_residual: float
    anchor_error: float
    expression: Optional[sp.Expr]
    exact: bool
    error: Optional[str] = None

    @property
    def reference_error(self):
        return float("nan")

class LaneEmdenN5Verifier:
    """Score only from x*y'' + 2*y' + x*y^5 = 0 and y(0)=1,y'(0)=0."""
    def __init__(self, env, x_eps=1e-3, x_max=4.0, n_ode_points=48,
                 certify_ode=1e-10, certify_anchor=1e-6,
                 elite_ode=1e-3, elite_anchor=1e-3, length_penalty=0.01):
        self.env = env
        self.x = env.local_dict["x"]
        self.x_eps = float(x_eps)
        self.certify_ode = float(certify_ode)
        self.certify_anchor = float(certify_anchor)
        self.elite_ode = float(elite_ode)
        self.elite_anchor = float(elite_anchor)
        self.length_penalty = float(length_penalty)
        self.ode_x = np.linspace(max(self.x_eps, 0.05), float(x_max), n_ode_points)
        e = self.x_eps
        self.anchor_y = 1.0 - e**2/6.0 + e**4/120.0
        self.anchor_dy = -e/3.0 + e**3/30.0

    def _has_coefficients(self, expr):
        return bool(_coefficient_symbols(self.env, expr))

    def _anchor_error(self, expr):
        try:
            yp = sp.diff(expr, self.x)
            fy = sp.lambdify(self.x, expr, "numpy")
            fyp = sp.lambdify(self.x, yp, "numpy")
            with np.errstate(all="ignore"):
                y = float(fy(self.x_eps)); dy = float(fyp(self.x_eps))
            if not (math.isfinite(y) and math.isfinite(dy)): return 1e6
            return float(np.sqrt(((y-self.anchor_y)**2+(dy-self.anchor_dy)**2)/2.0))
        except Exception: return 1e6

    def _ode_error(self, expr):
        try:
            residual = self.x*sp.diff(expr,self.x,2)+2*sp.diff(expr,self.x)+self.x*expr**5
            fn = sp.lambdify(self.x, residual, "numpy")
            with np.errstate(all="ignore"): r = np.asarray(fn(self.ode_x), dtype=float)
            if not np.all(np.isfinite(r)): return 1e6
            return float(np.mean(r*r))
        except Exception: return 1e6

    def _exact_ivp(self, expr):
        try:
            yp = sp.diff(expr,self.x)
            residual = self.x*sp.diff(expr,self.x,2)+2*yp+self.x*expr**5
            if sp.simplify(residual) != 0: return False
            return (sp.simplify(sp.limit(expr,self.x,0,dir="+")-1)==0 and
                    sp.simplify(sp.limit(yp,self.x,0,dir="+"))==0)
        except Exception: return False

    def score_expr(self, expr, sequence_length=0):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if self._has_coefficients(expr):
                    return N5Info(-10.0,True,False,False,1e6,1e6,expr,False,"free coefficients are not allowed")
                ode, anchor = self._ode_error(expr), self._anchor_error(expr)
                ode_score = float(np.clip(-math.log10(ode+1e-12),-6,12))
                anchor_score = float(np.clip(-math.log10(anchor+1e-12),-6,12))
                reward = ode_score + anchor_score - self.length_penalty*sequence_length
                exact = self._exact_ivp(expr)
                certified = ode <= self.certify_ode and anchor <= self.certify_anchor
                elite = ode <= self.elite_ode and anchor <= self.elite_anchor
                if exact: reward += 20.0; certified = elite = True
                return N5Info(float(reward),True,certified,elite,ode,anchor,expr,exact)
        except Exception as e:
            return N5Info(-20.0,False,False,False,1e6,1e6,expr,False,f"{type(e).__name__}: {e}")

    def score_ids(self, token_ids: Sequence[int]):
        try: return self.score_expr(ids_to_sympy(self.env,token_ids),len(token_ids))
        except Exception as e: return N5Info(-20.0,False,False,False,1e6,1e6,None,False,f"{type(e).__name__}: {e}")

    def evaluate_generated(self, generated, gen_len, cache=None):
        candidates = generated_to_candidates(self.env,generated,gen_len)
        infos=[]
        for ids,_ in candidates:
            key=tuple(int(i) for i in ids)
            if cache is not None and key in cache: info=cache[key]
            else:
                info=self.score_ids(ids)
                if cache is not None: cache[key]=info
            infos.append(info)
        return candidates,infos,0,0
