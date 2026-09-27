#!/usr/bin/env python3
import os, sys
import sympy as sp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import default_env_for_verifier
from ttrl_lane_emden.ivp_n2 import LaneEmdenN2Verifier

env = default_env_for_verifier()
v = LaneEmdenN2Verifier(env)
x = env.local_dict['x']
a8 = env.local_dict['a8']
a9 = env.local_dict['a9']

# High-order local n=2 series: should satisfy the IVP near the origin and be
# substantially better than trivial/wrong functions, though not globally exact.
series10 = 1 - x**2/sp.Integer(6) + x**4/sp.Integer(60) - 11*x**6/sp.Integer(7560) + x**8/sp.Integer(8505) - 97*x**10/sp.Integer(10692000)
good_global = sp.exp(-x**2/sp.Integer(6))
wrong = sp.exp(-x)
const = sp.Integer(1)
param = a8 + a9*x - x**2/sp.Integer(6)

# Residual-hacking family: after coefficient fitting it can make the cleared
# ODE residual almost zero by collapsing toward c/x, while badly violating
# the physical IVP/reference. The joint reward must rank it poorly.
residual_hacker = a8 * sp.sin(a9 - (2*x + 4)/(x + 2)) / x

for name, expr in [('good_global', good_global), ('series10', series10), ('wrong_exp', wrong), ('constant', const), ('param_family', param), ('resid_hacker', residual_hacker)]:
    z = v.score_expr(expr)
    print(f"{name:14s} reward={z.reward:8.3f} certified={int(z.certified_ivp)} ode={z.ode_rel_mse:.3e} ref={z.ref_nrmse:.3e} anchor={z.anchor_rmse:.3e}")
    print('  fitted:', z.fitted_expression)
    print('  coeffs:', z.fitted_coefficients)

z_good = v.score_expr(good_global)
z_wrong = v.score_expr(wrong)
assert z_good.valid_parse
assert z_good.reward > z_wrong.reward
assert z_good.ref_nrmse < z_wrong.ref_nrmse
assert z_good.certified_ivp
z_hacker = v.score_expr(residual_hacker)
assert z_hacker.ode_rel_mse < 1e-4
reasonable = v.score_expr(sp.sin(x)/x)
assert reasonable.reward > z_hacker.reward
print('\nN2 IVP VERIFIER SMOKE TEST PASSED')
