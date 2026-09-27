#!/usr/bin/env python3
import os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0,ROOT)

from ttrl_lane_emden.core import default_env_for_verifier, equation_to_tokens, residual_expression, exact_zero, numeric_generality_rank, exact_generality
from ttrl_ode_benchmark.latex_ode import parse_ode2_equation, parse_reference_solution

# Real examples from the public HSE `equations` corpus that preceded the 300k release.
# The middle row is intentionally mislabeled in the public CSV; our preprocessing
# must reject it rather than silently treating the reference as ground truth.
CASES=[
    (True, r'y\prime\prime -2y\prime - 3y = 5 e^{7x}', r'y = C_1 e^{3x} + C_2 e^{-x}+\frac{5}{32}e^{7x}'),
    (False, r'y\prime\prime-2y\prime-3y= 4 \sin 2x', r'y = C_1 e^{3x} + C_2 e^{-x}  -\frac{28}{65} \cos 2x + \frac{16}{65} \sin 2x'),
    (True, r'y\prime\prime - y =x', r'y= C_1 e^x + C_2 e^{-x} - x'),
    (True, r'y\prime\prime - 6y\prime +9y = 3 e^{3x}- e^{4x}', r'y = C_1 e^{3x} + C_2 x e^{3x}+\frac{3}{2} x^2 e^{3x} - e^{4x}'),
]

env=default_env_for_verifier()
for i,(expected_valid,eqt,anst) in enumerate(CASES):
    eq=parse_ode2_equation(eqt,env)
    ref=parse_reference_solution(anst,env,2)
    toks=equation_to_tokens(env,eq)
    res=residual_expression(env,eq,ref)
    zero,_=exact_zero(res,seconds=2)
    rank=numeric_generality_rank(env,ref,2)
    gen=exact_generality(env,ref,2,seconds=1)
    verified=bool(zero and rank==2 and gen)
    print(i, 'verified=',verified,'expected=',expected_valid)
    print('  eq=',eq)
    print('  ref=',ref)
    print('  prefix=',' '.join(toks))
    assert verified == expected_valid
print('HSE ADAPTER SMOKE TEST PASSED')
