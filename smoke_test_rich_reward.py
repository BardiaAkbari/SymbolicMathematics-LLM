#!/usr/bin/env python3
import os, sys
import sympy as sp
ROOT=os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from ttrl_lane_emden.core import default_env_for_verifier, Problem, score_general_candidate
from ttrl_ode_benchmark.rich_reward import rich_pre_exact_reward

env=default_env_for_verifier(); x=env.local_dict['x']; f=env.local_dict['f']; a8=env.local_dict['a8']; a9=env.local_dict['a9']
eq=sp.diff(f(x),x,2)-f(x)-sp.exp(2*x)
p=Problem('rich_reward_smoke',eq,2,'general',residual_terms=tuple(sp.Add.make_args(sp.expand(eq))))
exact=a8*sp.exp(x)+a9*sp.exp(-x)+sp.exp(2*x)/3
hom=a8*sp.exp(x)+a9*sp.exp(-x)
wrong=a8*sp.sin(x)+a9*sp.cos(x)
rows=[]
for name,h in [('exact',exact),('homogeneous_only',hom),('wrong_family',wrong)]:
    old=score_general_candidate(env,p,h)
    new=rich_pre_exact_reward(env,p,h)
    rows.append((name,old,new))
    print(f"{name:16s} exact={int(old.verified_general)} old={old.reward:8.3f} new_pre={new.pre_reward:8.3f} eq={new.equation_score:7.3f} dir={new.constant_direction_score:7.3f} G={new.independence_score:.3f} V={new.validity:.2f}")
assert rows[0][1].verified_general
assert not rows[1][1].verified_general and not rows[2][1].verified_general
assert rows[0][2].pre_reward > rows[1][2].pre_reward > rows[2][2].pre_reward
assert rows[1][2].constant_direction_score > rows[2][2].constant_direction_score
print('RICH REWARD SMOKE TEST PASSED')
