#!/usr/bin/env python3
import os,sys
from types import SimpleNamespace
import sympy as sp
ROOT=os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:sys.path.insert(0,ROOT)
from ttrl_ode_benchmark.rich_reward_v31 import rich_pre_exact_reward_v31

x=sp.Symbol('x',real=True);f=sp.Function('f');a8,a9=sp.symbols('a8 a9',real=True)
env=SimpleNamespace(local_dict={'x':x,'f':f,'a8':a8,'a9':a9},coefficients={'a8':a8,'a9':a9})
D=[
('10880',sp.diff(f(x),x,2)+11*sp.diff(f(x),x)-726*f(x),a8*sp.exp(-3*x)*sp.sinh(a9+25*x),14.9956,4.0,True),
('4945',sp.diff(f(x),x,2)+64*sp.diff(f(x),x)+735*f(x),(a8*sp.exp(x)+sp.exp(x))*sp.exp(a9-16*x),11.0,-3.99,False),
('12821',sp.diff(f(x),x,2)-49*sp.diff(f(x),x)+328*f(x),a9*(sp.exp(8*x)+4*sp.exp(a8+8*x)),11.0,-3.99,False),
('801',sp.diff(f(x),x,2)-30*sp.diff(f(x),x)-864*f(x),a8*sp.exp(45*x+25/sp.asinh(3))*sp.cosh(a9+3*x),6.358,3.0,True),
('11066',sp.diff(f(x),x,2)-15*sp.diff(f(x),x)-1134*f(x),a9*(a8-1)*sp.exp(-27*x),11.0,-3.99,False),
('14636',sp.diff(f(x),x,2)-5*sp.diff(f(x),x)-1116*f(x),a8*sp.exp(2*x)*sp.sinh(a9-34*x),15.0,4.0,True),
('3602',sp.diff(f(x),x,2)+3*sp.diff(f(x),x)-154*f(x),sp.exp(a8-x)*sp.cosh(a9+12*x),11.449,4.0,True),
]
for k,eq,h,v3,upper,gate_expected in D:
 z=rich_pre_exact_reward_v31(env,SimpleNamespace(equation=eq),h,probe_seed=12345+int(k))
 print(f'{k:>5s}: old-v3={v3:7.3f} -> v3.1={z.pre_reward:7.3f} gate={int(z.generality_gate_pass)} W0={int(z.wronskian_zero)}')
 assert z.pre_reward <= upper+1e-9,(k,z.pre_reward,upper)
 assert z.generality_gate_pass==gate_expected,(k,z.generality_gate_pass,gate_expected)
print('KNOWN V3 FAILURE REGRESSIONS PASSED')
