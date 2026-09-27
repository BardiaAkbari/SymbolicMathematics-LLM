#!/usr/bin/env python3
import os,sys
from types import SimpleNamespace
import sympy as sp

ROOT=os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:sys.path.insert(0,ROOT)
from ttrl_ode_benchmark.rich_reward_v31 import rich_pre_exact_reward_v31
from ttrl_ode_benchmark.rich_reward_v3 import rich_pre_exact_reward_v3

x=sp.Symbol('x',real=True); f=sp.Function('f'); a8,a9=sp.symbols('a8 a9',real=True)
env=SimpleNamespace(local_dict={'x':x,'f':f,'a8':a8,'a9':a9},coefficients={'a8':a8,'a9':a9})
def P(eq):return SimpleNamespace(equation=eq)
def sc(p,h):return rich_pre_exact_reward_v31(env,p,h,probe_seed=123)
def show(label,p,h):
    v3=rich_pre_exact_reward_v3(env,p,h,seed=0)
    z=sc(p,h)
    print(f"{label:32s} v3={v3.pre_reward:8.3f} v31={z.pre_reward:8.3f} eq={z.equation_score:7.3f} dmin={z.direction_min_score:7.3f} base={z.base_score:7.3f} G={z.independence_score:8.5f} gate={int(z.generality_gate_pass)} W0={int(z.wronskian_zero)}")
    print('  dir-dom=',z.direction_domain_scores,'eq-dom=',z.equation_domain_scores)
    return z

print('\n1) Positive-domain hiding: exact growing mode + wrong decaying mode')
p=P(sp.diff(f(x),x,2)+11*sp.diff(f(x),x)-726*f(x)) # roots 22,-33
bad=sp.exp(-3*x)*sp.sinh(a8+25*x)*sp.exp(a9) # modes 22,-28
closer=a8*sp.exp(22*x)+a9*sp.exp(-31*x)
exact=a8*sp.exp(22*x)+a9*sp.exp(-33*x)
zbad=show('bad roots 22,-28',p,bad)
zclos=show('closer roots 22,-31',p,closer)
zexact=show('exact roots 22,-33',p,exact)
assert zbad.pre_reward < zclos.pre_reward < zexact.pre_reward,(zbad.pre_reward,zclos.pre_reward,zexact.pre_reward)

print('\n2) Near-root path -32 -> -31 with other mode exact 36')
p=P(sp.diff(f(x),x,2)-5*sp.diff(f(x),x)-1116*f(x))
z32=show('roots -32,36',p,a8*sp.exp(-32*x)+a9*sp.exp(36*x))
z31=show('roots -31,36 exact',p,a8*sp.exp(-31*x)+a9*sp.exp(36*x))
assert z32.pre_reward < z31.pre_reward

print('\n3) Duplicate exact mode must hard-fail')
p=P(sp.diff(f(x),x,2)+64*sp.diff(f(x),x)+735*f(x)) # -15,-49
deg=(sp.exp(a8)+sp.exp(a9))*sp.exp(-15*x)
true=a8*sp.exp(-15*x)+a9*sp.exp(-49*x)
zd=show('duplicate -15 mode',p,deg);zt=show('exact two modes',p,true)
assert not zd.generality_gate_pass and zd.pre_reward <= -4.0+1e-9
assert zt.generality_gate_pass and zt.pre_reward>zd.pre_reward

print('\n4) Base-offset corruption must be caught after removing global q_equation')
wrongbase=a8*sp.exp(-15*x)+a9*sp.exp(-49*x)+x
zb=show('exact dirs + wrong base x',p,wrongbase)
assert zb.pre_reward < zt.pre_reward

print('\n5) Polynomial particular path remains monotone')
p=P(4*sp.diff(f(x),x,2)-5*sp.diff(f(x),x)-x**2)
hom=a8+a9*sp.exp(sp.Rational(5,4)*x)
p1=hom-sp.Rational(1,15)*x**3
p2=p1-sp.Rational(4,25)*x**2
p3=p2-sp.Rational(32,125)*x
s1=show('cubic only',p,p1);s2=show('+quadratic',p,p2);s3=show('exact particular',p,p3)
assert s1.pre_reward < s2.pre_reward < s3.pre_reward

print('\n6) Trig particular path remains monotone')
p=P(4*sp.diff(f(x),x,2)+4*sp.diff(f(x),x)-sp.cos(2*x))
hom=a8+a9*sp.exp(-x)
t1=hom-sp.cos(2*x)/16
t2=hom-sp.cos(2*x)/20
t3=t2+sp.sin(2*x)/40
r1=show('-cos/16',p,t1);r2=show('-cos/20',p,t2);r3=show('exact +sin/40',p,t3)
assert r1.pre_reward < r2.pre_reward < r3.pre_reward

print('\nREWARD V3.1 SAFETY/PATH TESTS PASSED')
