#!/usr/bin/env python3
import os, sys
from types import SimpleNamespace
import sympy as sp

ROOT=os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from ttrl_ode_benchmark.rich_reward_v3 import rich_pre_exact_reward_v3

x=sp.Symbol('x', real=True)
f=sp.Function('f')
a8,a9=sp.symbols('a8 a9', real=True)
env=SimpleNamespace(local_dict={'x':x,'f':f,'a8':a8,'a9':a9}, coefficients={'a8':a8,'a9':a9})

def problem(eq): return SimpleNamespace(equation=eq)

def show(title, p, seq):
    print('\n'+title)
    vals=[]
    for name,h in seq:
        z=rich_pre_exact_reward_v3(env,p,h,seed=0)
        vals.append(z.pre_reward)
        print(f"{name:18s} R3={z.pre_reward:8.3f} eq={z.equation_score:7.3f} dmin={z.direction_min_score:7.3f} G={z.independence_score:7.4f} Pdir={z.direction_penalty:6.3f} PG={z.independence_penalty:6.3f} :: {sp.sstr(h)}")
    assert all(vals[i+1] > vals[i] for i in range(len(vals)-1)), (title, vals)

# H60: exact roots -15, -33. Move second root -35 -> -34 -> -33.
p60=problem(sp.diff(f(x),x,2)+48*sp.diff(f(x),x)+495*f(x))
show('H60 root path -35 -> -34 -> -33', p60, [
    ('k=35', a8*sp.exp(-15*x)+a9*sp.exp(-35*x)),
    ('k=34', a8*sp.exp(-15*x)+a9*sp.exp(-34*x)),
    ('k=33 exact', a8*sp.exp(-15*x)+a9*sp.exp(-33*x)),
])

# H367: exact exp(-6x)(A cosh 34x + B sinh 34x), represented by amplitude/phase.
p367=problem(sp.diff(f(x),x,2)+12*sp.diff(f(x),x)-1120*f(x))
show('H367 hyperbolic frequency 36 -> 35 -> 34', p367, [
    ('k=36', sp.exp(a9-6*x)*sp.cosh(a8+36*x)),
    ('k=35', sp.exp(a9-6*x)*sp.cosh(a8+35*x)),
    ('k=34 exact', sp.exp(a9-6*x)*sp.cosh(a8+34*x)),
])

# H1264: 4y''-5y'=x^2; exact particular polynomial path.
p1264=problem(4*sp.diff(f(x),x,2)-5*sp.diff(f(x),x)-x**2)
hom1264=a8+a9*sp.exp(sp.Rational(5,4)*x)
show('H1264 polynomial particular completion', p1264, [
    ('cubic only', hom1264-sp.Rational(1,15)*x**3),
    ('+ quadratic', hom1264-sp.Rational(1,15)*x**3-sp.Rational(4,25)*x**2),
    ('exact particular', hom1264-sp.Rational(1,15)*x**3-sp.Rational(4,25)*x**2-sp.Rational(32,125)*x),
])

# H1516: 4y''+4y'=cos 2x; exact particular -cos/20+sin/40.
p1516=problem(4*sp.diff(f(x),x,2)+4*sp.diff(f(x),x)-sp.cos(2*x))
hom1516=a8+a9*sp.exp(-x)
show('H1516 trig particular completion', p1516, [
    ('-cos/16', hom1516-sp.cos(2*x)/16),
    ('-cos/20', hom1516-sp.cos(2*x)/20),
    ('exact +sin/40', hom1516-sp.cos(2*x)/20+sp.sin(2*x)/40),
])

# Degeneracy gate: one-mode disguised with two amplitude constants should be penalized.
p945=problem(sp.diff(f(x),x,2)+44*sp.diff(f(x),x)+403*f(x))
deg=(sp.exp(a8)+sp.exp(a9))*sp.exp(-17*x)
two=a8*sp.exp(-13*x)+a9*sp.exp(-31*x)
print('\nDegeneracy gate sanity')
for name,h in [('degenerate same mode',deg),('exact two modes',two)]:
    z=rich_pre_exact_reward_v3(env,p945,h,seed=0)
    print(f"{name:20s} R3={z.pre_reward:8.3f} dmin={z.direction_min_score:7.3f} G={z.independence_score:7.4f} PG={z.independence_penalty:6.3f}")
assert rich_pre_exact_reward_v3(env,p945,two).pre_reward > rich_pre_exact_reward_v3(env,p945,deg).pre_reward
print('\nREWARD V3 PATH TESTS PASSED')
