#!/usr/bin/env python3
import os
import sys
import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import default_env_for_verifier, make_problem, score_general_candidate, residual_expression, exact_zero


def show(name, info):
    print(f"{name:28s} reward={info.reward:8.3f} res0={int(info.exact_residual_zero)} rank={info.generality_rank} mse={info.numerical_mse:.3e}")
    print(f"  expr: {info.expression}")
    print(f"  residual: {info.residual}")


def main():
    env = default_env_for_verifier()
    x = env.local_dict['x']
    a0 = env.local_dict['a0']
    a1 = env.local_dict['a1']

    tests = []

    harmonic = make_problem(env, 'harmonic')
    tests += [
        ('harmonic general', harmonic, a0*sp.sin(x) + a1*sp.cos(x), True, 2),
        ('harmonic particular', harmonic, sp.sin(x), True, 0),
        ('harmonic zero', harmonic, sp.Integer(0), True, 0),
        ('harmonic wrong', harmonic, sp.exp(x), False, 0),
    ]

    le1 = make_problem(env, 'lane_emden', n=1, mode='general')
    tests += [
        ('LE1 general', le1, (a0*sp.sin(x) + a1*sp.cos(x))/x, True, 2),
        ('LE1 physical particular', le1, sp.sin(x)/x, True, 0),
        ('LE1 wrong', le1, sp.sin(x), False, 0),
    ]

    le0 = make_problem(env, 'lane_emden', n=0, mode='general')
    tests += [
        ('LE0 general', le0, a0 + a1/x - x**2/sp.Integer(6), True, 2),
        ('LE0 physical particular', le0, 1 - x**2/sp.Integer(6), True, 0),
    ]

    failed = 0
    for name, problem, expr, want_zero, want_rank in tests:
        info = score_general_candidate(env, problem, expr)
        show(name, info)
        if info.exact_residual_zero != want_zero or info.generality_rank != want_rank:
            print(f"  FAIL expected res0={want_zero}, rank={want_rank}")
            failed += 1

    # n=5 IVP exact solution: residual check only in this first POC.
    le5 = make_problem(env, 'lane_emden', n=5, mode='ivp')
    residual = residual_expression(env, le5.equation, le5.expected)
    is_zero, simplified = exact_zero(residual, seconds=2)
    print(f"{'LE5 physical residual':28s} res0={int(is_zero)} expr={le5.expected}")
    print(f"  residual: {simplified}")
    if not is_zero:
        failed += 1

    if failed:
        raise SystemExit(f"{failed} verifier tests failed")
    print("\nALL VERIFIER TESTS PASSED")


if __name__ == '__main__':
    main()
