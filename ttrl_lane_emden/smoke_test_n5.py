#!/usr/bin/env python3
import os
import sys
import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import default_env_for_verifier
from ttrl_lane_emden.ivp_n5 import LaneEmdenN5Verifier


def main():
    env = default_env_for_verifier()
    x = env.local_dict["x"]
    verifier = LaneEmdenN5Verifier(env)

    exact = 1 / sp.sqrt(1 + x**2 / 3)
    wrong = 1 / sp.sqrt(1 + x**2 / 2)
    with_coeff = sp.Symbol("a8") / sp.sqrt(1 + x**2 / 3)

    a = verifier.score_expr(exact, sequence_length=9)
    b = verifier.score_expr(wrong, sequence_length=9)
    c = verifier.score_expr(with_coeff, sequence_length=9)

    print("exact:", a)
    print("wrong:", b)
    print("with coefficient:", c)

    assert a.exact
    assert a.ode_residual < 1e-8
    assert not b.exact
    # Coefficients are allowed for dense scoring and may be fitted.
    assert c.valid_parse
    assert c.fitted_expression is not None

    print("ALL N5 VERIFIER TESTS PASSED")


if __name__ == "__main__":
    main()
