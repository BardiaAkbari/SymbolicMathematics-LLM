#!/usr/bin/env python3
"""Time pure reward evaluation on 128 synthetic expressions (no model)."""
import os
import sys
import time

import numpy as np
import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import default_env_for_verifier
from ttrl_lane_emden.ivp_n5 import LaneEmdenN5Verifier


def main():
    env = default_env_for_verifier()
    v = LaneEmdenN5Verifier(env, exact_check=False)  # dense path only
    x = env.local_dict["x"]
    a8, a9 = sp.symbols("a8 a9")

    # mix of families the model might emit
    pool = [
        1 / sp.sqrt(1 + x**2 / 3),
        sp.sin(x) / x,
        sp.cos(x),
        sp.exp(-x**2 / 2),
        1 - x**2 / 6,
        a8 * sp.cos(a9 + x),
        a8 / (1 + a9 * x**2),
        sp.sqrt(1 + x**2),
        1 / (1 + x**2),
        sp.exp(-x),
    ]
    exprs = [pool[i % len(pool)] for i in range(128)]

    # warm-up
    for e in exprs[:4]:
        v.score_expr(e, 10)

    t0 = time.perf_counter()
    infos = [v.score_expr(e, 10) for e in exprs]
    dt = time.perf_counter() - t0

    rewards = np.array([z.reward for z in infos])
    print(f"n=128  total={dt:.3f}s  per_expr={dt/128*1e3:.2f} ms")
    print(f"reward mean={rewards.mean():.3f}  max={rewards.max():.3f}  finite={sum(z.finite for z in infos)}")
    print("REWARD BATCH BENCH DONE")


if __name__ == "__main__":
    main()
