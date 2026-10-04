#!/usr/bin/env python3
"""Deterministic smoke for the fast n=5 verifier. No training."""
import argparse
import os
import sys
import time

import sympy as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import load_pretrained, default_env_for_verifier
from ttrl_lane_emden.ivp_n5 import LaneEmdenN5Verifier, BAD_REWARD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="", help="optional; uses default env if empty")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if args.checkpoint:
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        env, _, _, _, _ = load_pretrained(args.checkpoint, device)
    else:
        env = default_env_for_verifier()

    v = LaneEmdenN5Verifier(env, exact_check=True)
    x = env.local_dict["x"]
    a8, a9 = sp.symbols("a8 a9")

    exact = 1 / sp.sqrt(1 + x**2 / 3)
    bad = sp.sin(x)
    coeff = a8 * sp.cos(a9 + x)

    print("=== smoke: exact solution ===")
    t0 = time.perf_counter()
    z = v.score_expr(exact, sequence_length=12)
    print(f"  reward={z.reward:.3f} ode={z.ode_residual:.3e} anchor={z.anchor_error:.3e} exact={z.exact}")
    assert z.exact, "exact n=5 solution failed symbolic certification"
    assert z.reward > 20.0, "exact solution should receive exact bonus"
    print(f"  time={time.perf_counter()-t0:.3f}s")

    print("=== smoke: wrong solution ===")
    z = v.score_expr(bad, sequence_length=8)
    print(f"  reward={z.reward:.3f} ode={z.ode_residual:.3e} anchor={z.anchor_error:.3e} exact={z.exact}")
    assert not z.exact
    assert z.reward < 5.0, "obviously wrong candidate scored too high"

    print("=== smoke: coefficient-bearing candidate ===")
    z = v.score_expr(coeff, sequence_length=10)
    print(f"  reward={z.reward:.3f} ode={z.ode_residual:.3e} anchor={z.anchor_error:.3e} exact={z.exact}")
    print(f"  fitted_coeffs={z.fitted_coefficients}")
    assert z.reward > BAD_REWARD, "a8/a9 candidate treated as invalid"
    assert z.finite, "finite coefficient candidate marked non-finite"
    assert not z.exact, "coefficient candidate must not be marked exact"

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
