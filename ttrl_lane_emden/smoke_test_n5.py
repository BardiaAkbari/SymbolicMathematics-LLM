#!/usr/bin/env python3
"""Fast verifier smoke test; does not train the model."""
import argparse
import os, sys
import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import load_pretrained
from ttrl_lane_emden.ivp_n5 import LaneEmdenN5Verifier, BAD_REWARD

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--cpu", action="store_true")
args = p.parse_args()

device = "cpu" if args.cpu else None
# load_pretrained expects a torch.device-like value
import torch
dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
env, _, _, _, _ = load_pretrained(args.checkpoint, dev)
v = LaneEmdenN5Verifier(env)
x = env.local_dict["x"]
a8, a9 = sp.symbols("a8 a9")

cases = {
    "coefficient_candidate": a8 * sp.cos(a9 + x),
    "bad_candidate": sp.sin(x),
    "known_exact_solution_smoke_only": 1 / sp.sqrt(1 + x**2 / 3),
}

for name, expr in cases.items():
    z = v.score_expr(expr, sequence_length=10)
    print(name)
    print("  expr   =", expr)
    print("  reward =", z.reward)
    print("  ode    =", z.ode_residual)
    print("  anchor =", z.anchor_error)
    print("  exact  =", z.exact)

z = v.score_expr(cases["coefficient_candidate"], sequence_length=10)
assert z.reward > BAD_REWARD, "a8/a9 candidates are being treated as invalid"
assert z.finite, "finite coefficient candidate was marked non-finite"
z2 = v.score_expr(cases["known_exact_solution_smoke_only"], sequence_length=10)
assert z2.exact, "exact n=5 solution did not pass symbolic IVP/O​DE certification"
print("SMOKE TEST PASSED")
