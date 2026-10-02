#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import (
    load_pretrained,
    equation_to_tokens,
    encode_tokens,
)

from ttrl_lane_emden.ivp_n5 import LaneEmdenN5Verifier


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        required=True
    )

    p.add_argument(
        "--samples",
        type=int,
        default=2048
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    p.add_argument(
        "--temperature",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--max-len",
        type=int,
        default=128
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0
    )

    p.add_argument(
        "--cpu",
        action="store_true"
    )

    p.add_argument(
        "--save",
        default="n5_frozen_search.jsonl"
    )

    args = p.parse_args()

    seed_all(args.seed)

    device = torch.device(
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else "cuda"
    )

    print("device =", device)

    env, encoder, decoder, params, _ = load_pretrained(
        args.checkpoint,
        device
    )

    verifier = LaneEmdenN5Verifier(env)

    x = env.local_dict["x"]
    f = env.local_dict["f"]

    y = f(x)
    yp = torch.tensor(0.0, device=device)

    # x*y'' + 2*y' + x*y^5 = 0
    import sympy as sp

    equation = (
        x * sp.diff(y, x, 2)
        + 2 * sp.diff(y, x)
        + x * y**5
    )

    src_tokens = equation_to_tokens(
        env,
        equation
    )

    print("\nproblem: Lane-Emden n=5")
    print("equation:", equation)
    print("input prefix:", " ".join(src_tokens))
    print("exact solution:", verifier.target)

    enc_x, src_len = encode_tokens(
        env,
        src_tokens,
        device
    )

    with torch.no_grad():
        enc = encoder(
            "fwd",
            x=enc_x,
            lengths=src_len,
            causal=False
        ).transpose(0, 1)

    fout = open(
        args.save,
        "w",
        encoding="utf-8"
    )

    best = None
    successes = 0

    total = args.samples

    print(
        f"\nFrozen search: "
        f"{total} samples, "
        f"T={args.temperature}"
    )

    for start in range(
        0,
        total,
        args.batch_size
    ):

        n = min(
            args.batch_size,
            total - start
        )

        with torch.no_grad():

            generated, gen_len = decoder.generate(
                enc.expand(n, -1, -1).contiguous(),
                src_len.expand(n).contiguous(),
                max_len=args.max_len,
                sample_temperature=args.temperature,
            )

        candidates, infos = verifier.evaluate_generated(
            generated,
            gen_len
        )

        for i, ((ids, words), info) in enumerate(
            zip(candidates, infos)
        ):

            if info.exact:
                successes += 1

            row = {
                "global_idx": start + i,
                "reward": info.reward,
                "ode_residual": info.ode_residual,
                "reference_error": info.reference_error,
                "anchor_error": info.anchor_error,
                "exact": info.exact,
                "expression": str(info.expression),
                "fitted_expression": str(info.fitted_expression),
                "tokens": " ".join(words),
            }

            fout.write(
                json.dumps(row) + "\n"
            )

            if (
                best is None
                or info.reward > best["reward"]
            ):
                best = row

        fout.flush()

        print(
            f"{start+n:5d}/{total} | "
            f"best={best['reward']:.3f} | "
            f"exact={successes}"
        )

    fout.close()

    print("\n==============================")
    print("FINAL FROZEN SEARCH RESULT")
    print("==============================")

    print("exact solutions:", successes)
    print("best reward:", best["reward"])
    print("best expression:", best["expression"])
    print("fitted:", best["fitted_expression"])
    print("ODE residual:", best["ode_residual"])
    print("reference error:", best["reference_error"])
    print("anchor error:", best["anchor_error"])

    print("\nNo gradient updates were performed.")


if __name__ == "__main__":
    main()
