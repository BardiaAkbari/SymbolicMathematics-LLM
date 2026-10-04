#!/usr/bin/env python3
"""One TTRL step timing: sample 128 → reward → REINFORCE update."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import sympy as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import (
    encode_tokens,
    equation_to_tokens,
    load_pretrained,
    sequence_logprobs,
)
from ttrl_lane_emden.ivp_n5 import LaneEmdenN5Verifier
from ttrl_lane_emden.run_ttrl_n5 import configure_scope, encode_problem, make_optimizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rollouts", type=int, default=128)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--scope", choices=["last_layer", "enc_last", "enc_dec"], default="last_layer")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    env, encoder, decoder, _, _ = load_pretrained(args.checkpoint, device)

    x = env.local_dict["x"]
    f = env.local_dict["f"]
    y = f(x)
    equation = x * sp.diff(y, x, 2) + 2 * sp.diff(y, x) + x * y**5
    src_tokens = equation_to_tokens(env, equation)
    src, src_len = encode_tokens(env, src_tokens, device)

    verifier = LaneEmdenN5Verifier(env, exact_check=False)  # dense path for speed
    trainable, _, _ = configure_scope(encoder, decoder, args.scope)
    optimizer = make_optimizer(encoder, decoder, args.scope, lr=3e-6, encoder_lr=3e-7)

    print("device=", device, "scope=", args.scope, "rollouts=", args.rollouts)
    print("trainable_params=", sum(p.numel() for p in trainable))

    # ----- sample -----
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    encoder.eval(); decoder.eval()
    enc = encode_problem(encoder, src, src_len, grad=False)
    with torch.no_grad():
        generated, gen_len = decoder.generate(
            enc.expand(args.rollouts, -1, -1).contiguous(),
            src_len.expand(args.rollouts).contiguous(),
            max_len=args.max_len,
            sample_temperature=args.temperature,
        )
    torch.cuda.synchronize() if device.type == "cuda" else None
    t_sample = time.perf_counter() - t0

    # ----- reward -----
    t1 = time.perf_counter()
    candidates, infos, hits, misses = verifier.evaluate_generated(generated, gen_len, cache={})
    t_reward = time.perf_counter() - t1
    rewards_np = np.asarray([z.reward for z in infos], dtype=np.float32)

    # ----- REINFORCE update -----
    t2 = time.perf_counter()
    rewards_t = torch.tensor(rewards_np, device=device)
    if rewards_t.numel() > 1 and float(rewards_t.std(unbiased=False)) > 1e-8:
        adv = (rewards_t - rewards_t.mean()) / (rewards_t.std(unbiased=False) + 1e-6)
    else:
        adv = rewards_t * 0.0
    enc_policy = encode_problem(encoder, src, src_len, grad=True)
    encoder.train(); decoder.train()
    logp = sequence_logprobs(
        decoder,
        enc_policy.expand(args.rollouts, -1, -1).contiguous(),
        src_len.expand(args.rollouts).contiguous(),
        generated,
        gen_len,
        length_normalize=True,
    )
    loss = -(adv.detach() * logp).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 0.5)
    optimizer.step()
    torch.cuda.synchronize() if device.type == "cuda" else None
    t_update = time.perf_counter() - t2

    total = t_sample + t_reward + t_update
    print(f"\n--- one-step timings (s) ---")
    print(f"  sample : {t_sample:.3f}")
    print(f"  reward : {t_reward:.3f}   (hits={hits} misses={misses})")
    print(f"  update : {t_update:.3f}")
    print(f"  TOTAL  : {total:.3f}")
    print(f"reward mean={rewards_np.mean():.3f} max={rewards_np.max():.3f} "
          f"finite={sum(z.finite for z in infos)}/{len(infos)}")
    print("ONE-STEP BENCH DONE")


if __name__ == "__main__":
    main()
