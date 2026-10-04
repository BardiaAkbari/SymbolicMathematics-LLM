#!/usr/bin/env python3
"""Reference-free TTRL for Lane-Emden n=5.

Run from inside ttrl_lane_emden/.

Scopes:
  last_layer : last decoder block + output projection
  enc_last   : last encoder block + full decoder
  enc_dec    : full encoder + full decoder
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import (
    encode_tokens,
    equation_to_tokens,
    load_pretrained,
    sequence_logprobs,
    token_sequence_logprobs,
)
from ttrl_lane_emden.ivp_n5 import LaneEmdenN5Verifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--rollouts", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.8)
    p.add_argument("--temperature-end", type=float, default=1.1)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-6,
                   help="decoder learning rate")
    p.add_argument("--encoder-lr", type=float, default=3e-7,
                   help="encoder learning rate; intentionally smaller")
    p.add_argument("--scope", choices=["last_layer", "enc_last", "enc_dec"], default="last_layer")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--elite-size", type=int, default=4)
    p.add_argument("--elite-updates", type=int, default=2)
    p.add_argument("--replay-updates", type=int, default=1)
    p.add_argument("--length-penalty", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-jsonl", default="ttrl_lane_emden_n5_results.jsonl")
    p.add_argument("--save-encoder", default="")
    p.add_argument("--save-decoder", default="")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unique_params(modules):
    out, seen = [], set()
    for module in modules:
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                out.append(p)
                seen.add(id(p))
    return out


def configure_scope(encoder, decoder, scope):
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)

    if scope == "last_layer":
        enc_modules = []
        dec_modules = [
            decoder.attentions[-1], decoder.layer_norm1[-1],
            decoder.layer_norm15[-1], decoder.encoder_attn[-1],
            decoder.ffns[-1], decoder.layer_norm2[-1], decoder.proj,
        ]
    elif scope == "enc_last":
        enc_modules = [
            encoder.attentions[-1], encoder.layer_norm1[-1],
            encoder.ffns[-1], encoder.layer_norm2[-1],
        ]
        dec_modules = [decoder]
    elif scope == "enc_dec":
        enc_modules = [encoder]
        dec_modules = [decoder]
    else:
        raise ValueError(scope)

    for module in enc_modules + dec_modules:
        for p in module.parameters():
            p.requires_grad_(True)

    return unique_params(enc_modules + dec_modules), enc_modules, dec_modules


def encode_problem(encoder, src, src_len, grad=False):
    if grad:
        return encoder("fwd", x=src, lengths=src_len, causal=False).transpose(0, 1)
    with torch.no_grad():
        return encoder("fwd", x=src, lengths=src_len, causal=False).transpose(0, 1)


def add_elites(elites, candidates, infos, max_elites):
    existing = {tuple(e["ids"]) for e in elites}
    added = []
    for (ids, words), info in zip(candidates, infos):
        if not info.elite_eligible or info.expression is None:
            continue
        key = tuple(int(v) for v in ids)
        if key in existing:
            continue
        entry = {
            "ids": list(key),
            "words": list(words),
            "expression": str(info.expression),
            "reward": float(info.reward),
            "ode": float(info.ode_residual),
            "anchor": float(info.anchor_error),
            "exact": bool(info.exact),
            "length": len(key),
        }
        elites.append(entry)
        existing.add(key)
        added.append(entry)
    elites.sort(key=lambda e: (-e["reward"], e["length"]))
    del elites[max_elites:]
    keep = {tuple(e["ids"]) for e in elites}
    return [e for e in added if tuple(e["ids"]) in keep]


def make_optimizer(encoder, decoder, scope, lr, encoder_lr):
    enc_params = []
    dec_params = []
    if scope in ("enc_last", "enc_dec"):
        enc_params = [p for p in encoder.parameters() if p.requires_grad]
    dec_params = [p for p in decoder.parameters() if p.requires_grad]
    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": encoder_lr})
    if dec_params:
        groups.append({"params": dec_params, "lr": lr})
    return torch.optim.Adam(groups)


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    env, encoder, decoder, _, _ = load_pretrained(args.checkpoint, device)

    import sympy as sp
    x = env.local_dict["x"]
    f = env.local_dict["f"]
    y = f(x)
    equation = x * sp.diff(y, x, 2) + 2 * sp.diff(y, x) + x * y**5
    src_tokens = equation_to_tokens(env, equation)

    verifier = LaneEmdenN5Verifier(env, length_penalty=args.length_penalty)
    trainable, _, _ = configure_scope(encoder, decoder, args.scope)
    optimizer = make_optimizer(encoder, decoder, args.scope, args.lr, args.encoder_lr)

    print("device=", device)
    print("\n=== Lane-Emden n=5 TTRL ===")
    print("equation:", equation)
    print("input_prefix:", " ".join(src_tokens))
    print("IVP: y(0)=1, y'(0)=0")
    print("reward: ODE residual + IVP Taylor anchor + length; NO reference solution")
    print("scope:", args.scope)
    print("trainable_parameters=", sum(p.numel() for p in trainable))
    print("encoder_trainable=", any(p.requires_grad for p in encoder.parameters()))
    print("decoder_trainable=", any(p.requires_grad for p in decoder.parameters()))
    print("decoder_lr=", args.lr, "encoder_lr=", args.encoder_lr)

    src, src_len = encode_tokens(env, src_tokens, device)
    elites = []
    cache = {}
    fout = open(args.save_jsonl, "w", encoding="utf-8")

    for step in range(args.steps):
        tic = time.perf_counter()
        frac = step / max(args.steps - 1, 1)
        temperature = args.temperature + frac * (args.temperature_end - args.temperature)

        encoder.eval(); decoder.eval()
        enc_sample = encode_problem(encoder, src, src_len, grad=False)
        with torch.no_grad():
            generated, gen_len = decoder.generate(
                enc_sample.expand(args.rollouts, -1, -1).contiguous(),
                src_len.expand(args.rollouts).contiguous(),
                max_len=args.max_len,
                sample_temperature=temperature,
            )

        candidates, infos, hits, misses = verifier.evaluate_generated(generated, gen_len, cache)
        rewards_np = np.asarray([z.reward for z in infos], dtype=np.float32)
        finite_mask = np.asarray([z.finite for z in infos], dtype=bool)

        print(
            f"\n[step {step:03d}] T={temperature:.3f} "
            f"mean={rewards_np.mean():.3f} std={rewards_np.std():.3f} "
            f"max={rewards_np.max():.3f} finite={int(finite_mask.sum())}/{len(infos)} "
            f"exact={sum(z.exact for z in infos)}"
        )

        for i in np.argsort(-rewards_np)[:5]:
            z = infos[int(i)]
            print(
                f"  reward={z.reward:8.3f} ode={z.ode_residual:.3e} "
                f"anchor={z.anchor_error:.3e} exact={int(z.exact)} :: {z.expression}"
            )

        new_elites = add_elites(elites, candidates, infos, args.elite_size)
        if new_elites:
            print("  new elites:")
            for e in new_elites:
                print("   ", e["expression"])

        # REINFORCE. Invalid/non-finite expressions are strongly below finite
        # candidates, so the policy learns away from them rather than treating
        # them as the best available class.
        rewards_t = torch.tensor(rewards_np, device=device)
        if rewards_t.numel() > 1 and float(rewards_t.std(unbiased=False)) > 1e-8:
            adv = (rewards_t - rewards_t.mean()) / (rewards_t.std(unbiased=False) + 1e-6)
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
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            print(f"  policy_loss={loss.item():.6f}")

        if elites:
            enc_replay = encode_problem(
                encoder, src, src_len, grad=any(p.requires_grad for p in encoder.parameters())
            )
            encoder.train(); decoder.train()
            for _ in range(args.elite_updates if new_elites else args.replay_updates):
                lp = token_sequence_logprobs(
                    env, decoder, enc_replay, src_len,
                    [e["ids"] for e in elites], device,
                    length_normalize=False,
                )
                loss = -lp.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                print(f"  replay_loss={loss.item():.4f}")

        encoder.eval(); decoder.eval()
        enc_g = encode_problem(encoder, src, src_len, grad=False)
        with torch.no_grad():
            g, gl = decoder.generate(enc_g, src_len, max_len=args.max_len, sample_temperature=None)
        _, gi, _, _ = verifier.evaluate_generated(g, gl, cache)
        greedy = gi[0]
        print(
            f"  greedy reward={greedy.reward:.3f} ode={greedy.ode_residual:.3e} "
            f"anchor={greedy.anchor_error:.3e} exact={int(greedy.exact)}"
        )

        fout.write(json.dumps({
            "step": step,
            "temperature": temperature,
            "reward_mean": float(rewards_np.mean()),
            "reward_std": float(rewards_np.std()),
            "reward_max": float(rewards_np.max()),
            "finite_count": int(finite_mask.sum()),
            "exact_count": int(sum(z.exact for z in infos)),
            "best_expression": str(infos[int(np.argmax(rewards_np))].expression),
            "greedy_expression": str(greedy.expression),
            "greedy_reward": float(greedy.reward),
            "greedy_exact": bool(greedy.exact),
            "elite_buffer_size": len(elites),
            "cache_hits": hits,
            "cache_misses": misses,
            "time_s": time.perf_counter() - tic,
        }) + "\n")
        fout.flush()

    fout.close()
    if args.save_decoder:
        torch.save(decoder.state_dict(), args.save_decoder)
    if args.save_encoder:
        torch.save(encoder.state_dict(), args.save_encoder)


if __name__ == "__main__":
    main()
