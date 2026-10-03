#!/usr/bin/env python3
"""TTRL for Lane-Emden n=5 with optional encoder adaptation.

Problem:
    x*y'' + 2*y' + x*y^5 = 0
    y(0) = 1, y'(0) = 0

The verifier uses only the ODE residual and the IVP conditions.
It does not use the known closed-form solution as a reward target.

Scopes:
    proj              : decoder output projection only
    last_layer        : decoder last block + projection
    decoder           : full decoder
    enc_last          : encoder last block + decoder last block + projection
    enc_dec            : full encoder + full decoder
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Iterable

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
    p = argparse.ArgumentParser(description="TTRL for Lane-Emden n=5")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--rollouts", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--temperature-end", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument(
        "--scope",
        choices=["proj", "last_layer", "decoder", "enc_last", "enc_dec"],
        default="last_layer",
        help="Parameter adaptation scope; see module docstring.",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--elite-size", type=int, default=4)
    p.add_argument("--elite-updates", type=int, default=5)
    p.add_argument("--replay-updates", type=int, default=1)
    p.add_argument("--elite-weight", type=float, default=1.0)
    p.add_argument("--length-penalty", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-jsonl", default="n5_ttrl_results.jsonl")
    p.add_argument("--save-decoder", default="")
    p.add_argument("--save-encoder", default="")
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_requires_grad(module, enabled: bool):
    for p in module.parameters():
        p.requires_grad_(enabled)


def set_module_trainable(module, enabled: bool):
    for p in module.parameters():
        p.requires_grad_(enabled)


def configure_scope(encoder, decoder, scope: str):
    """Configure trainable parameters and return a deduplicated parameter list."""
    set_requires_grad(encoder, False)
    set_requires_grad(decoder, False)

    def enable(module):
        set_module_trainable(module, True)

    decoder_last_modules = [
        decoder.attentions[-1],
        decoder.layer_norm1[-1],
        decoder.layer_norm15[-1],
        decoder.encoder_attn[-1],
        decoder.ffns[-1],
        decoder.layer_norm2[-1],
        decoder.proj,
    ]

    encoder_last_modules = [
        encoder.attentions[-1],
        encoder.layer_norm1[-1],
        encoder.ffns[-1],
        encoder.layer_norm2[-1],
    ]

    if scope == "proj":
        enable(decoder.proj)
    elif scope == "last_layer":
        for m in decoder_last_modules:
            enable(m)
    elif scope == "decoder":
        enable(decoder)
    elif scope == "enc_last":
        for m in encoder_last_modules:
            enable(m)
        for m in decoder_last_modules:
            enable(m)
    elif scope == "enc_dec":
        enable(encoder)
        enable(decoder)
    else:
        raise ValueError(f"Unknown scope: {scope}")

    params = []
    seen = set()
    for module in (encoder, decoder):
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
    return params


def train_mode(encoder, decoder, encoder_trainable: bool):
    encoder.train(encoder_trainable)
    decoder.train(True)


def eval_mode(encoder, decoder):
    encoder.eval()
    decoder.eval()


def encode_current(encoder, src, src_len, grad: bool):
    if grad:
        return encoder(
            "fwd", x=src, lengths=src_len, causal=False
        ).transpose(0, 1)
    with torch.no_grad():
        return encoder(
            "fwd", x=src, lengths=src_len, causal=False
        ).transpose(0, 1)


def elite_ids(elites):
    return [e["ids"] for e in elites]


def add_elites(elites, candidates, infos, max_elites):
    existing = {tuple(e["ids"]) for e in elites}
    new = []

    for (ids, words), info in zip(candidates, infos):
        if info.expression is None or not np.isfinite(info.reward):
            continue

        key = tuple(int(x) for x in ids)
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
        new.append(entry)

    elites.sort(key=lambda e: (-e["reward"], e["length"]))
    del elites[max_elites:]
    keep = {tuple(e["ids"]) for e in elites}
    return [e for e in new if tuple(e["ids"]) in keep]


def replay(
    env,
    encoder,
    decoder,
    src,
    src_len,
    elites,
    optimizer,
    trainable,
    device,
    updates,
    weight,
    grad_clip,
):
    if not elites or updates <= 0:
        return None

    losses = []
    encoder_trainable = any(p.requires_grad for p in encoder.parameters())

    for _ in range(updates):
        train_mode(encoder, decoder, encoder_trainable)
        enc = encode_current(encoder, src, src_len, grad=True)

        lp = token_sequence_logprobs(
            env,
            decoder,
            enc,
            src_len,
            elite_ids(elites),
            device,
            length_normalize=False,
        )
        loss = -float(weight) * lp.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))

    eval_mode(encoder, decoder)
    return {
        "updates": int(updates),
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print("device=", device)

    env, encoder, decoder, params, _ = load_pretrained(args.checkpoint, device)

    import sympy as sp

    x = env.local_dict["x"]
    f = env.local_dict["f"]
    y = f(x)

    # Cleared Lane-Emden equation used as model input.
    equation = x * sp.diff(y, x, 2) + 2 * sp.diff(y, x) + x * y**5
    src_tokens = equation_to_tokens(env, equation)

    verifier = LaneEmdenN5Verifier(
        env,
        length_penalty=args.length_penalty,
    )

    print("\n=== Lane-Emden n=5 TTRL ===")
    print("equation:", equation)
    print("input_prefix:", " ".join(src_tokens))
    print("IVP: y(0)=1, y'(0)=0")
    print("reward: ODE residual + IVP anchor + length penalty; NO reference solution")
    print("scope:", args.scope)

    trainable = configure_scope(encoder, decoder, args.scope)
    encoder_is_trainable = any(p.requires_grad for p in encoder.parameters())
    decoder_is_trainable = any(p.requires_grad for p in decoder.parameters())

    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    print("trainable_parameters=", sum(p.numel() for p in trainable))
    print("encoder_trainable=", encoder_is_trainable)
    print("decoder_trainable=", decoder_is_trainable)

    src, src_len = encode_tokens(env, src_tokens, device)

    elites = []
    verifier_cache = {}
    fout = open(args.save_jsonl, "w", encoding="utf-8")

    for step in range(args.steps):
        t0 = time.perf_counter()
        alpha = step / max(args.steps - 1, 1)
        temperature = args.temperature + alpha * (
            args.temperature_end - args.temperature
        )

        # ------------------------------------------------------
        # Sample using the CURRENT encoder/decoder.
        # ------------------------------------------------------
        eval_mode(encoder, decoder)
        enc_sample = encode_current(encoder, src, src_len, grad=False)

        with torch.no_grad():
            generated, gen_len = decoder.generate(
                enc_sample.expand(args.rollouts, -1, -1).contiguous(),
                src_len.expand(args.rollouts).contiguous(),
                max_len=args.max_len,
                sample_temperature=temperature,
            )

        candidates, infos, hits, misses = verifier.evaluate_generated(
            generated, gen_len, verifier_cache
        )

        rewards_np = np.asarray(
            [z.reward for z in infos], dtype=np.float32
        )
        best_i = int(np.argmax(rewards_np))
        best = infos[best_i]

        print(
            f"\n[step {step:03d}] T={temperature:.3f} "
            f"mean={rewards_np.mean():.3f} "
            f"std={rewards_np.std():.3f} "
            f"max={rewards_np.max():.3f} "
            f"exact={sum(z.exact for z in infos)}"
        )

        order = np.argsort(-rewards_np)[:5]
        for i in order:
            z = infos[int(i)]
            print(
                f"  reward={z.reward:8.3f} "
                f"ode={z.ode_residual:.3e} "
                f"anchor={z.anchor_error:.3e} "
                f"exact={int(z.exact)} :: {z.expression}"
            )

        new_elites = add_elites(
            elites, candidates, infos, args.elite_size
        )

        # ------------------------------------------------------
        # Group-relative TTRL update.
        # Recompute encoder under autograd when encoder is trainable.
        # ------------------------------------------------------
        rewards_t = torch.tensor(rewards_np, device=device)
        rl_loss = None

        if float(rewards_t.std(unbiased=False).item()) > 1e-8:
            advantages = (
                rewards_t - rewards_t.mean()
            ) / (rewards_t.std(unbiased=False) + 1e-6)

            train_mode(encoder, decoder, encoder_is_trainable)
            enc_loss = encode_current(
                encoder, src, src_len, grad=True
            )

            logp = sequence_logprobs(
                decoder,
                enc_loss.expand(args.rollouts, -1, -1).contiguous(),
                src_len.expand(args.rollouts).contiguous(),
                generated,
                gen_len,
                length_normalize=True,
            )

            loss = -(advantages.detach() * logp).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()

            rl_loss = float(loss.item())
            eval_mode(encoder, decoder)
            print("  policy_loss=", f"{rl_loss:.6f}")

        # ------------------------------------------------------
        # Replay best approximate trajectories.
        # ------------------------------------------------------
        replay_updates = (
            args.elite_updates if new_elites else args.replay_updates
        )

        replay_stats = replay(
            env,
            encoder,
            decoder,
            src,
            src_len,
            elites,
            optimizer,
            trainable,
            device,
            replay_updates,
            args.elite_weight,
            args.grad_clip,
        )

        if replay_stats:
            print(
                f"  replay {replay_stats['loss_first']:.3f}"
                f" -> {replay_stats['loss_last']:.3f}"
            )

        # ------------------------------------------------------
        # Greedy diagnostic using the adapted model.
        # ------------------------------------------------------
        eval_mode(encoder, decoder)
        enc_greedy = encode_current(encoder, src, src_len, grad=False)

        with torch.no_grad():
            greedy_gen, greedy_len = decoder.generate(
                enc_greedy,
                src_len,
                max_len=args.max_len,
                sample_temperature=None,
            )

        _, greedy_infos, _, _ = verifier.evaluate_generated(
            greedy_gen,
            greedy_len,
            verifier_cache,
        )
        greedy = greedy_infos[0]

        print(
            f"  greedy reward={greedy.reward:.3f} "
            f"ode={greedy.ode_residual:.3e} "
            f"anchor={greedy.anchor_error:.3e} "
            f"exact={int(greedy.exact)}"
        )

        fout.write(json.dumps({
            "step": step,
            "temperature": temperature,
            "scope": args.scope,
            "reward_mean": float(rewards_np.mean()),
            "reward_std": float(rewards_np.std()),
            "reward_max": float(rewards_np.max()),
            "exact_count": int(sum(z.exact for z in infos)),
            "best_expression": str(best.expression),
            "best_ode": float(best.ode_residual),
            "best_anchor": float(best.anchor_error),
            "greedy_expression": str(greedy.expression),
            "greedy_reward": float(greedy.reward),
            "greedy_ode": float(greedy.ode_residual),
            "greedy_anchor": float(greedy.anchor_error),
            "greedy_exact": bool(greedy.exact),
            "rl_loss": rl_loss,
            "replay": replay_stats,
            "elite_buffer_size": len(elites),
            "new_elites": [
                {k: v for k, v in e.items() if k != "ids"}
                for e in new_elites
            ],
            "cache_hits": hits,
            "cache_misses": misses,
            "time_s": time.perf_counter() - t0,
        }) + "\n")
        fout.flush()

    fout.close()

    if args.save_decoder:
        torch.save(decoder.state_dict(), args.save_decoder)
        print("saved_decoder=", args.save_decoder)

    if args.save_encoder:
        torch.save(encoder.state_dict(), args.save_encoder)
        print("saved_encoder=", args.save_encoder)


if __name__ == "__main__":
    main()
