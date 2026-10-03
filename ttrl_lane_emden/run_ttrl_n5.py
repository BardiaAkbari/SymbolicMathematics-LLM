#!/usr/bin/env python3
"""Reference-free TTRL for Lane-Emden n=5.

Run from inside ttrl_lane_emden/ in Colab:
    python run_ttrl_n5.py --checkpoint ode2.pth --scope last_layer

Scopes:
    proj       : decoder output projection only
    last_layer : last decoder block + output projection
    decoder    : full decoder
    enc_last   : last encoder block + full decoder
    enc_dec    : full encoder + full decoder

The n=5 verifier uses only the cleared ODE and IVP constraints. The known
closed-form solution is NOT used by the TTRL reward.
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
    p = argparse.ArgumentParser(description="Reference-free TTRL for Lane-Emden n=5")
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
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--elite-size", type=int, default=4)
    p.add_argument("--elite-updates", type=int, default=5)
    p.add_argument("--replay-updates", type=int, default=1)
    p.add_argument("--elite-weight", type=float, default=1.0)
    p.add_argument("--length-penalty", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-jsonl", default="ttrl_lane_emden_n5_results.jsonl")
    p.add_argument("--save-decoder", default="")
    p.add_argument("--save-encoder", default="")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unique_params(modules):
    params = []
    seen = set()
    for module in modules:
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
    return params


def configure_scope(encoder, decoder, scope):
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)

    if scope == "proj":
        modules_d = [decoder.proj]
        modules_e = []

    elif scope == "last_layer":
        modules_d = [
            decoder.attentions[-1],
            decoder.layer_norm1[-1],
            decoder.layer_norm15[-1],
            decoder.encoder_attn[-1],
            decoder.ffns[-1],
            decoder.layer_norm2[-1],
            decoder.proj,
        ]
        modules_e = []

    elif scope == "decoder":
        modules_d = [decoder]
        modules_e = []

    elif scope == "enc_last":
        # This is intentionally: LAST ENCODER BLOCK + FULL DECODER.
        # It isolates whether a small encoder adaptation helps while giving the
        # decoder enough capacity to solve the OOD symbolic-generation problem.
        modules_d = [decoder]
        modules_e = [
            encoder.attentions[-1],
            encoder.layer_norm1[-1],
            encoder.ffns[-1],
            encoder.layer_norm2[-1],
        ]

    elif scope == "enc_dec":
        modules_d = [decoder]
        modules_e = [encoder]

    else:
        raise ValueError(f"unknown scope: {scope}")

    for module in modules_e + modules_d:
        for p in module.parameters():
            p.requires_grad_(True)

    return unique_params(modules_e + modules_d), modules_e, modules_d


def encode_problem(encoder, src, src_len, grad_enabled):
    if grad_enabled:
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
        key = tuple(int(i) for i in ids)
        if key in existing:
            continue
        entry = {
            "ids": list(key),
            "words": list(words),
            "expression": str(info.expression),
            "fitted_expression": str(info.fitted_expression),
            "fitted_coefficients": dict(info.fitted_coefficients),
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


def replay(encoder, decoder, env, enc_src, src_len, elites, optimizer, trainable, device, updates, weight, grad_clip, encoder_trainable):
    if not elites or updates <= 0:
        return None

    losses = []
    for _ in range(updates):
        # Recompute encoder states when encoder is trainable. This is essential:
        # otherwise the replay loss would backprop through a stale/frozen encoding.
        enc_for_replay = enc_src
        if encoder_trainable:
            enc_for_replay = encode_problem(encoder, enc_src[0], src_len, True)

        lp = token_sequence_logprobs(
            env,
            decoder,
            enc_for_replay,
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

    decoder.eval()
    encoder.eval()
    return {
        "updates": int(updates),
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("device=", device)

    env, encoder, decoder, params, _ = load_pretrained(args.checkpoint, device)

    import sympy as sp
    x = env.local_dict["x"]
    f = env.local_dict["f"]
    y = f(x)

    # Cleared form used by both the model and the verifier.
    equation = x * sp.diff(y, x, 2) + 2 * sp.diff(y, x) + x * y**5
    src_tokens = equation_to_tokens(env, equation)

    verifier = LaneEmdenN5Verifier(
        env,
        length_penalty=args.length_penalty,
    )

    trainable, _, _ = configure_scope(encoder, decoder, args.scope)
    encoder_trainable = any(p.requires_grad for p in encoder.parameters())

    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    print("\n=== Lane-Emden n=5 TTRL ===")
    print("equation:", equation)
    print("input_prefix:", " ".join(src_tokens))
    print("IVP: y(0)=1, y'(0)=0")
    print("reward: ODE residual + IVP anchor + length penalty; NO reference solution")
    print("scope:", args.scope)
    print("trainable_parameters=", sum(p.numel() for p in trainable))
    print("encoder_trainable=", encoder_trainable)
    print("decoder_trainable=", any(p.requires_grad for p in decoder.parameters()))

    src, src_len = encode_tokens(env, src_tokens, device)
    enc_base = encode_problem(encoder, src, src_len, grad_enabled=False)

    elites = []
    verifier_cache = {}
    fout = open(args.save_jsonl, "w", encoding="utf-8")

    for step in range(args.steps):
        t0 = time.perf_counter()
        alpha = step / max(args.steps - 1, 1)
        temperature = args.temperature + alpha * (args.temperature_end - args.temperature)

        encoder.eval()
        decoder.eval()

        # Exploration always samples from the current model without building a
        # gradient graph; sampled token sequences are treated as actions.
        enc_sample = encode_problem(
            encoder, src, src_len, grad_enabled=False
        )

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

        rewards_np = np.asarray([z.reward for z in infos], dtype=np.float32)
        best_i = int(np.argmax(rewards_np))
        best = infos[best_i]

        print(
            f"\n[step {step:03d}] T={temperature:.3f} "
            f"mean={rewards_np.mean():.3f} "
            f"std={rewards_np.std():.3f} "
            f"max={rewards_np.max():.3f} "
            f"exact={sum(z.exact for z in infos)}"
        )

        for i in np.argsort(-rewards_np)[:5]:
            z = infos[int(i)]
            print(
                f"  reward={z.reward:8.3f} "
                f"ode={z.ode_residual:.3e} "
                f"anchor={z.anchor_error:.3e} "
                f"exact={int(z.exact)} :: {z.expression}"
            )
            if z.fitted_expression is not None and z.fitted_expression != z.expression:
                print(f"      fitted={z.fitted_expression} coeffs={z.fitted_coefficients}")

        new_elites = add_elites(elites, candidates, infos, args.elite_size)
        if new_elites:
            print("  new elites:")
            for e in new_elites:
                print("   ", e["expression"])

        # TTRL policy-gradient update.
        rl_loss = None
        rewards_t = torch.tensor(rewards_np, device=device)
        if float(rewards_t.std(unbiased=False).item()) > 1e-8:
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std(unbiased=False) + 1e-6)

            # Recompute encoder states WITH gradients when encoder is trainable.
            enc_policy = encode_problem(
                encoder, src, src_len,
                grad_enabled=encoder_trainable,
            )

            decoder.train()
            if encoder_trainable:
                encoder.train()

            logp = sequence_logprobs(
                decoder,
                enc_policy.expand(args.rollouts, -1, -1).contiguous(),
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
            decoder.eval()
            encoder.eval()
            print(f"  policy_loss={rl_loss:.6f}")

        # Replay elites. Recompute the encoder graph when needed.
        replay_updates = args.elite_updates if new_elites else args.replay_updates
        replay_stats = None
        if elites and replay_updates > 0:
            for _ in range(replay_updates):
                enc_replay = encode_problem(
                    encoder, src, src_len,
                    grad_enabled=encoder_trainable,
                )
                decoder.train()
                if encoder_trainable:
                    encoder.train()

                lp = token_sequence_logprobs(
                    env,
                    decoder,
                    enc_replay,
                    src_len,
                    elite_ids(elites),
                    device,
                    length_normalize=False,
                )
                loss = -args.elite_weight * lp.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()

            replay_stats = {"updates": int(replay_updates), "loss_last": float(loss.item())}
            decoder.eval()
            encoder.eval()
            print(f"  replay updates={replay_stats['updates']} loss_last={replay_stats['loss_last']:.3f}")

        # Greedy diagnostic after adaptation.
        enc_greedy = encode_problem(encoder, src, src_len, grad_enabled=False)
        with torch.no_grad():
            g, gl = decoder.generate(
                enc_greedy,
                src_len,
                max_len=args.max_len,
                sample_temperature=None,
            )
        _, gi, _, _ = verifier.evaluate_generated(g, gl, verifier_cache)
        greedy = gi[0]

        print(
            f"  greedy reward={greedy.reward:.3f} "
            f"ode={greedy.ode_residual:.3e} "
            f"anchor={greedy.anchor_error:.3e} "
            f"exact={int(greedy.exact)}"
        )

        fout.write(json.dumps({
            "step": step,
            "temperature": temperature,
            "reward_mean": float(rewards_np.mean()),
            "reward_std": float(rewards_np.std()),
            "reward_max": float(rewards_np.max()),
            "exact_count": int(sum(z.exact for z in infos)),
            "best_expression": str(best.expression),
            "best_fitted_expression": str(best.fitted_expression),
            "best_fitted_coefficients": dict(best.fitted_coefficients),
            "best_ode": float(best.ode_residual),
            "best_anchor": float(best.anchor_error),
            "greedy_expression": str(greedy.expression),
            "greedy_fitted_expression": str(greedy.fitted_expression),
            "greedy_reward": float(greedy.reward),
            "greedy_exact": bool(greedy.exact),
            "rl_loss": rl_loss,
            "replay": replay_stats,
            "elite_buffer_size": len(elites),
            "new_elites": [{k: v for k, v in e.items() if k != "ids"} for e in new_elites],
            "cache_hits": hits,
            "cache_misses": misses,
            "time_s": time.perf_counter() - t0,
        }) + "\n")
        fout.flush()

    fout.close()

    # Save adapted weights when requested.
    if args.save_decoder:
        torch.save(decoder.state_dict(), args.save_decoder)
        print("saved_decoder=", args.save_decoder)
    if args.save_encoder:
        torch.save(encoder.state_dict(), args.save_encoder)
        print("saved_encoder=", args.save_encoder)


if __name__ == "__main__":
    main()
