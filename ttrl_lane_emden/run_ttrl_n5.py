#!/usr/bin/env python3

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
    configure_trainable,
    encode_tokens,
    equation_to_tokens,
    load_pretrained,
    sequence_logprobs,
    token_sequence_logprobs,
)

from ttrl_lane_emden.ivp_n5 import LaneEmdenN5Verifier


def parse_args():
    p = argparse.ArgumentParser(
        description="TTRL for Lane-Emden n=5"
    )

    p.add_argument("--checkpoint", required=True)

    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--rollouts", type=int, default=128)

    # Exploration
    p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--temperature-end", type=float, default=1.0)

    p.add_argument("--max-len", type=int, default=128)

    # Optimization
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument(
        "--scope",
        choices=["proj", "last_layer", "decoder"],
        default="last_layer",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Reward / replay
    p.add_argument("--elite-size", type=int, default=4)
    p.add_argument("--elite-updates", type=int, default=5)
    p.add_argument("--replay-updates", type=int, default=1)
    p.add_argument("--elite-weight", type=float, default=1.0)

    # Small simplicity pressure
    p.add_argument(
        "--length-penalty",
        type=float,
        default=0.01,
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")

    p.add_argument(
        "--save-jsonl",
        default="ttrl_lane_emden_n5_results.jsonl",
    )

    p.add_argument("--save-decoder", default="")

    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def elite_ids(elites):
    return [e["ids"] for e in elites]


def print_top(candidates, infos, k=5):
    order = sorted(
        range(len(infos)),
        key=lambda i: infos[i].reward,
        reverse=True,
    )

    for i in order[:k]:
        ids, words = candidates[i]
        z = infos[i]

        print(
            f"  {i:3d} "
            f"reward={z.reward:8.3f} "
            f"ode={z.ode_residual:.3e} "
            f"ref={z.reference_error:.3e} "
            f"anchor={z.anchor_error:.3e} "
            f"exact={int(z.exact)}"
        )
        print(f"      {z.expression}")


def add_elites(
    elites,
    candidates,
    infos,
    max_elites,
    length_penalty,
):
    """
    Keep the best approximate symbolic trajectories.

    We do NOT require an exact solution.
    This is important because the base model cannot currently
    discover the exact n=5 solution.
    """

    existing = {
        tuple(e["ids"])
        for e in elites
    }

    new = []

    for candidate, info in zip(candidates, infos):

        if info.expression is None:
            continue

        ids, words = candidate
        key = tuple(int(x) for x in ids)

        if key in existing:
            continue

        # Add a mild simplicity preference.
        adjusted_reward = (
            float(info.reward)
            - length_penalty * len(ids)
        )

        elites.append(
            {
                "ids": list(key),
                "words": list(words),
                "expression": str(info.expression),
                "reward": adjusted_reward,
                "raw_reward": float(info.reward),
                "exact": bool(info.exact),
                "ode": float(info.ode_residual),
                "reference": float(info.reference_error),
                "anchor": float(info.anchor_error),
                "length": len(ids),
            }
        )

        new.append(elites[-1])
        existing.add(key)

    elites.sort(
        key=lambda x: (
            -x["reward"],
            x["length"],
        )
    )

    del elites[max_elites:]

    kept = {
        tuple(e["ids"])
        for e in elites
    }

    return [
        e for e in new
        if tuple(e["ids"]) in kept
    ]


def replay(
    env,
    decoder,
    enc,
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

    decoder.train()

    for _ in range(updates):

        logp = token_sequence_logprobs(
            env,
            decoder,
            enc.detach(),
            src_len,
            elite_ids(elites),
            device,
            length_normalize=False,
        )

        loss = -weight * logp.mean()

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable,
            grad_clip,
        )

        optimizer.step()

        losses.append(float(loss.item()))

    decoder.eval()

    return {
        "updates": updates,
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }


def main():

    args = parse_args()

    seed_all(args.seed)

    device = torch.device(
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else "cuda"
    )

    print("device =", device)

    env, encoder, decoder, params, _ = load_pretrained(
        args.checkpoint,
        device,
    )

    # ---------------------------------------------------------
    # Lane-Emden n=5
    #
    # x y'' + 2 y' + x y^5 = 0
    # y(0)=1, y'(0)=0
    # ---------------------------------------------------------

    import sympy as sp

    x = env.local_dict["x"]
    f = env.local_dict["f"]

    y = f(x)

    equation = (
        x * sp.diff(y, x, 2)
        + 2 * sp.diff(y, x)
        + x * y**5
    )

    verifier = LaneEmdenN5Verifier(env)

    src_tokens = equation_to_tokens(
        env,
        equation,
    )

    print("\n================================")
    print("Lane-Emden n=5 TTRL")
    print("================================")

    print("equation:", equation)
    print("input prefix:", " ".join(src_tokens))
    print("known exact solution:", verifier.target)

    # Freeze encoder.
    for p in encoder.parameters():
        p.requires_grad_(False)

    trainable = configure_trainable(
        decoder,
        args.scope,
    )

    optimizer = torch.optim.Adam(
        trainable,
        lr=args.lr,
    )

    print(
        f"trainable parameters="
        f"{sum(p.numel() for p in trainable):,}"
    )

    # ---------------------------------------------------------
    # Encode problem once
    # ---------------------------------------------------------

    src, src_len = encode_tokens(
        env,
        src_tokens,
        device,
    )

    with torch.no_grad():
        enc = encoder(
            "fwd",
            x=src,
            lengths=src_len,
            causal=False,
        ).transpose(0, 1)

    elites = []

    fout = open(
        args.save_jsonl,
        "w",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # TTRL
    # ---------------------------------------------------------

    for step in range(args.steps):

        t0 = time.perf_counter()

        # Linear temperature schedule:
        # high exploration -> lower exploration.
        alpha = step / max(args.steps - 1, 1)

        temperature = (
            args.temperature
            + alpha
            * (args.temperature_end - args.temperature)
        )

        decoder.eval()

        # -----------------------------------------------------
        # Sample rollouts
        # -----------------------------------------------------

        with torch.no_grad():

            enc_batch = enc.expand(
                args.rollouts,
                -1,
                -1,
            ).contiguous()

            len_batch = src_len.expand(
                args.rollouts
            ).contiguous()

            generated, gen_len = decoder.generate(
                enc_batch,
                len_batch,
                max_len=args.max_len,
                sample_temperature=temperature,
            )

        # -----------------------------------------------------
        # Verifier
        # -----------------------------------------------------

        candidates, infos = verifier.evaluate_generated(
            generated,
            gen_len,
        )

        rewards = np.asarray(
            [z.reward for z in infos],
            dtype=np.float32,
        )

        exact_count = sum(
            z.exact
            for z in infos
        )

        best_idx = int(
            np.argmax(rewards)
        )

        best = infos[best_idx]

        print(
            f"\n[step {step:03d}] "
            f"T={temperature:.3f} "
            f"mean={rewards.mean():.3f} "
            f"max={rewards.max():.3f} "
            f"exact={exact_count}"
        )

        print_top(
            candidates,
            infos,
            k=5,
        )

        # -----------------------------------------------------
        # Save best approximate trajectories
        # -----------------------------------------------------

        new_elites = add_elites(
            elites,
            candidates,
            infos,
            args.elite_size,
            args.length_penalty,
        )

        if new_elites:
            print(
                f"  + new elites: {len(new_elites)} "
                f"(buffer={len(elites)})"
            )

            for e in new_elites:
                print(
                    "    ",
                    e["expression"],
                )

        # -----------------------------------------------------
        # TTRL policy-gradient update
        # -----------------------------------------------------

        rewards_t = torch.tensor(
            rewards,
            device=device,
        )

        rl_loss = None

        if rewards_t.std(unbiased=False) > 1e-8:

            advantages = (
                rewards_t - rewards_t.mean()
            ) / (
                rewards_t.std(unbiased=False)
                + 1e-6
            )

            decoder.train()

            logp = sequence_logprobs(
                decoder,
                enc.detach().expand(
                    args.rollouts,
                    -1,
                    -1,
                ).contiguous(),
                src_len.expand(
                    args.rollouts
                ).contiguous(),
                generated,
                gen_len,
                length_normalize=True,
            )

            loss = -(
                advantages.detach()
                * logp
            ).mean()

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable,
                args.grad_clip,
            )

            optimizer.step()

            rl_loss = float(loss.item())

            decoder.eval()

            print(
                f"  policy_loss={rl_loss:.6f}"
            )

        # -----------------------------------------------------
        # Replay best approximate solutions
        # -----------------------------------------------------

        replay_updates = (
            args.elite_updates
            if new_elites
            else args.replay_updates
        )

        replay_stats = replay(
            env,
            decoder,
            enc,
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
                f"  replay "
                f"{replay_stats['loss_first']:.3f}"
                f" -> "
                f"{replay_stats['loss_last']:.3f}"
            )

        # -----------------------------------------------------
        # Greedy diagnostic
        # -----------------------------------------------------

        decoder.eval()

        with torch.no_grad():

            greedy_gen, greedy_len = decoder.generate(
                enc,
                src_len,
                max_len=args.max_len,
                sample_temperature=None,
            )

        gc, gi = verifier.evaluate_generated(
            greedy_gen,
            greedy_len,
        )

        gz = gi[0]

        print(
            f"  greedy: "
            f"reward={gz.reward:.3f} "
            f"ode={gz.ode_residual:.3e} "
            f"ref={gz.reference_error:.3e} "
            f"exact={int(gz.exact)}"
        )

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        row = {
            "step": step,
            "temperature": temperature,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_max": float(rewards.max()),
            "exact_count": int(exact_count),
            "best_expression": str(best.expression),
            "best_ode": float(best.ode_residual),
            "best_reference": float(best.reference_error),
            "best_anchor": float(best.anchor_error),
            "greedy_expression": str(gz.expression),
            "greedy_reward": float(gz.reward),
            "greedy_exact": bool(gz.exact),
            "rl_loss": rl_loss,
            "replay": replay_stats,
            "elite_buffer_size": len(elites),
            "new_elites": [
                {
                    k: v
                    for k, v in e.items()
                    if k != "ids"
                }
                for e in new_elites
            ],
            "time": time.perf_counter() - t0,
        }

        fout.write(
            json.dumps(row) + "\n"
        )

        fout.flush()

    fout.close()

    # ---------------------------------------------------------
    # Final greedy
    # ---------------------------------------------------------

    decoder.eval()

    with torch.no_grad():

        final_gen, final_len = decoder.generate(
            enc,
            src_len,
            max_len=args.max_len,
            sample_temperature=None,
        )

    fc, fi = verifier.evaluate_generated(
        final_gen,
        final_len,
    )

    final = fi[0]

    print("\n================================")
    print("FINAL")
    print("================================")

    print("expression:", final.expression)
    print("reward:", final.reward)
    print("ODE residual:", final.ode_residual)
    print("reference error:", final.reference_error)
    print("anchor error:", final.anchor_error)
    print("exact:", final.exact)

    if args.save_decoder:
        torch.save(
            decoder.state_dict(),
            args.save_decoder,
        )
        print(
            "saved:",
            args.save_decoder,
        )


if __name__ == "__main__":
    main()
