#!/usr/bin/env python3
"""Diagnostic: can one self-discovered verified solution be made more probable?

This deliberately isolates the gradient path from exploration.  It either accepts a
known verified token sequence (e.g. one discovered in a previous TTRL run), or samples
until it finds one.  It then performs pure exact-trajectory likelihood updates and
measures raw log P plus fresh exact-solution sampling rate.
"""
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
    evaluate_rollout_batch_cached,
    load_pretrained,
    make_problem,
    score_candidate_ids,
    token_sequence_logprobs,
)


def parse_args():
    p = argparse.ArgumentParser(description="Elite-learning diagnostic for symbolic ODE TTRL")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--problem", choices=["harmonic", "lane_emden"], default="lane_emden")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--lane-form", choices=["standard", "cleared"], default="standard")
    p.add_argument("--scope", choices=["proj", "last_layer", "decoder"], default="last_layer")
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--search-samples", type=int, default=2048)
    p.add_argument("--search-batch-size", type=int, default=64)
    p.add_argument("--elite-tokens", default="",
                   help="Space-separated prefix tokens of a previously self-discovered verified solution")
    p.add_argument("--eval-rollouts", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-jsonl", default="elite_probe_results.jsonl")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_logp(env, decoder, enc1, src_len_1, ids, device):
    decoder.eval()
    with torch.no_grad():
        lp = token_sequence_logprobs(
            env, decoder, enc1, src_len_1, [ids], device,
            length_normalize=False,
        )[0]
    return float(lp.item())


def fresh_eval(env, decoder, enc1, src_len_1, problem, device, n, temp, max_len, cache):
    decoder.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        enc_k = enc1.expand(n, -1, -1).contiguous()
        src_len = src_len_1.expand(n).contiguous()
        generated, gen_len = decoder.generate(
            enc_k, src_len, max_len=max_len, sample_temperature=temp
        )
    t_gen = time.perf_counter() - t0
    t1 = time.perf_counter()
    candidates, infos, hits, misses = evaluate_rollout_batch_cached(
        env, problem, generated, gen_len, cache
    )
    t_verify = time.perf_counter() - t1
    success = [z.exact_residual_zero and z.verified_general for z in infos]
    return {
        "n": n,
        "successes": int(sum(success)),
        "rate": float(np.mean(success)),
        "generation_s": t_gen,
        "verification_s": t_verify,
        "cache_hits": hits,
        "cache_misses": misses,
    }


def find_verified_elite(env, decoder, enc1, src_len_1, problem, device,
                        max_len, temperature, max_samples, batch_size, cache):
    searched = 0
    while searched < max_samples:
        bs = min(batch_size, max_samples - searched)
        with torch.no_grad():
            enc_k = enc1.expand(bs, -1, -1).contiguous()
            src_len = src_len_1.expand(bs).contiguous()
            generated, gen_len = decoder.generate(
                enc_k, src_len, max_len=max_len, sample_temperature=temperature
            )
        candidates, infos, _, _ = evaluate_rollout_batch_cached(
            env, problem, generated, gen_len, cache
        )
        for (ids, words), info in zip(candidates, infos):
            if info.exact_residual_zero and info.verified_general:
                return list(ids), list(words), info, searched + bs
        searched += bs
        print(f"searched={searched}/{max_samples}: no verified elite yet")
    return None, None, None, searched


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}")

    env, encoder, decoder, params, _ = load_pretrained(args.checkpoint, device)
    problem = make_problem(env, args.problem, n=args.n, mode="general", lane_form=args.lane_form)
    src_tokens = equation_to_tokens(env, problem.equation)
    print(f"problem={problem.name}")
    print(f"input_prefix={' '.join(src_tokens)}")

    for p in encoder.parameters():
        p.requires_grad_(False)
    trainable = configure_trainable(decoder, args.scope)
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    print(f"trainable_parameters={sum(p.numel() for p in trainable):,} scope={args.scope}")

    x, src_len_1 = encode_tokens(env, src_tokens, device)
    with torch.no_grad():
        enc1 = encoder("fwd", x=x, lengths=src_len_1, causal=False).transpose(0, 1)

    cache = {}

    # Use a previously SELF-DISCOVERED exact sequence if supplied; otherwise search for one.
    if args.elite_tokens.strip():
        words = args.elite_tokens.strip().split()
        missing = [w for w in words if w not in env.word2id]
        if missing:
            raise ValueError(f"Unknown elite tokens: {missing}")
        ids = [env.word2id[w] for w in words]
        info = score_candidate_ids(env, problem, ids)
        if not (info.exact_residual_zero and info.verified_general):
            raise RuntimeError(
                "Provided --elite-tokens are NOT verified as an exact general solution: "
                f"reward={info.reward} res0={info.exact_residual_zero} rank={info.generality_rank} "
                f"expr={info.expression}"
            )
        searched = 0
        print("using supplied self-discovered verified elite")
    else:
        ids, words, info, searched = find_verified_elite(
            env, decoder, enc1, src_len_1, problem, device,
            args.max_len, args.temperature, args.search_samples,
            args.search_batch_size, cache,
        )
        if ids is None:
            raise RuntimeError(f"No verified solution found in {searched} search samples")

    print("\n[elite]")
    print(f"searched_samples={searched}")
    print(f"len={len(ids)} reward={info.reward:.3f} rank={info.generality_rank}")
    print(f"expr={info.expression}")
    print(f"tokens={' '.join(words)}")

    fout = open(args.save_jsonl, "w", encoding="utf-8")

    lp0 = raw_logp(env, decoder, enc1, src_len_1, ids, device)
    print(f"raw_logP_before={lp0:.6f}")
    ev0 = fresh_eval(
        env, decoder, enc1, src_len_1, problem, device,
        args.eval_rollouts, args.temperature, args.max_len, cache,
    ) if args.eval_rollouts > 0 else None
    if ev0:
        print(f"fresh_before={ev0['successes']}/{ev0['n']} ({100*ev0['rate']:.3f}%)")
    fout.write(json.dumps({"update": 0, "raw_logp": lp0, "eval": ev0}) + "\n")
    fout.flush()

    for u in range(1, args.updates + 1):
        decoder.train()
        lp = token_sequence_logprobs(
            env, decoder, enc1.detach(), src_len_1, [ids], device,
            length_normalize=False,
        )[0]
        loss = -lp
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        decoder.eval()

        should_eval = (u == 1 or u == args.updates or (args.eval_every > 0 and u % args.eval_every == 0))
        if should_eval:
            current_lp = raw_logp(env, decoder, enc1, src_len_1, ids, device)
            ev = fresh_eval(
                env, decoder, enc1, src_len_1, problem, device,
                args.eval_rollouts, args.temperature, args.max_len, cache,
            ) if args.eval_rollouts > 0 else None
            msg = f"update={u:02d} loss={loss.item():.6f} raw_logP={current_lp:.6f}"
            if ev:
                msg += f" fresh={ev['successes']}/{ev['n']} ({100*ev['rate']:.3f}%)"
            print(msg)
            fout.write(json.dumps({
                "update": u,
                "loss": float(loss.item()),
                "raw_logp": current_lp,
                "eval": ev,
            }) + "\n")
            fout.flush()

    fout.close()
    lpf = raw_logp(env, decoder, enc1, src_len_1, ids, device)
    print("\n[summary]")
    print(f"raw_logP: {lp0:.6f} -> {lpf:.6f}  delta={lpf-lp0:+.6f}")
    print(f"result_log={args.save_jsonl}")


if __name__ == "__main__":
    main()
