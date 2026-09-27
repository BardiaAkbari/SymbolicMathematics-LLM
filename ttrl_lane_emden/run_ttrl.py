#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

# Allow running as `python ttrl_lane_emden/run_ttrl.py` from repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import (
    configure_trainable,
    encode_tokens,
    equation_to_tokens,
    evaluate_rollout_batch,
    evaluate_rollout_batch_cached,
    load_pretrained,
    make_problem,
    sample_rollouts,
    sequence_logprobs,
    summarize_infos,
    token_sequence_logprobs,
)


def parse_args():
    p = argparse.ArgumentParser(description="Verifier-guided test-time RL for Lample--Charton ODE2")
    p.add_argument("--checkpoint", required=True, help="Path to official ode2.pth")
    p.add_argument("--problem", choices=["harmonic", "lane_emden"], default="lane_emden")
    p.add_argument("--n", type=int, default=1, help="Lane--Emden index")
    p.add_argument("--lane-form", choices=["standard", "cleared"], default="cleared")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--rollouts", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--scope", choices=["proj", "last_layer", "decoder"], default="last_layer")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-jsonl", default="ttrl_lane_emden_elite_results.jsonl")
    p.add_argument("--save-decoder", default="", help="Optional path to save adapted decoder state_dict")
    p.add_argument("--length-normalize", action=argparse.BooleanOptionalAction, default=True,
                   help="Length-normalize ordinary RL rollout log-probs (elite log-probs are never normalized)")

    # Verified-elite replay.  These are deliberately explicit so the experiment can be ablated.
    p.add_argument("--elite-updates", type=int, default=10,
                   help="Extra MLE-style updates immediately after discovering a NEW verified solution")
    p.add_argument("--elite-replay-updates", type=int, default=1,
                   help="Replay updates on the elite buffer at later TTRL steps")
    p.add_argument("--elite-weight", type=float, default=1.0)
    p.add_argument("--max-elites", type=int, default=4,
                   help="Keep up to this many unique verified solutions; shortest are retained")

    # Fresh, non-training evaluation samples.  Set eval-rollouts=0 to disable.
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-rollouts", type=int, default=64)
    p.add_argument("--eval-temperature", type=float, default=1.0)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_top(rows):
    for r in rows:
        print(
            f"  idx={r['idx']:>2} reward={r['reward']:>8.3f} relres={r['mse']:.3e} "
            f"rank={r['rank']} res0={int(r['res0'])} gen={int(r['verified_general'])} :: {r['expr']}"
        )


def greedy_decode(env, encoder, decoder, problem, device, max_len):
    _, generated, glen = sample_rollouts(
        env, encoder, decoder, problem.equation,
        n_samples=1, temperature=None, max_len=max_len, device=device
    )
    candidates, infos = evaluate_rollout_batch(env, problem, generated, glen)
    return candidates[0], infos[0]


def elite_ids(elites):
    return [e["ids"] for e in elites]


def elite_logprob_stats(env, decoder, enc1, src_len_1, elites, device):
    if not elites:
        return None
    decoder.eval()
    with torch.no_grad():
        lp = token_sequence_logprobs(
            env, decoder, enc1, src_len_1, elite_ids(elites), device,
            length_normalize=False,
        )
    vals = lp.detach().cpu().numpy().astype(float)
    return {
        "mean": float(vals.mean()),
        "max": float(vals.max()),
        "min": float(vals.min()),
        "per_elite": vals.tolist(),
    }


def update_elite_buffer(elites, candidates, infos, step, max_elites):
    """Add newly verified sequences, deduplicate, and keep the shortest unique elites."""
    existing = {tuple(e["ids"]) for e in elites}
    new = []
    for (ids, words), info in zip(candidates, infos):
        if not (info.exact_residual_zero and info.verified_general):
            continue
        key = tuple(int(x) for x in ids)
        if key in existing:
            continue
        e = {
            "ids": list(key),
            "words": list(words),
            "expr": str(info.expression),
            "reward": float(info.reward),
            "discovered_step": int(step),
            "length": len(key),
        }
        elites.append(e)
        new.append(e)
        existing.add(key)

    # Simplicity bias among exact solutions: retain shortest sequences.
    elites.sort(key=lambda e: (e["length"], e["discovered_step"]))
    if len(elites) > max_elites:
        kept_keys = {tuple(e["ids"]) for e in elites[:max_elites]}
        new = [e for e in new if tuple(e["ids"]) in kept_keys]
        del elites[max_elites:]
    return new


def run_elite_updates(env, decoder, enc1, src_len_1, elites, optimizer, trainable,
                      device, n_updates, weight, grad_clip):
    if not elites or n_updates <= 0:
        return None
    losses = []
    decoder.train()
    for _ in range(n_updates):
        lp = token_sequence_logprobs(
            env, decoder, enc1.detach(), src_len_1, elite_ids(elites), device,
            length_normalize=False,  # IMPORTANT: exact trajectories get full sequence credit.
        )
        loss = -float(weight) * lp.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    decoder.eval()
    return {
        "updates": int(n_updates),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": float(np.mean(losses)),
    }


def fresh_eval(env, decoder, enc1, src_len_1, problem, device, n_samples,
               temperature, max_len, verifier_cache):
    if n_samples <= 0:
        return None
    decoder.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        enc_k = enc1.expand(n_samples, -1, -1).contiguous()
        src_len = src_len_1.expand(n_samples).contiguous()
        generated, gen_len = decoder.generate(
            enc_k, src_len, max_len=max_len, sample_temperature=temperature
        )
    t_gen = time.perf_counter() - t0
    t1 = time.perf_counter()
    candidates, infos, hits, misses = evaluate_rollout_batch_cached(
        env, problem, generated, gen_len, verifier_cache
    )
    t_verify = time.perf_counter() - t1
    successes = [i.exact_residual_zero and i.verified_general for i in infos]
    rewards = np.asarray([i.reward for i in infos], dtype=np.float64)
    top = summarize_infos(candidates, infos, top_k=1)[0]
    return {
        "n": int(n_samples),
        "successes": int(sum(successes)),
        "exact_rate": float(np.mean(successes)),
        "best_reward": float(rewards.max()),
        "top_expr": top["expr"],
        "cache_hits": int(hits),
        "cache_misses": int(misses),
        "generation_s": float(t_gen),
        "verification_s": float(t_verify),
    }


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}")

    env, encoder, decoder, params, _ = load_pretrained(args.checkpoint, device)
    problem = make_problem(env, args.problem, n=args.n, mode="general", lane_form=args.lane_form)
    src_tokens = equation_to_tokens(env, problem.equation)
    print(f"problem={problem.name}")
    print(f"equation={problem.equation}")
    print(f"input_prefix={' '.join(src_tokens)}")
    print(f"expected_reference={problem.expected}")
    print(f"checkpoint_arch emb={params.emb_dim} enc={params.n_enc_layers} dec={params.n_dec_layers} heads={params.n_heads}")

    for p in encoder.parameters():
        p.requires_grad_(False)
    trainable = configure_trainable(decoder, args.scope)
    print(f"trainable_parameters={sum(p.numel() for p in trainable):,} scope={args.scope}")
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    print("\n[baseline greedy]")
    _, info = greedy_decode(env, encoder, decoder, problem, device, args.max_len)
    print(f"reward={info.reward:.3f} rank={info.generality_rank} residual_zero={info.exact_residual_zero}")
    print(f"expr={info.expression}")

    # Encoder representation is frozen for this per-instance TTRL experiment.
    x, src_len_1 = encode_tokens(env, src_tokens, device)
    with torch.no_grad():
        enc1 = encoder("fwd", x=x, lengths=src_len_1, causal=False).transpose(0, 1)

    verifier_cache = {}
    elites = []
    fout = open(args.save_jsonl, "w", encoding="utf-8")

    # Baseline fresh sampling estimate, before ANY adaptation.
    if args.eval_rollouts > 0:
        print(f"\n[baseline fresh eval: {args.eval_rollouts} samples]")
        ev = fresh_eval(
            env, decoder, enc1, src_len_1, problem, device,
            args.eval_rollouts, args.eval_temperature, args.max_len, verifier_cache,
        )
        print(
            f"success={ev['successes']}/{ev['n']} rate={100*ev['exact_rate']:.3f}% "
            f"best_reward={ev['best_reward']:.3f} "
            f"time(gen={ev['generation_s']:.2f}s verify={ev['verification_s']:.2f}s)"
        )
        fout.write(json.dumps({"event": "baseline_eval", "eval": ev}) + "\n")
        fout.flush()

    for step in range(args.steps):
        t_step = time.perf_counter()

        # -------- sample train rollouts --------
        decoder.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            enc_k = enc1.expand(args.rollouts, -1, -1).contiguous()
            src_len = src_len_1.expand(args.rollouts).contiguous()
            generated, gen_len = decoder.generate(
                enc_k, src_len, max_len=args.max_len, sample_temperature=args.temperature
            )
        t_generation = time.perf_counter() - t0

        # -------- mathematical rewards --------
        t0 = time.perf_counter()
        candidates, infos, cache_hits, cache_misses = evaluate_rollout_batch_cached(
            env, problem, generated, gen_len, verifier_cache
        )
        t_verification = time.perf_counter() - t0

        rewards_np = np.array([z.reward for z in infos], dtype=np.float32)
        full_success = [z.exact_residual_zero and z.verified_general for z in infos]
        top = summarize_infos(candidates, infos, top_k=5)
        print(
            f"\n[step {step:02d}] reward mean={rewards_np.mean():.3f} std={rewards_np.std():.3f} "
            f"max={rewards_np.max():.3f} successes={sum(full_success)}"
        )
        print_top(top)

        # Persist every exact verified discovery instead of throwing it away.
        new_elites = update_elite_buffer(elites, candidates, infos, step, args.max_elites)
        before_elite_lp = elite_logprob_stats(env, decoder, enc1, src_len_1, elites, device)
        if new_elites:
            print(f"  +++ discovered {len(new_elites)} NEW verified elite(s); buffer={len(elites)} +++")
            for e in new_elites:
                print(f"      elite len={e['length']} :: {e['expr']}")
            if before_elite_lp is not None:
                print(f"      elite raw logP before update: mean={before_elite_lp['mean']:.3f}")

        # -------- ordinary group-relative policy update --------
        t0 = time.perf_counter()
        rl_loss_value = None
        rewards = torch.tensor(rewards_np, device=device)
        if float(rewards.std(unbiased=False).item()) >= 1e-8:
            advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
            decoder.train()
            enc_k = enc1.detach().expand(args.rollouts, -1, -1).contiguous()
            src_len = src_len_1.expand(args.rollouts).contiguous()
            seq_lp = sequence_logprobs(
                decoder, enc_k, src_len, generated, gen_len,
                length_normalize=args.length_normalize,
            )
            loss = -(advantages.detach() * seq_lp).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            rl_loss_value = float(loss.item())
            decoder.eval()
            print(f"  policy_loss={rl_loss_value:.6f}")
        else:
            print("  all rewards equal; skipping ordinary RL update")
        t_rl = time.perf_counter() - t0

        # -------- verified elite replay --------
        t0 = time.perf_counter()
        n_elite_updates = args.elite_updates if new_elites else args.elite_replay_updates
        elite_train_stats = run_elite_updates(
            env, decoder, enc1, src_len_1, elites, optimizer, trainable, device,
            n_updates=n_elite_updates, weight=args.elite_weight, grad_clip=args.grad_clip,
        )
        t_elite = time.perf_counter() - t0
        after_elite_lp = elite_logprob_stats(env, decoder, enc1, src_len_1, elites, device)
        if elite_train_stats is not None:
            print(
                f"  elite_replay updates={elite_train_stats['updates']} "
                f"loss {elite_train_stats['loss_first']:.3f}->{elite_train_stats['loss_last']:.3f}"
            )
            if before_elite_lp is not None and after_elite_lp is not None:
                print(
                    f"  elite raw logP mean: {before_elite_lp['mean']:.3f} -> "
                    f"{after_elite_lp['mean']:.3f} (higher is better)"
                )

        # Greedy is useful but NOT sufficient to diagnose probability movement.
        _, ginfo = greedy_decode(env, encoder, decoder, problem, device, args.max_len)
        print(
            f"  adapted_greedy reward={ginfo.reward:.3f} rank={ginfo.generality_rank} "
            f"res0={int(ginfo.exact_residual_zero)} :: {ginfo.expression}"
        )

        # Fresh samples are never used in the loss.
        eval_stats = None
        if args.eval_rollouts > 0 and args.eval_every > 0 and ((step + 1) % args.eval_every == 0 or new_elites):
            eval_stats = fresh_eval(
                env, decoder, enc1, src_len_1, problem, device,
                args.eval_rollouts, args.eval_temperature, args.max_len, verifier_cache,
            )
            print(
                f"  [fresh eval] exact={eval_stats['successes']}/{eval_stats['n']} "
                f"rate={100*eval_stats['exact_rate']:.3f}% best={eval_stats['best_reward']:.3f} "
                f"time(gen={eval_stats['generation_s']:.2f}s verify={eval_stats['verification_s']:.2f}s)"
            )

        total_s = time.perf_counter() - t_step
        print(
            f"  timing: generate={t_generation:.2f}s verify={t_verification:.2f}s "
            f"rl={t_rl:.2f}s elite={t_elite:.2f}s total={total_s:.2f}s "
            f"cache(hit={cache_hits},miss={cache_misses},size={len(verifier_cache)})"
        )

        row = {
            "step": step,
            "reward_mean": float(rewards_np.mean()),
            "reward_std": float(rewards_np.std()),
            "reward_max": float(rewards_np.max()),
            "n_success": int(sum(full_success)),
            "top": top,
            "elite_buffer_size": len(elites),
            "new_elites": [{k: v for k, v in e.items() if k != "ids"} for e in new_elites],
            "elite_logp_before": before_elite_lp,
            "elite_logp_after": after_elite_lp,
            "elite_train": elite_train_stats,
            "rl_loss": rl_loss_value,
            "greedy": {
                "reward": float(ginfo.reward),
                "res0": bool(ginfo.exact_residual_zero),
                "verified_general": bool(ginfo.verified_general),
                "expr": str(ginfo.expression),
            },
            "eval": eval_stats,
            "timing": {
                "generation_s": t_generation,
                "verification_s": t_verification,
                "rl_s": t_rl,
                "elite_s": t_elite,
                "total_s": total_s,
            },
            "cache": {"hits": cache_hits, "misses": cache_misses, "size": len(verifier_cache)},
        }
        fout.write(json.dumps(row) + "\n")
        fout.flush()

    fout.close()
    print("\n[final greedy]")
    _, final_info = greedy_decode(env, encoder, decoder, problem, device, args.max_len)
    print(
        f"reward={final_info.reward:.3f} rank={final_info.generality_rank} "
        f"residual_zero={final_info.exact_residual_zero} verified_general={final_info.verified_general}"
    )
    print(f"expr={final_info.expression}")
    print(f"elite_buffer_size={len(elites)}")
    if elites:
        stats = elite_logprob_stats(env, decoder, enc1, src_len_1, elites, device)
        print(f"final_elite_raw_logP_mean={stats['mean']:.3f}")
    if args.save_decoder:
        torch.save(decoder.state_dict(), args.save_decoder)
        print(f"saved_decoder={args.save_decoder}")
    print(f"result_log={args.save_jsonl}")


if __name__ == "__main__":
    main()
