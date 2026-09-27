#!/usr/bin/env python3
"""Pre-exact GRPO with Reward v3.1, then exact-verifier replay consolidation.

Key scientific design:
  * No reference / target solution is used.
  * Before first exact discovery, GRPO optimizes Reward v3.1 only.
  * A frozen-base rollout fraction is used for discovery support, but those
    off-policy samples NEVER enter the GRPO policy-gradient loss.
  * If any exact candidate appears (policy OR frozen-base), the dense-reward
    update is skipped and the run switches immediately to the already-tested
    exact-verifier + replay consolidation phase.
  * --preexact-mode frozen gives a compute-matched no-learning control.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import (
    configure_trainable,
    load_pretrained,
    sequence_logprobs,
    token_sequence_logprobs,
)
from ttrl_ode_benchmark.common import (
    capture_rng_state,
    decode_batch,
    decode_greedy,
    encode_problem,
    make_problem_from_record,
    read_jsonl,
    restore_rng_state,
    seed_all,
)
from ttrl_ode_benchmark.exact_verifier import (
    ExactODEVerifier,
    ParallelVerifierPool,
    exact_count,
    resolve_verifier_workers,
    timeout_count,
)
from ttrl_ode_benchmark.grpo_utils import (
    grpo_clipped_loss,
    group_normalized_advantages,
    mean_sampled_kl,
    teacher_forced_token_stats,
)
from ttrl_ode_benchmark.rich_reward_v31 import ParallelRewardV31Pool


def parse_args():
    p = argparse.ArgumentParser(description="Reward-v3.1 pre-exact GRPO for symbolic ODE2 TTRL")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--problems", required=True)
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--output", default="grpo_v31_results.jsonl")
    p.add_argument("--summary", default="grpo_v31_summary.json")

    # Search budget.
    p.add_argument("--warmup-samples", type=int, default=256)
    p.add_argument("--warmup-batch-size", type=int, default=64)
    p.add_argument("--warmup-temperature", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--rollouts", type=int, default=64,
                   help="total search candidates per step; policy+base mixture sums to this")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=128)

    # Pre-exact discovery policy.
    p.add_argument("--preexact-mode", choices=["grpo", "frozen"], default="grpo")
    p.add_argument("--base-rollout-fraction", type=float, default=0.25,
                   help="frozen-base samples used for discovery only; never used in GRPO loss")
    p.add_argument("--scope", choices=["proj", "last_layer", "decoder"], default="last_layer")
    p.add_argument("--grpo-lr", type=float, default=1e-6)
    p.add_argument("--grpo-epochs", type=int, default=1)
    p.add_argument("--grpo-clip", type=float, default=0.20)
    p.add_argument("--adv-clip", type=float, default=5.0)
    p.add_argument("--reward-std-min", type=float, default=1e-5)
    p.add_argument("--kl-coef", type=float, default=0.02,
                   help="token-level KL to original pretrained policy")
    p.add_argument("--entropy-coef", type=float, default=0.002)
    p.add_argument("--max-old-kl", type=float, default=0.08,
                   help="diagnostic/early-stop threshold across repeated GRPO epochs")
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Reward-v3.1 compute.
    p.add_argument("--reward-timeout", type=float, default=6.0)
    p.add_argument("--reward-workers", type=int, default=16)
    p.add_argument("--reward-seed-stride", type=int, default=104729)

    # Exact verifier + post-discovery consolidation (kept close to prior runner).
    p.add_argument("--candidate-timeout", type=float, default=4.0)
    p.add_argument("--verifier-workers", type=int, default=16)
    p.add_argument("--post-lr", type=float, default=3e-6)
    p.add_argument("--post-length-normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--replay-updates-new-exact", type=int, default=20)
    p.add_argument("--replay-updates", type=int, default=2)
    p.add_argument("--replay-weight", type=float, default=1.0)
    p.add_argument("--max-exact-buffer", type=int, default=32)
    p.add_argument("--max-approx-buffer", type=int, default=8)

    # Held-out evaluation.
    p.add_argument("--eval-rollouts", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def _set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = float(lr)


def _snapshot_trainable(trainable):
    return [p.detach().cpu().clone() for p in trainable]


def _load_trainable(trainable, state):
    with torch.no_grad():
        for p, s in zip(trainable, state):
            p.copy_(s.to(device=p.device, dtype=p.dtype))


@contextmanager
def _temporary_trainable_state(trainable, target_state, current_state=None):
    if current_state is None:
        current_state = _snapshot_trainable(trainable)
    _load_trainable(trainable, target_state)
    try:
        yield
    finally:
        _load_trainable(trainable, current_state)


def candidate_entry(cand, z, step, source):
    return {
        "ids": list(map(int, cand[0])),
        "tokens": " ".join(cand[1]),
        "expr": str(z.expression),
        "reward": float(z.reward),
        "exact": bool(z.verified_general),
        "mse": float(z.numerical_mse),
        "rank": int(z.generality_rank),
        "step": int(step),
        "source": str(source),
        "length": len(cand[0]),
    }


def trim_buffer(buf, max_exact, max_approx):
    exact = sorted([e for e in buf if e["exact"]], key=lambda e: (e["length"], -e["reward"], e["step"]))[:max_exact]
    approx = sorted([e for e in buf if not e["exact"]], key=lambda e: (-e["reward"], e["length"], e["step"]))[:max_approx]
    buf[:] = exact + approx


def update_buffer(buf, cands, infos, step, source, max_exact, max_approx):
    existing = {tuple(e["ids"]) for e in buf}
    new_exact = []
    for c, z in zip(cands, infos):
        if not z.valid_parse or not np.isfinite(float(z.reward)):
            continue
        key = tuple(map(int, c[0]))
        if key in existing:
            continue
        e = candidate_entry(c, z, step, source)
        buf.append(e)
        existing.add(key)
        if e["exact"]:
            new_exact.append(e)
    trim_buffer(buf, max_exact, max_approx)
    keep = {tuple(e["ids"]) for e in buf}
    return [e for e in new_exact if tuple(e["ids"]) in keep]


def replay(env, decoder, enc1, src_len, buf, optimizer, trainable, device, updates, weight, clip):
    if not buf or updates <= 0:
        return None
    exact = [e for e in buf if e["exact"]]
    if not exact:
        return None
    seqs = [e["ids"] for e in exact]
    losses = []
    decoder.eval()  # gradients still flow; deterministic dropout behavior helps reproducibility.
    for _ in range(updates):
        lp = token_sequence_logprobs(
            env, decoder, enc1.detach(), src_len, seqs, device,
            length_normalize=False,
        )
        loss = -float(weight) * lp.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return {"updates": updates, "n_replayed": len(seqs), "loss_first": losses[0], "loss_last": losses[-1]}


def eval_policy(decoder, enc1, src_len, verifier, n, max_len, temp, cache):
    if n <= 0:
        return {"n": 0, "exact": 0, "rate": 0.0, "timeouts": 0}
    gen, glen = decode_batch(decoder, enc1, src_len, n, max_len, temp)
    cands, infos, h, m = verifier.evaluate_generated(gen, glen, cache)
    ne = exact_count(infos)
    examples = []
    for c, z in zip(cands, infos):
        if z.verified_general and len(examples) < 3:
            examples.append({"expr": str(z.expression), "tokens": " ".join(c[1]), "reward": float(z.reward)})
    return {
        "n": n,
        "exact": ne,
        "rate": float(ne / n),
        "timeouts": timeout_count(infos),
        "best_reward": float(max(z.reward for z in infos)),
        "examples": examples,
        "cache_hits": h,
        "cache_misses": m,
    }


def _score_dense_unique(pool, env, problem, cands, timeout_s, probe_seed):
    key_to_ids = {}
    for ids, _ in cands:
        key = tuple(map(int, ids))
        key_to_ids.setdefault(key, ids)
    keys = list(key_to_ids)
    infos_unique = pool.score_many(
        env, problem, timeout_s,
        [key_to_ids[k] for k in keys],
        probe_seed=probe_seed,
    )
    lookup = dict(zip(keys, infos_unique))
    infos = [lookup[tuple(map(int, ids))] for ids, _ in cands]
    return infos, len(keys)


def _policy_old_stats(decoder, enc1, src_len, gen, glen, temperature):
    bs = gen.shape[1]
    with torch.no_grad():
        lp, _, mask = teacher_forced_token_stats(
            decoder,
            enc1.detach().expand(bs, -1, -1).contiguous(),
            src_len.expand(bs).contiguous(),
            gen,
            glen,
            temperature=temperature,
            require_entropy=False,
        )
    return lp.detach(), mask.detach()


def _base_reference_and_rollouts(
    decoder, trainable, base_state, current_state,
    enc1, src_len, policy_gen, policy_glen,
    n_base, max_len, temperature, kl_needed,
):
    base_gen = base_glen = ref_lp = None
    with _temporary_trainable_state(trainable, base_state, current_state=current_state):
        decoder.eval()
        if n_base > 0:
            base_gen, base_glen = decode_batch(decoder, enc1, src_len, n_base, max_len, temperature)
        if kl_needed and policy_gen is not None and policy_gen.shape[1] > 0:
            bs = policy_gen.shape[1]
            with torch.no_grad():
                ref_lp, _, _ = teacher_forced_token_stats(
                    decoder,
                    enc1.detach().expand(bs, -1, -1).contiguous(),
                    src_len.expand(bs).contiguous(),
                    policy_gen,
                    policy_glen,
                    temperature=temperature,
                    require_entropy=False,
                )
                ref_lp = ref_lp.detach()
    decoder.eval()
    return base_gen, base_glen, ref_lp


def _grpo_update(
    decoder, optimizer, trainable, enc1, src_len, gen, glen,
    old_lp, ref_lp, old_mask, rewards, a,
):
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=gen.device)
    advantages = group_normalized_advantages(rewards_t, clip=a.adv_clip)
    epoch_infos = []
    old_kl_after = 0.0
    decoder.eval()
    for epoch in range(max(1, int(a.grpo_epochs))):
        bs = gen.shape[1]
        cur_lp, entropy, mask = teacher_forced_token_stats(
            decoder,
            enc1.detach().expand(bs, -1, -1).contiguous(),
            src_len.expand(bs).contiguous(),
            gen,
            glen,
            temperature=a.temperature,
            require_entropy=(a.entropy_coef != 0.0),
        )
        # Generation lengths are fixed; masks should exactly match old masks.
        if not torch.equal(mask, old_mask):
            raise RuntimeError("teacher-forced token mask changed for fixed sampled trajectories")
        loss, info = grpo_clipped_loss(
            cur_lp, old_lp, ref_lp, entropy, mask, advantages,
            clip_eps=a.grpo_clip,
            kl_coef=a.kl_coef,
            entropy_coef=a.entropy_coef,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, a.grad_clip)
        optimizer.step()

        with torch.no_grad():
            chk_lp, _, chk_mask = teacher_forced_token_stats(
                decoder,
                enc1.detach().expand(bs, -1, -1).contiguous(),
                src_len.expand(bs).contiguous(),
                gen,
                glen,
                temperature=a.temperature,
                require_entropy=False,
            )
            old_kl_after = mean_sampled_kl(chk_lp, old_lp, chk_mask)
        row = info.to_dict()
        row.update({"epoch": epoch, "grad_norm": float(grad_norm), "old_kl_after": float(old_kl_after)})
        epoch_infos.append(row)
        if a.max_old_kl > 0 and old_kl_after > a.max_old_kl:
            row["early_stop_old_kl"] = True
            break

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "adv_min": float(advantages.min().item()),
        "adv_max": float(advantages.max().item()),
        "epochs": epoch_infos,
        "old_kl_after": float(old_kl_after),
    }


def _post_exact_reinforce(decoder, optimizer, trainable, enc1, src_len, gen, glen, infos, a):
    rewards_np = np.asarray([z.reward for z in infos], dtype=np.float32)
    if float(rewards_np.std()) <= 1e-8:
        return None
    rewards = torch.tensor(rewards_np, device=gen.device)
    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
    decoder.eval()
    lp = sequence_logprobs(
        decoder,
        enc1.detach().expand(gen.shape[1], -1, -1).contiguous(),
        src_len.expand(gen.shape[1]).contiguous(),
        gen,
        glen,
        length_normalize=a.post_length_normalize,
    )
    loss = -(adv.detach() * lp).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, a.grad_clip)
    optimizer.step()
    return {"loss": float(loss.item()), "grad_norm": float(grad_norm), "reward_std": float(rewards_np.std())}


def main():
    a = parse_args()
    if not 0.0 <= a.base_rollout_fraction < 1.0:
        raise ValueError("--base-rollout-fraction must be in [0,1)")
    if a.rollouts < 2:
        raise ValueError("--rollouts must be >=2")
    # IMPORTANT: create CPU-only symbolic worker pools BEFORE any CUDA call.
    # On Linux we use fork for fast startup. Forking after CUDA initialization is unsafe;
    # using spawn here is extremely expensive because every worker re-imports PyTorch/SymPy.
    vworkers = resolve_verifier_workers(a.verifier_workers)
    rworkers = max(1, int(a.reward_workers))
    print(f"starting_cpu_pools verifier_workers={vworkers} reward_workers={rworkers}", flush=True)
    verifier_pool = ParallelVerifierPool(vworkers)
    reward_pool = ParallelRewardV31Pool(rworkers)
    print("cpu_pools_ready", flush=True)

    seed_all(a.seed)
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}", flush=True)
    env, encoder, decoder, params, ckpt = load_pretrained(a.checkpoint, device)
    trainable = configure_trainable(decoder, a.scope)
    base_state = _snapshot_trainable(trainable)
    print(f"trainable_parameters={sum(p.numel() for p in trainable):,} scope={a.scope}", flush=True)

    problems = list(read_jsonl(a.problems))
    if a.max_problems > 0:
        problems = problems[:a.max_problems]

    fout = open(a.output, "w", encoding="utf-8")
    summaries = []
    try:
        for pi, rec in enumerate(problems):
            # Independent test-time adaptation task.
            _load_trainable(trainable, base_state)
            decoder.eval()
            seed_all(a.seed + 1009 * pi)
            optimizer = torch.optim.Adam(trainable, lr=a.grpo_lr)

            problem = make_problem_from_record(env, rec)
            prefix, enc1, src_len = encode_problem(env, encoder, problem, device)
            verifier = ExactODEVerifier(env, problem, a.candidate_timeout, worker_pool=verifier_pool)
            train_cache, eval_cache = {}, {}
            buf = []
            search_calls = search_unique = dense_calls = 0
            heldout_calls = heldout_unique = 0
            first_exact_by = None
            first_exact_source = None

            print("\n" + "=" * 110)
            print(f"problem {pi+1}/{len(problems)} id={rec.get('id')} mode={a.preexact_mode}")
            print(f"equation={rec.get('equation_latex')}")
            print(f"prefix={' '.join(prefix)}")

            # Held-out frozen baseline; RNG restored so it does not affect search trajectories.
            st = capture_rng_state()
            baseline = eval_policy(decoder, enc1, src_len, verifier, a.eval_rollouts, a.max_len, a.eval_temperature, eval_cache)
            restore_rng_state(st)
            heldout_calls += a.eval_rollouts
            heldout_unique += baseline.get("cache_misses", 0)
            gg, gl = decode_greedy(decoder, enc1, src_len, a.max_len)
            _, ginfo, _, _ = verifier.evaluate_generated(gg, gl, {})
            baseline["greedy_exact"] = bool(ginfo[0].verified_general)
            print(f"baseline exact={baseline['exact']}/{baseline['n']} ({100*baseline['rate']:.2f}%) greedy={int(baseline['greedy_exact'])}")

            # Frozen warmup.
            warm_exact = 0
            seen = 0
            while seen < a.warmup_samples:
                bs = min(a.warmup_batch_size, a.warmup_samples - seen)
                gen, glen = decode_batch(decoder, enc1, src_len, bs, a.max_len, a.warmup_temperature)
                cands, infos, h, m = verifier.evaluate_generated(gen, glen, train_cache)
                ne = exact_count(infos)
                warm_exact += ne
                newe = update_buffer(buf, cands, infos, -1, "warmup", a.max_exact_buffer, a.max_approx_buffer)
                search_calls += bs
                search_unique += m
                if ne and first_exact_by is None:
                    first_exact_by = search_calls
                    first_exact_source = "warmup"
                seen += bs
                print(f"  warmup {seen}/{a.warmup_samples} exact={warm_exact}")

            have_exact = any(e["exact"] for e in buf)
            warm_replay = None
            if have_exact:
                _set_lr(optimizer, a.post_lr)
                warm_replay = replay(
                    env, decoder, enc1, src_len, buf, optimizer, trainable, device,
                    a.replay_updates_new_exact, a.replay_weight, a.grad_clip,
                )
            print(f"warmup summary exact={warm_exact}/{a.warmup_samples} replay={warm_replay}")

            step_rows = []
            for step in range(a.steps):
                have_exact_before = any(e["exact"] for e in buf)
                phase = "post_exact" if have_exact_before else "pre_exact"
                dense_info = None
                post_rl = None
                rep = None
                exact_source_this_step = []

                if have_exact_before:
                    # Known-good consolidation stage: current policy only.
                    _set_lr(optimizer, a.post_lr)
                    n_policy, n_base = a.rollouts, 0
                    gen, glen = decode_batch(decoder, enc1, src_len, n_policy, a.max_len, a.temperature)
                    cands, infos, h, m = verifier.evaluate_generated(gen, glen, train_cache)
                    search_calls += n_policy
                    search_unique += m
                    ne = exact_count(infos)
                    newe = update_buffer(buf, cands, infos, step, "policy_post", a.max_exact_buffer, a.max_approx_buffer)
                    if ne:
                        exact_source_this_step.append("policy_post")
                    post_rl = _post_exact_reinforce(decoder, optimizer, trainable, enc1, src_len, gen, glen, infos, a)
                    nup = a.replay_updates_new_exact if newe else a.replay_updates
                    rep = replay(env, decoder, enc1, src_len, buf, optimizer, trainable, device, nup, a.replay_weight, a.grad_clip)
                    train_exact = ne
                    policy_exact = ne
                    base_exact = 0
                else:
                    # Pre-exact search. Only policy samples are on-policy for GRPO.
                    if a.preexact_mode == "grpo":
                        n_base = int(round(a.rollouts * a.base_rollout_fraction))
                        n_policy = a.rollouts - n_base
                        n_policy = max(2, n_policy)
                        n_base = a.rollouts - n_policy
                    else:
                        n_policy, n_base = a.rollouts, 0

                    decoder.eval()
                    policy_gen, policy_glen = decode_batch(decoder, enc1, src_len, n_policy, a.max_len, a.temperature)
                    old_lp = old_mask = None
                    if a.preexact_mode == "grpo":
                        old_lp, old_mask = _policy_old_stats(decoder, enc1, src_len, policy_gen, policy_glen, a.temperature)

                    current_state = _snapshot_trainable(trainable) if (a.preexact_mode == "grpo" and (n_base > 0 or a.kl_coef != 0.0)) else None
                    base_gen = base_glen = ref_lp = None
                    if a.preexact_mode == "grpo" and (n_base > 0 or a.kl_coef != 0.0):
                        base_gen, base_glen, ref_lp = _base_reference_and_rollouts(
                            decoder, trainable, base_state, current_state,
                            enc1, src_len, policy_gen, policy_glen,
                            n_base, a.max_len, a.temperature, a.kl_coef != 0.0,
                        )

                    pcands, pinfos, ph, pm = verifier.evaluate_generated(policy_gen, policy_glen, train_cache)
                    policy_exact = exact_count(pinfos)
                    new_policy = update_buffer(buf, pcands, pinfos, step, "policy_pre", a.max_exact_buffer, a.max_approx_buffer)
                    search_calls += n_policy
                    search_unique += pm
                    if policy_exact:
                        exact_source_this_step.append("policy_pre")

                    bcands, binfos = [], []
                    base_exact = 0
                    new_base = []
                    if n_base > 0:
                        bcands, binfos, bh, bm = verifier.evaluate_generated(base_gen, base_glen, train_cache)
                        base_exact = exact_count(binfos)
                        new_base = update_buffer(buf, bcands, binfos, step, "base_pre", a.max_exact_buffer, a.max_approx_buffer)
                        search_calls += n_base
                        search_unique += bm
                        if base_exact:
                            exact_source_this_step.append("base_pre")

                    train_exact = policy_exact + base_exact
                    have_exact_now = any(e["exact"] for e in buf)
                    if have_exact_now:
                        if first_exact_by is None:
                            first_exact_by = search_calls
                            first_exact_source = "+".join(exact_source_this_step) if exact_source_this_step else "pre_exact_batch"
                        # Critical safety rule: exact batch gets NO dense-reward update.
                        _set_lr(optimizer, a.post_lr)
                        nup = a.replay_updates_new_exact if (new_policy or new_base) else a.replay_updates
                        rep = replay(env, decoder, enc1, src_len, buf, optimizer, trainable, device, nup, a.replay_weight, a.grad_clip)
                    elif a.preexact_mode == "grpo":
                        # Fresh randomized Reward-v3.1 probes for this optimization step.
                        probe_seed = a.seed + 1000003 * pi + a.reward_seed_stride * (step + 1)
                        rinfos, nuniq = _score_dense_unique(
                            reward_pool, env, problem, pcands,
                            a.reward_timeout, probe_seed,
                        )
                        dense_calls += nuniq
                        rewards = np.asarray([r.pre_reward for r in rinfos], dtype=np.float32)
                        reward_std = float(rewards.std())
                        if np.isfinite(rewards).all() and reward_std > a.reward_std_min:
                            _set_lr(optimizer, a.grpo_lr)
                            dense_info = _grpo_update(
                                decoder, optimizer, trainable, enc1, src_len,
                                policy_gen, policy_glen, old_lp, ref_lp, old_mask,
                                rewards, a,
                            )
                            dense_info.update({
                                "probe_seed": int(probe_seed),
                                "unique_dense_scores": int(nuniq),
                                "gate_failures": int(sum(not r.generality_gate_pass for r in rinfos)),
                                "invalid_rewards": int(sum(r.validity < 1.0 for r in rinfos)),
                                "top_examples": [
                                    {
                                        "reward": float(rinfos[i].pre_reward),
                                        "expr": rinfos[i].expression,
                                        "gate": bool(rinfos[i].generality_gate_pass),
                                        "eq": float(rinfos[i].equation_score),
                                        "dmin": float(rinfos[i].direction_min_score),
                                    }
                                    for i in np.argsort(-rewards)[:3]
                                ],
                            })
                        else:
                            dense_info = {
                                "skipped": True,
                                "reason": "reward_std_too_small_or_nonfinite",
                                "reward_mean": float(np.mean(rewards)),
                                "reward_std": reward_std,
                                "reward_min": float(np.min(rewards)),
                                "reward_max": float(np.max(rewards)),
                                "probe_seed": int(probe_seed),
                                "unique_dense_scores": int(nuniq),
                            }

                # Greedy + optional held-out exact evaluation.
                gg, gl = decode_greedy(decoder, enc1, src_len, a.max_len)
                _, ginfo, _, _ = verifier.evaluate_generated(gg, gl, {})
                ev = None
                if a.eval_every > 0 and (step + 1) % a.eval_every == 0:
                    st = capture_rng_state()
                    ev = eval_policy(decoder, enc1, src_len, verifier, a.eval_rollouts, a.max_len, a.eval_temperature, eval_cache)
                    restore_rng_state(st)
                    heldout_calls += a.eval_rollouts
                    heldout_unique += ev.get("cache_misses", 0)

                msg = (
                    f"  step={step:02d} phase={phase} policy/base={n_policy}/{n_base} "
                    f"exact={train_exact} (p={policy_exact},b={base_exact}) greedy={int(ginfo[0].verified_general)}"
                )
                if dense_info and not dense_info.get("skipped"):
                    msg += f" R31={dense_info['reward_mean']:.3f}±{dense_info['reward_std']:.3f} KLold={dense_info['old_kl_after']:.4f}"
                if ev:
                    msg += f" heldout={ev['exact']}/{ev['n']} ({100*ev['rate']:.1f}%)"
                print(msg)

                step_rows.append({
                    "step": step,
                    "phase": phase,
                    "policy_rollouts": n_policy,
                    "base_rollouts": n_base,
                    "train_exact": train_exact,
                    "policy_exact": policy_exact,
                    "base_exact": base_exact,
                    "exact_source": exact_source_this_step,
                    "greedy_exact": bool(ginfo[0].verified_general),
                    "grpo": dense_info,
                    "post_exact_rl": post_rl,
                    "replay": rep,
                    "eval": ev,
                    "search_calls": search_calls,
                    "dense_reward_unique_calls": dense_calls,
                })

            st = capture_rng_state()
            final = eval_policy(decoder, enc1, src_len, verifier, a.eval_rollouts, a.max_len, a.eval_temperature, eval_cache)
            restore_rng_state(st)
            heldout_calls += a.eval_rollouts
            heldout_unique += final.get("cache_misses", 0)
            gg, gl = decode_greedy(decoder, enc1, src_len, a.max_len)
            _, ginfo, _, _ = verifier.evaluate_generated(gg, gl, {})

            summary = {
                "id": rec.get("id"),
                "equation_latex": rec.get("equation_latex"),
                "preexact_mode": a.preexact_mode,
                "baseline": baseline,
                "warmup_exact": warm_exact,
                "warmup_samples": a.warmup_samples,
                "first_exact_by_search_calls": first_exact_by,
                "first_exact_source": first_exact_source,
                "final": final,
                "final_greedy_exact": bool(ginfo[0].verified_general),
                "final_greedy_expr": str(ginfo[0].expression),
                "search_candidate_samples": search_calls,
                "search_unique_verifier_evals": search_unique,
                "dense_reward_unique_calls": dense_calls,
                "heldout_candidate_samples": heldout_calls,
                "heldout_unique_verifier_evals": heldout_unique,
                "steps": step_rows,
            }
            summaries.append(summary)
            fout.write(json.dumps(summary, ensure_ascii=False) + "\n")
            fout.flush()
            print(
                f"FINAL exact={final['exact']}/{final['n']} ({100*final['rate']:.2f}%) "
                f"greedy={int(summary['final_greedy_exact'])} first_exact={first_exact_by} "
                f"search={search_calls} dense={dense_calls}"
            )
    finally:
        fout.close()
        verifier_pool.close()
        reward_pool.close()

    discovered = [s["first_exact_by_search_calls"] is not None for s in summaries]
    agg = {
        "n_problems": len(summaries),
        "preexact_mode": a.preexact_mode,
        "discovery_rate": float(np.mean(discovered)) if summaries else 0.0,
        "n_discovered": int(sum(discovered)),
        "greedy_exact_before": int(sum(bool(s["baseline"].get("greedy_exact", False)) for s in summaries)),
        "greedy_exact_after": int(sum(bool(s["final_greedy_exact"]) for s in summaries)),
        "mean_baseline_exact_rate": float(np.mean([s["baseline"]["rate"] for s in summaries])) if summaries else 0.0,
        "mean_final_exact_rate": float(np.mean([s["final"]["rate"] for s in summaries])) if summaries else 0.0,
        "mean_first_exact_calls_discovered": float(np.mean([s["first_exact_by_search_calls"] for s in summaries if s["first_exact_by_search_calls"] is not None])) if any(discovered) else None,
        "per_problem": [{k: v for k, v in s.items() if k != "steps"} for s in summaries],
    }
    with open(a.summary, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print("\n=== aggregate ===")
    print(json.dumps({k: v for k, v in agg.items() if k != "per_problem"}, indent=2))
    print(f"results={a.output}\nsummary={a.summary}")


if __name__ == "__main__":
    main()
