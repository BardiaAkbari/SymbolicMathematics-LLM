#!/usr/bin/env python3
"""Verifier-guided test-time RL for the Lane--Emden n=2 physical IVP.

Success is *approximate IVP certification*, not exact symbolic equivalence.
V7 performs a frozen best-of-N warmup search before *any* gradient update, then
uses certification-first, quality-prioritized replay during test-time adaptation.
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
    load_pretrained,
    make_problem,
    sequence_logprobs,
    token_sequence_logprobs,
)
from ttrl_lane_emden.ivp_n2 import LaneEmdenN2Verifier, summarize_ivp


def parse_args():
    p = argparse.ArgumentParser(description='TTRL for Lane--Emden n=2 IVP')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--lane-form', choices=['standard', 'cleared'], default='cleared')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--rollouts', type=int, default=64)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-6)
    p.add_argument('--scope', choices=['proj', 'last_layer', 'decoder'], default='last_layer')
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--length-normalize', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--save-jsonl', default='ttrl_lane_emden_n2_results.jsonl')
    p.add_argument('--save-decoder', default='')

    p.add_argument('--warmup-samples', type=int, default=512,
                   help='Frozen search samples collected before the first gradient update')
    p.add_argument('--warmup-batch-size', type=int, default=64)
    p.add_argument('--warmup-temperature', type=float, default=1.0,
                   help='Sampling temperature for the frozen warmup search')

    p.add_argument('--elite-updates', type=int, default=5,
                   help='MLE replay updates for a newly discovered promising approximate elite')
    p.add_argument('--certified-updates', type=int, default=20,
                   help='MLE replay updates when a newly discovered candidate passes strict IVP certification')
    p.add_argument('--elite-replay-updates', type=int, default=1)
    p.add_argument('--high-quality-updates', type=int, default=10,
                   help='Replay updates when a new candidate beats --high-quality-reward')
    p.add_argument('--high-quality-reward', type=float, default=-0.5)
    p.add_argument('--elite-weight', type=float, default=0.7)
    p.add_argument('--replay-beta', type=float, default=4.0,
                   help='Softmax temperature multiplier for quality-prioritized replay')
    p.add_argument('--max-elites', type=int, default=8)
    p.add_argument('--candidate-timeout', type=float, default=3.0,
                   help='Hard wall-clock seconds allowed for one candidate verification; <=0 disables')

    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--eval-rollouts', type=int, default=128)
    p.add_argument('--eval-temperature', type=float, default=1.0)

    p.add_argument('--certify-ode-rel', type=float, default=5e-2)
    p.add_argument('--certify-ref-nrmse', type=float, default=3e-2)
    p.add_argument('--certify-anchor-rmse', type=float, default=2e-3)
    p.add_argument('--elite-ode-rel', type=float, default=1.2e-1)
    p.add_argument('--elite-ref-nrmse', type=float, default=1.2e-1)
    p.add_argument('--elite-anchor-rmse', type=float, default=1e-2)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def print_top(rows):
    for r in rows:
        print(
            f"  idx={r['idx']:>2} reward={r['reward']:>7.3f} cert={int(r['certified'])} "
            f"elite={int(r['elite_eligible'])} ode={r['ode_rel']:.2e} ref={r['ref_nrmse']:.2e} "
            f"anchor={r['anchor_rmse']:.2e} :: {r['expr']}"
        )
        print(f"      fitted={r['fitted_expr']} coeffs={r['coeffs']}")


def elite_ids(elites):
    return [e['ids'] for e in elites]


def elite_logp_stats(env, decoder, enc1, src_len_1, elites, device):
    if not elites: return None
    decoder.eval()
    with torch.no_grad():
        lp = token_sequence_logprobs(env, decoder, enc1, src_len_1, elite_ids(elites), device,
                                     length_normalize=False)
    v = lp.detach().cpu().numpy().astype(float)
    return {'mean': float(v.mean()), 'max': float(v.max()), 'min': float(v.min()), 'per_elite': v.tolist()}


def buffer_entry(candidate, z, step):
    ids, words = candidate
    key = tuple(int(i) for i in ids)
    return {
        'ids': list(key), 'words': list(words),
        'expr': str(z.expression) if z.expression is not None else '<parse error>',
        'fitted_expr': str(z.fitted_expression) if z.fitted_expression is not None else '<invalid>',
        'reward': float(z.reward), 'certified': bool(z.certified_ivp),
        'elite_eligible': bool(z.elite_eligible),
        'ode_rel': float(z.ode_rel_mse), 'ref_nrmse': float(z.ref_nrmse),
        'anchor_rmse': float(z.anchor_rmse), 'coeffs': dict(z.fitted_coefficients),
        'step': int(step), 'length': len(key),
    }


def trim_replay_buffer(elites, max_elites):
    """Keep every certified trajectory plus top-K non-certified trajectories."""
    cert = sorted(
        [e for e in elites if e['certified']],
        key=lambda e: (-e['reward'], e['length'], e['step']),
    )
    approx = sorted(
        [e for e in elites if not e['certified']],
        key=lambda e: (-e['reward'], e['length'], e['step']),
    )[:max_elites]
    elites[:] = cert + approx


def update_elites(elites, candidates, infos, step, max_elites):
    """Add newly discovered replay-eligible candidates; preserve all certified ones."""
    existing = {tuple(e['ids']) for e in elites}
    new = []
    for candidate, z in zip(candidates, infos):
        if not (z.elite_eligible or z.certified_ivp):
            continue
        key = tuple(int(i) for i in candidate[0])
        if key in existing:
            continue
        e = buffer_entry(candidate, z, step)
        elites.append(e); new.append(e); existing.add(key)

    trim_replay_buffer(elites, max_elites)
    keep = {tuple(e['ids']) for e in elites}
    return [e for e in new if tuple(e['ids']) in keep]


def warmup_collect(elites, candidates, infos, step, max_elites):
    """Frozen-search admission: keep global top-K valid candidates plus every certified one.

    Unlike online replay admission, warmup is allowed to retain a strong candidate even if it
    narrowly misses the loose elite thresholds. This makes the initial search genuinely
    best-of-N while certification remains the highest-priority tier.
    """
    existing = {tuple(e['ids']) for e in elites}
    added = []
    for candidate, z in zip(candidates, infos):
        if z.expression is None or z.fitted_expression is None or not np.isfinite(float(z.reward)):
            continue
        if z.error and str(z.error).startswith('TIMEOUT:'):
            continue
        key = tuple(int(i) for i in candidate[0])
        if key in existing:
            continue
        e = buffer_entry(candidate, z, step)
        elites.append(e); added.append(e); existing.add(key)
    trim_replay_buffer(elites, max_elites)
    keep = {tuple(e['ids']) for e in elites}
    return [e for e in added if tuple(e['ids']) in keep]


def replay_weights(elites, beta):
    if not elites:
        return np.empty((0,), dtype=np.float32)
    certified_mask = np.asarray([bool(e['certified']) for e in elites], dtype=bool)
    scores = np.asarray([float(e['reward']) for e in elites], dtype=np.float64)
    # Once a certified trajectory exists, replay only certified trajectories.
    if np.any(certified_mask):
        scores = scores.copy()
        scores[~certified_mask] = -np.inf
    finite = np.isfinite(scores)
    if not np.any(finite):
        return np.full(len(elites), 1.0 / len(elites), dtype=np.float32)
    z = np.full(len(elites), -np.inf, dtype=np.float64)
    m = float(np.max(scores[finite]))
    z[finite] = float(beta) * (scores[finite] - m)
    w = np.zeros(len(elites), dtype=np.float64)
    w[finite] = np.exp(np.clip(z[finite], -60.0, 0.0))
    w /= max(float(w.sum()), 1e-12)
    return w.astype(np.float32)


def replay(env, decoder, enc1, src_len_1, elites, optimizer, trainable, device, updates, weight, grad_clip, beta):
    if not elites or updates <= 0: return None
    losses = []
    weights_np = replay_weights(elites, beta)
    weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    decoder.train()
    for _ in range(updates):
        lp = token_sequence_logprobs(env, decoder, enc1.detach(), src_len_1, elite_ids(elites), device,
                                     length_normalize=False)
        loss = -float(weight) * (weights * lp).sum()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip); optimizer.step()
        losses.append(float(loss.item()))
    decoder.eval()
    return {
        'updates': updates, 'loss_first': losses[0], 'loss_last': losses[-1],
        'loss_mean': float(np.mean(losses)), 'weights': weights_np.tolist(),
    }


def candidate_row(idx, candidate, z):
    ids, words = candidate
    return {
        'idx': int(idx), 'reward': float(z.reward), 'certified': bool(z.certified_ivp),
        'elite_eligible': bool(z.elite_eligible), 'ode_rel': float(z.ode_rel_mse),
        'ref_nrmse': float(z.ref_nrmse), 'anchor_rmse': float(z.anchor_rmse),
        'expr': str(z.expression) if z.expression is not None else '<parse error>',
        'fitted_expr': str(z.fitted_expression) if z.fitted_expression is not None else '<invalid>',
        'coeffs': dict(z.fitted_coefficients), 'tokens': ' '.join(words),
        'error': z.error,
    }


def certified_rows(candidates, infos):
    return [candidate_row(i, c, z) for i, (c, z) in enumerate(zip(candidates, infos)) if z.certified_ivp]


def timeout_count(infos):
    return int(sum(bool(z.error and str(z.error).startswith('TIMEOUT:')) for z in infos))


def print_certified(rows, label):
    for r in rows:
        print(f"  !!! CERTIFIED [{label}] reward={r['reward']:.3f} ode={r['ode_rel']:.3e} ref={r['ref_nrmse']:.3e} anchor={r['anchor_rmse']:.3e}")
        print(f"      expr={r['expr']}")
        print(f"      fitted={r['fitted_expr']} coeffs={r['coeffs']}")
        print(f"      tokens={r['tokens']}")


def log_certified_events(fout, rows, source, step=None):
    for r in rows:
        event = {'event': 'certified_candidate', 'source': source, 'candidate': r}
        if step is not None:
            event['step'] = int(step)
        fout.write(json.dumps(event) + '\n')
    if rows:
        fout.flush()

def decode_batch(decoder, enc1, src_len_1, n, max_len, temp):
    with torch.no_grad():
        return decoder.generate(
            enc1.expand(n, -1, -1).contiguous(), src_len_1.expand(n).contiguous(),
            max_len=max_len, sample_temperature=temp,
        )


def greedy_eval(env, decoder, enc1, src_len_1, verifier, max_len):
    with torch.no_grad():
        gen, glen = decoder.generate(enc1, src_len_1, max_len=max_len, sample_temperature=None)
    c, z, _, _ = verifier.evaluate_generated(gen, glen, cache=None)
    return c[0], z[0]


def fresh_eval(decoder, enc1, src_len_1, verifier, n, max_len, temp, cache):
    if n <= 0: return None
    t0 = time.perf_counter(); gen, glen = decode_batch(decoder, enc1, src_len_1, n, max_len, temp); tg = time.perf_counter()-t0
    t0 = time.perf_counter(); cands, infos, hits, misses = verifier.evaluate_generated(gen, glen, cache); tv = time.perf_counter()-t0
    rows = summarize_ivp(cands, infos, top_k=1)
    cert_rows = certified_rows(cands, infos)
    return {
        'n': n,
        'certified': int(sum(z.certified_ivp for z in infos)),
        'certified_rate': float(np.mean([z.certified_ivp for z in infos])),
        'elite_eligible': int(sum(z.elite_eligible for z in infos)),
        'elite_rate': float(np.mean([z.elite_eligible for z in infos])),
        'best_reward': float(max(z.reward for z in infos)),
        'top': rows[0],
        'certified_candidates': cert_rows,
        'timeouts': timeout_count(infos),
        'generation_s': tg, 'verification_s': tv,
        'cache_hits': hits, 'cache_misses': misses,
    }


def frozen_warmup(decoder, enc1, src_len_1, verifier, env, n_samples, batch_size, max_len, temp, cache, elites, max_elites, fout):
    if n_samples <= 0:
        return {
            'n': 0, 'certified': 0, 'elite_eligible': 0, 'timeouts': 0,
            'best_reward': float('-inf'), 'generation_s': 0.0, 'verification_s': 0.0,
            'cache_hits': 0, 'cache_misses': 0, 'buffer_size': len(elites),
        }
    total_cert = total_elite = total_timeouts = total_hits = total_misses = 0
    total_gen = total_ver = 0.0
    best_reward = float('-inf')
    best_row = None
    seen = 0
    decoder.eval()
    print(f"\n[warmup frozen search] samples={n_samples} batch_size={batch_size} temperature={temp:g}")
    while seen < n_samples:
        bs = min(batch_size, n_samples - seen)
        t0 = time.perf_counter(); gen, glen = decode_batch(decoder, enc1, src_len_1, bs, max_len, temp); total_gen += time.perf_counter()-t0
        t0 = time.perf_counter(); cands, infos, hits, misses = verifier.evaluate_generated(gen, glen, cache); total_ver += time.perf_counter()-t0
        total_hits += hits; total_misses += misses
        total_cert += int(sum(z.certified_ivp for z in infos))
        total_elite += int(sum(z.elite_eligible for z in infos))
        total_timeouts += timeout_count(infos)
        certs = certified_rows(cands, infos)
        print_certified(certs, f'warmup {seen}-{seen+bs-1}')
        log_certified_events(fout, certs, 'warmup')
        warmup_collect(elites, cands, infos, step=-1, max_elites=max_elites)
        rows = summarize_ivp(cands, infos, top_k=1)
        if rows and float(rows[0]['reward']) > best_reward:
            best_reward = float(rows[0]['reward']); best_row = rows[0]
        seen += bs
        print(
            f"  {seen:>4}/{n_samples} certified={total_cert} elite={total_elite} "
            f"best={best_reward:.3f} buffer={len(elites)} timeouts={total_timeouts}"
        )
    summary = {
        'n': int(n_samples), 'certified': int(total_cert), 'elite_eligible': int(total_elite),
        'timeouts': int(total_timeouts), 'best_reward': float(best_reward), 'top': best_row,
        'generation_s': float(total_gen), 'verification_s': float(total_ver),
        'cache_hits': int(total_hits), 'cache_misses': int(total_misses),
        'buffer_size': int(len(elites)),
        'buffer': [{k:v for k,v in e.items() if k != 'ids'} for e in elites],
    }
    print(
        f"[warmup summary] certified={total_cert}/{n_samples} ({100*total_cert/max(n_samples,1):.2f}%) "
        f"elite={total_elite}/{n_samples} ({100*total_elite/max(n_samples,1):.2f}%) "
        f"best={best_reward:.3f} buffer={len(elites)} verify={total_ver:.1f}s"
    )
    print('  warmup replay buffer (best-ever; all certified retained):')
    for e in elites:
        print(
            f"      cert={int(e['certified'])} elite={int(e.get('elite_eligible', False))} "
            f"reward={e['reward']:.3f} ode={e['ode_rel']:.3e} ref={e['ref_nrmse']:.3e} :: {e['expr']}"
        )
    fout.write(json.dumps({'event': 'warmup_summary', 'warmup': summary}) + '\n'); fout.flush()
    return summary


def main():
    a = parse_args(); seed_all(a.seed)
    device = torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'device={device}')
    env, encoder, decoder, params, _ = load_pretrained(a.checkpoint, device)
    input_problem = make_problem(env, 'lane_emden', n=2, mode='general', lane_form=a.lane_form)
    src_tokens = equation_to_tokens(env, input_problem.equation)
    verifier = LaneEmdenN2Verifier(
        env,
        certify_ode_rel=a.certify_ode_rel, certify_ref_nrmse=a.certify_ref_nrmse,
        certify_anchor_rmse=a.certify_anchor_rmse,
        elite_ode_rel=a.elite_ode_rel, elite_ref_nrmse=a.elite_ref_nrmse,
        elite_anchor_rmse=a.elite_anchor_rmse, candidate_timeout_s=a.candidate_timeout,
    )
    print('problem=lane_emden_n2_ivp')
    print(f'input_form={a.lane_form}')
    print(f'equation={input_problem.equation}')
    print('input_prefix=' + ' '.join(src_tokens))
    print('IVP: y(0)=1, y\'(0)=0; coefficient fitting uses numerical reference at x=0.1')
    print(f'certify: ode<={verifier.certify_ode_rel:g} ref<={verifier.certify_ref_nrmse:g} anchor<={verifier.certify_anchor_rmse:g}')
    print(f'elite:   ode<={verifier.elite_ode_rel:g} ref<={verifier.elite_ref_nrmse:g} anchor<={verifier.elite_anchor_rmse:g}')
    print(f'candidate_timeout={verifier.candidate_timeout_s:g}s replay_beta={a.replay_beta:g} high_quality_reward>{a.high_quality_reward:g}')
    print(f'warmup_samples={a.warmup_samples} warmup_batch_size={a.warmup_batch_size} warmup_temperature={a.warmup_temperature:g}')

    for p in encoder.parameters(): p.requires_grad_(False)
    trainable = configure_trainable(decoder, a.scope)
    optimizer = torch.optim.Adam(trainable, lr=a.lr)
    print(f'trainable_parameters={sum(p.numel() for p in trainable):,} scope={a.scope}')

    x, src_len_1 = encode_tokens(env, src_tokens, device)
    with torch.no_grad(): enc1 = encoder('fwd', x=x, lengths=src_len_1, causal=False).transpose(0, 1)

    _, g0 = greedy_eval(env, decoder, enc1, src_len_1, verifier, a.max_len)
    print('\n[baseline greedy]')
    print(f'reward={g0.reward:.3f} certified={g0.certified_ivp} elite={g0.elite_eligible} ode={g0.ode_rel_mse:.3e} ref={g0.ref_nrmse:.3e}')
    print(f'expr={g0.expression}')
    print(f'fitted={g0.fitted_expression} coeffs={g0.fitted_coefficients}')

    # Keep adaptation/search cache completely separate from held-out evaluation cache.
    train_cache = {}; eval_cache = {}; elites = []
    fout = open(a.save_jsonl, 'w', encoding='utf-8')
    heldout_eval_calls = 0
    if a.eval_rollouts > 0:
        eval_rng = capture_rng_state()
        ev = fresh_eval(decoder, enc1, src_len_1, verifier, a.eval_rollouts, a.max_len, a.eval_temperature, eval_cache)
        restore_rng_state(eval_rng)
        heldout_eval_calls += int(a.eval_rollouts)
        print(f"\n[baseline fresh] certified={ev['certified']}/{ev['n']} ({100*ev['certified_rate']:.2f}%) elite={ev['elite_eligible']}/{ev['n']} best={ev['best_reward']:.3f} timeouts={ev['timeouts']}")
        print_certified(ev['certified_candidates'], 'baseline fresh — held out, NOT replayed')
        fout.write(json.dumps({'event': 'baseline_eval', 'eval': ev})+'\n'); fout.flush()
        log_certified_events(fout, ev['certified_candidates'], 'baseline_fresh')

    # Phase A: frozen exploration. No parameter update is allowed before this finishes.
    warm = frozen_warmup(
        decoder, enc1, src_len_1, verifier, env,
        a.warmup_samples, a.warmup_batch_size, a.max_len, a.warmup_temperature,
        train_cache, elites, a.max_elites, fout,
    )
    search_verifier_calls = int(a.warmup_samples)

    # First adaptation happens only after the full frozen warmup search.
    warm_before_lp = elite_logp_stats(env, decoder, enc1, src_len_1, elites, device)
    if any(e['certified'] for e in elites):
        warm_updates = a.certified_updates
        warm_tier = 'certified'
    elif any(e['reward'] > a.high_quality_reward for e in elites):
        warm_updates = a.high_quality_updates
        warm_tier = 'high_quality'
    elif any(e.get('elite_eligible', False) for e in elites):
        warm_updates = a.elite_updates
        warm_tier = 'elite'
    else:
        warm_updates = 0
        warm_tier = 'none'
    warm_replay = replay(
        env, decoder, enc1, src_len_1, elites, optimizer, trainable, device,
        warm_updates, a.elite_weight, a.grad_clip, a.replay_beta,
    )
    warm_after_lp = elite_logp_stats(env, decoder, enc1, src_len_1, elites, device)
    print(f"[warmup adaptation] tier={warm_tier} updates={warm_updates}")
    if warm_replay:
        print(f"  prioritized_replay loss={warm_replay['loss_first']:.3f}->{warm_replay['loss_last']:.3f}")
        if warm_before_lp and warm_after_lp:
            print(f"  buffer logP mean {warm_before_lp['mean']:.3f}->{warm_after_lp['mean']:.3f}")
        print('  replay weights:')
        for e, w in zip(elites, warm_replay['weights']):
            print(f"      w={w:.3f} cert={int(e['certified'])} reward={e['reward']:.3f} ref={e['ref_nrmse']:.3e} :: {e['expr']}")
    fout.write(json.dumps({
        'event':'warmup_adaptation','tier':warm_tier,'updates':warm_updates,
        'elite_logp_before':warm_before_lp,'elite_logp_after':warm_after_lp,
        'replay':warm_replay,'search_verifier_calls':search_verifier_calls,
    })+'\n'); fout.flush()

    for step in range(a.steps):
        tstep = time.perf_counter()
        decoder.eval(); t0=time.perf_counter(); gen, glen = decode_batch(decoder, enc1, src_len_1, a.rollouts, a.max_len, a.temperature); tgen=time.perf_counter()-t0
        t0=time.perf_counter(); cands, infos, hits, misses = verifier.evaluate_generated(gen, glen, train_cache); tver=time.perf_counter()-t0
        search_verifier_calls += int(a.rollouts)
        rewards_np = np.asarray([z.reward for z in infos], dtype=np.float32)
        cert_n = int(sum(z.certified_ivp for z in infos)); elite_n=int(sum(z.elite_eligible for z in infos))
        top = summarize_ivp(cands, infos, top_k=5)
        ntimeouts = timeout_count(infos)
        print(f"\n[step {step:02d}] reward mean={rewards_np.mean():.3f} std={rewards_np.std():.3f} max={rewards_np.max():.3f} certified={cert_n} elite={elite_n} timeouts={ntimeouts}")
        print_top(top)
        train_certified = certified_rows(cands, infos)
        print_certified(train_certified, f'train step {step}')
        log_certified_events(fout, train_certified, 'train', step)

        new = update_elites(elites, cands, infos, step, a.max_elites)
        before_lp = elite_logp_stats(env, decoder, enc1, src_len_1, elites, device)
        if new:
            print(f'  +++ new replay elites={len(new)} buffer={len(elites)} +++')
            for e in new:
                print(f"      certified={int(e['certified'])} reward={e['reward']:.3f} ref={e['ref_nrmse']:.3e} :: {e['expr']}")

        # group-relative RL
        t0=time.perf_counter(); rl_loss=None
        rewards = torch.tensor(rewards_np, device=device)
        if float(rewards.std(unbiased=False).item()) >= 1e-8:
            adv = (rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-6)
            decoder.train()
            seq_lp = sequence_logprobs(
                decoder, enc1.detach().expand(a.rollouts,-1,-1).contiguous(),
                src_len_1.expand(a.rollouts).contiguous(), gen, glen,
                length_normalize=a.length_normalize,
            )
            loss = -(adv.detach()*seq_lp).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,a.grad_clip); optimizer.step(); decoder.eval()
            rl_loss=float(loss.item()); print(f'  policy_loss={rl_loss:.6f}')
        trl=time.perf_counter()-t0

        # Quality-prioritized replay: certified > high-quality > ordinary elite.
        t0=time.perf_counter()
        if any(e['certified'] for e in new):
            nup=a.certified_updates
        elif any(e['reward'] > a.high_quality_reward for e in new):
            nup=a.high_quality_updates
        elif new:
            nup=a.elite_updates
        else:
            nup=a.elite_replay_updates
        replay_stats = replay(
            env, decoder, enc1, src_len_1, elites, optimizer, trainable, device,
            nup, a.elite_weight, a.grad_clip, a.replay_beta
        )
        treplay=time.perf_counter()-t0
        after_lp = elite_logp_stats(env, decoder, enc1, src_len_1, elites, device)
        if replay_stats:
            print(f"  prioritized_replay updates={replay_stats['updates']} loss={replay_stats['loss_first']:.3f}->{replay_stats['loss_last']:.3f}")
            if before_lp and after_lp: print(f"  elite logP mean {before_lp['mean']:.3f}->{after_lp['mean']:.3f}")
            print('  replay buffer:')
            for e, w in zip(elites, replay_stats['weights']):
                print(f"      w={w:.3f} cert={int(e['certified'])} reward={e['reward']:.3f} ref={e['ref_nrmse']:.3e} step={e['step']} :: {e['expr']}")

        _, gz = greedy_eval(env, decoder, enc1, src_len_1, verifier, a.max_len)
        print(f"  greedy reward={gz.reward:.3f} certified={int(gz.certified_ivp)} elite={int(gz.elite_eligible)} ode={gz.ode_rel_mse:.2e} ref={gz.ref_nrmse:.2e} :: {gz.expression}")

        ev=None
        if a.eval_rollouts>0 and a.eval_every>0 and (((step+1)%a.eval_every)==0):
            eval_rng = capture_rng_state()
            ev=fresh_eval(decoder,enc1,src_len_1,verifier,a.eval_rollouts,a.max_len,a.eval_temperature,eval_cache)
            restore_rng_state(eval_rng)
            heldout_eval_calls += int(a.eval_rollouts)
            print(f"  [fresh] certified={ev['certified']}/{ev['n']} ({100*ev['certified_rate']:.2f}%) elite={ev['elite_eligible']}/{ev['n']} ({100*ev['elite_rate']:.2f}%) best={ev['best_reward']:.3f} timeouts={ev['timeouts']} verify={ev['verification_s']:.1f}s")
            print_certified(ev['certified_candidates'], f'fresh step {step} — held out, NOT replayed')
            log_certified_events(fout, ev['certified_candidates'], 'fresh_eval', step)

        total=time.perf_counter()-tstep
        print(f'  timing generate={tgen:.2f}s verify={tver:.2f}s rl={trl:.2f}s replay={treplay:.2f}s total={total:.2f}s cache={hits}/{hits+misses} search_calls={search_verifier_calls}')
        row={
            'step':step,'reward_mean':float(rewards_np.mean()),'reward_std':float(rewards_np.std()),'reward_max':float(rewards_np.max()),
            'n_certified':cert_n,'n_elite_eligible':elite_n,'top':top,'elite_buffer_size':len(elites),
            'new_elites':[{k:v for k,v in e.items() if k!='ids'} for e in new],
            'elite_logp_before':before_lp,'elite_logp_after':after_lp,'elite_train':replay_stats,'rl_loss':rl_loss,
            'greedy':{'reward':gz.reward,'certified':gz.certified_ivp,'elite_eligible':gz.elite_eligible,'ode_rel':gz.ode_rel_mse,'ref_nrmse':gz.ref_nrmse,'expr':str(gz.expression),'fitted_expr':str(gz.fitted_expression)},
            'eval':ev,'timing':{'generation_s':tgen,'verification_s':tver,'rl_s':trl,'replay_s':treplay,'total_s':total},
            'cache':{'hits':hits,'misses':misses,'size':len(train_cache)}, 'timeouts':ntimeouts,
            'budgets':{'search_verifier_calls':search_verifier_calls,'heldout_eval_calls':heldout_eval_calls},
            'replay_buffer':[{k:v for k,v in e.items() if k!='ids'} for e in elites],
        }
        fout.write(json.dumps(row)+'\n'); fout.flush()

    fout.close()
    _, gf = greedy_eval(env,decoder,enc1,src_len_1,verifier,a.max_len)
    print('\n[final greedy]')
    print(f'reward={gf.reward:.3f} certified={gf.certified_ivp} elite={gf.elite_eligible} ode={gf.ode_rel_mse:.3e} ref={gf.ref_nrmse:.3e}')
    print(f'expr={gf.expression}')
    print(f'fitted={gf.fitted_expression} coeffs={gf.fitted_coefficients}')
    if a.save_decoder:
        torch.save(decoder.state_dict(), a.save_decoder); print(f'saved_decoder={a.save_decoder}')
    print(f'budgets: search_verifier_calls={search_verifier_calls} heldout_eval_calls={heldout_eval_calls} diagnostic_greedy_calls={a.steps + 2}')
    print(f'result_log={a.save_jsonl}')


if __name__=='__main__':
    main()
