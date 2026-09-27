#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import (
    evaluate_rollout_batch,
    load_pretrained,
    make_problem,
    sample_rollouts,
    summarize_infos,
)


def main():
    p = argparse.ArgumentParser(description='Compute-matched frozen-policy search baseline')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--problem', choices=['harmonic','lane_emden'], default='lane_emden')
    p.add_argument('--n', type=int, default=1)
    p.add_argument('--lane-form', choices=['standard','cleared'], default='cleared')
    p.add_argument('--samples', type=int, default=640)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-len', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--output', default='search_baseline.json')
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device = torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    env, encoder, decoder, params, ckpt = load_pretrained(a.checkpoint, device)
    problem = make_problem(env, a.problem, n=a.n, mode='general', lane_form=a.lane_form)

    best = None
    best_expr = None
    successes = 0
    seen = 0
    all_top = []

    while seen < a.samples:
        k = min(a.batch_size, a.samples - seen)
        _, generated, glen = sample_rollouts(
            env, encoder, decoder, problem.equation,
            n_samples=k, temperature=a.temperature, max_len=a.max_len, device=device,
        )
        candidates, infos = evaluate_rollout_batch(env, problem, generated, glen)
        for cand, info in zip(candidates, infos):
            seen += 1
            if info.verified_general and info.exact_residual_zero:
                successes += 1
            if best is None or info.reward > best.reward:
                best = info
                best_expr = str(info.expression)
        print(f'{seen}/{a.samples} best_reward={best.reward:.3f} successes={successes} best={best_expr}')

    result = {
        'problem': problem.name,
        'samples': a.samples,
        'temperature': a.temperature,
        'lane_form': a.lane_form,
        'successes': successes,
        'pass_at_n': bool(successes > 0),
        'best_reward': best.reward,
        'best_expr': best_expr,
        'best_exact_residual_zero': best.exact_residual_zero,
        'best_verified_general': best.verified_general,
        'best_rank': best.generality_rank,
    }
    with open(a.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
