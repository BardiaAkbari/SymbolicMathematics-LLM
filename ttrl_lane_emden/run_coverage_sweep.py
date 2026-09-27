#!/usr/bin/env python3
"""Frozen-model support diagnostic before doing any more TTRL."""
import argparse, os, random, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import load_pretrained, make_problem, sample_rollouts, evaluate_rollout_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--samples-per-setting', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--temperatures', default='0.3,0.5,0.7,1.0')
    p.add_argument('--max-len', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device = torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    env, enc, dec, _, _ = load_pretrained(a.checkpoint, device)
    temps = [float(z) for z in a.temperatures.split(',')]

    for form in ['standard', 'cleared']:
        problem = make_problem(env, 'lane_emden', n=1, mode='general', lane_form=form)
        print(f'\n=== form={form} equation={problem.equation} ===')
        for temp in temps:
            seen = successes = 0
            best = None
            while seen < a.samples_per_setting:
                k = min(a.batch_size, a.samples_per_setting-seen)
                _, generated, glen = sample_rollouts(env, enc, dec, problem.equation, k, temp, a.max_len, device)
                _, infos = evaluate_rollout_batch(env, problem, generated, glen)
                seen += k
                successes += sum(int(z.exact_residual_zero and z.verified_general) for z in infos)
                b = max(infos, key=lambda z: z.reward)
                if best is None or b.reward > best.reward:
                    best = b
            print(f'temp={temp:<4} success={successes:>3}/{seen} best_reward={best.reward:7.3f} '
                  f'relres={best.numerical_mse:.3e} res0={int(best.exact_residual_zero)} '
                  f'expr={best.expression}')

if __name__ == '__main__':
    main()
