#!/usr/bin/env python3
"""Frozen-policy coverage sweep for Lane--Emden n=2 IVP."""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import encode_tokens, equation_to_tokens, load_pretrained, make_problem
from ttrl_lane_emden.ivp_n2 import LaneEmdenN2Verifier, summarize_ivp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--lane-form', choices=['standard', 'cleared'], default='cleared')
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-len', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--certify-ode-rel', type=float, default=5e-2)
    p.add_argument('--certify-ref-nrmse', type=float, default=3e-2)
    p.add_argument('--elite-ode-rel', type=float, default=1.2e-1)
    p.add_argument('--elite-ref-nrmse', type=float, default=1.2e-1)
    return p.parse_args()


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device = torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'device={device}')
    env, enc, dec, params, _ = load_pretrained(a.checkpoint, device)
    problem = make_problem(env, 'lane_emden', n=2, mode='general', lane_form=a.lane_form)
    src_tokens = equation_to_tokens(env, problem.equation)
    print('problem=lane_emden_n2_ivp')
    print(f'input_form={a.lane_form}')
    print(f'equation={problem.equation}')
    print('input_prefix=' + ' '.join(src_tokens))
    verifier = LaneEmdenN2Verifier(
        env,
        certify_ode_rel=a.certify_ode_rel,
        certify_ref_nrmse=a.certify_ref_nrmse,
        elite_ode_rel=a.elite_ode_rel,
        elite_ref_nrmse=a.elite_ref_nrmse,
    )
    print(f'anchor x={verifier.x_anchor:.3f} target_y={verifier.anchor_target[0]:.8f} target_dy={verifier.anchor_target[1]:.8f}')
    print(f'certify: ode_rel<={verifier.certify_ode_rel:g} ref_nrmse<={verifier.certify_ref_nrmse:g} anchor<={verifier.certify_anchor_rmse:g}')
    print(f'elite:   ode_rel<={verifier.elite_ode_rel:g} ref_nrmse<={verifier.elite_ref_nrmse:g} anchor<={verifier.elite_anchor_rmse:g}')

    x, slen = encode_tokens(env, src_tokens, device)
    with torch.no_grad():
        enc1 = enc('fwd', x=x, lengths=slen, causal=False).transpose(0, 1)

    cache = {}
    all_rows = []
    certified = elite = 0
    seen = 0
    t0 = time.perf_counter()
    while seen < a.samples:
        bs = min(a.batch_size, a.samples - seen)
        tg = time.perf_counter()
        with torch.no_grad():
            gen, glen = dec.generate(
                enc1.expand(bs, -1, -1).contiguous(),
                slen.expand(bs).contiguous(),
                max_len=a.max_len,
                sample_temperature=a.temperature,
            )
        gen_s = time.perf_counter() - tg
        tv = time.perf_counter()
        cands, infos, hits, misses = verifier.evaluate_generated(gen, glen, cache)
        ver_s = time.perf_counter() - tv
        certified += sum(z.certified_ivp for z in infos)
        elite += sum(z.elite_eligible for z in infos)
        all_rows.extend(summarize_ivp(cands, infos, top_k=min(8, len(infos))))
        seen += bs
        best = max(infos, key=lambda z: z.reward)
        print(
            f'{seen:4d}/{a.samples} certified={certified} replay_elite={elite} '
            f'best_batch={best.reward:.3f} ode={best.ode_rel_mse:.3e} ref={best.ref_nrmse:.3e} '
            f'gen={gen_s:.2f}s verify={ver_s:.2f}s cache={hits}/{hits+misses}'
        )

    all_rows.sort(key=lambda r: r['reward'], reverse=True)
    print('\n[top frozen candidates]')
    for r in all_rows[:10]:
        print(
            f"reward={r['reward']:7.3f} cert={int(r['certified'])} elite={int(r['elite_eligible'])} "
            f"ode={r['ode_rel']:.3e} ref={r['ref_nrmse']:.3e} anchor={r['anchor_rmse']:.3e} :: {r['expr']}"
        )
        print(f"  fitted={r['fitted_expr']} coeffs={r['coeffs']}")
    print('\n[summary]')
    print(f'certified={certified}/{a.samples} ({100*certified/a.samples:.3f}%)')
    print(f'replay_elite={elite}/{a.samples} ({100*elite/a.samples:.3f}%)')
    print(f'total_s={time.perf_counter()-t0:.2f}')


if __name__ == '__main__':
    main()
