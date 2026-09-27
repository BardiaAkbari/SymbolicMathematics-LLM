#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, sys
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import load_pretrained, generated_to_candidates
from ttrl_ode_benchmark.common import read_jsonl, seed_all, make_problem_from_record, encode_problem, decode_batch
from ttrl_ode_benchmark.exact_verifier import ExactODEVerifier, ParallelVerifierPool, resolve_verifier_workers
from ttrl_ode_benchmark.rich_reward_v3 import ParallelRewardV3Pool


def parse_args():
    p=argparse.ArgumentParser(description='Compare old verifier shaping vs Reward-v3 on fixed HSE Hard-10')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--problems', default='hse10_failures_blind.jsonl')
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--max-len', type=int, default=128)
    p.add_argument('--candidate-timeout', type=float, default=4.0)
    p.add_argument('--verifier-workers', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--output', default='hard10_reward_v3_ranking.jsonl')
    return p.parse_args()


def key(ids):
    return tuple(int(x) for x in ids)


def main():
    a=parse_args(); seed_all(a.seed)
    device=torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'device={device}')
    env,encoder,decoder,params,ckpt=load_pretrained(a.checkpoint,device)
    decoder.eval()
    nworkers=resolve_verifier_workers(a.verifier_workers)
    print(f'verifier_workers={nworkers}')
    old_pool=ParallelVerifierPool(nworkers)
    v3_pool=ParallelRewardV3Pool(nworkers)
    out=open(a.output,'w',encoding='utf-8')

    try:
        for pi,rec in enumerate(read_jsonl(a.problems)):
            seed_all(a.seed+1009*pi)
            problem=make_problem_from_record(env,rec)
            prefix,enc1,src_len=encode_problem(env,encoder,problem,device)
            old_verifier=ExactODEVerifier(env,problem,a.candidate_timeout,worker_pool=old_pool)

            uniq={}
            seen=0
            while seen<a.samples:
                bs=min(a.batch_size,a.samples-seen)
                gen,glen=decode_batch(decoder,enc1,src_len,bs,a.max_len,a.temperature)
                cands=generated_to_candidates(env,gen,glen)
                for ids,toks in cands:
                    uniq.setdefault(key(ids),(list(map(int,ids)),toks))
                seen+=bs

            ids_list=[v[0] for v in uniq.values()]
            tok_list=[v[1] for v in uniq.values()]
            old_infos=old_pool.score_many(env,problem,a.candidate_timeout,ids_list)
            v3_infos=v3_pool.score_many(env,problem,a.candidate_timeout,ids_list)

            rows=[]
            for ids,toks,old,new in zip(ids_list,tok_list,old_infos,v3_infos):
                rows.append({
                    'expr': str(old.expression) if old.expression is not None else new.expression,
                    'tokens': ' '.join(toks),
                    'token_len': len(ids),
                    'exact': bool(old.verified_general),
                    'old_reward': float(old.reward),
                    'old_mse': float(old.numerical_mse),
                    'rank': int(old.generality_rank),
                    'v3_reward': float(new.pre_reward),
                    'q_equation': float(new.equation_score),
                    'q_direction_min': float(new.direction_min_score),
                    'q_direction_each': list(new.direction_scores),
                    'independence': float(new.independence_score),
                    'validity': float(new.validity),
                    'direction_penalty': float(new.direction_penalty),
                    'independence_penalty': float(new.independence_penalty),
                    'missing_penalty': float(new.missing_constant_penalty),
                    'invalid_penalty': float(new.invalid_penalty),
                    'v3_error': new.error,
                })

            old_top=sorted(rows,key=lambda r:(r['old_reward'],r['exact']),reverse=True)[:a.top_k]
            v3_top=sorted(rows,key=lambda r:(r['v3_reward'],r['exact']),reverse=True)[:a.top_k]
            old_ids={r['tokens'] for r in old_top}; v3_ids={r['tokens'] for r in v3_top}
            overlap=len(old_ids & v3_ids)
            record={
                'id':rec.get('id'),'equation_latex':rec.get('equation_latex'),
                'failure_family':rec.get('failure_family'),'samples':a.samples,
                'unique_candidates':len(rows),'temperature':a.temperature,
                'exact_discovered':int(sum(r['exact'] for r in rows)),
                'top_k':a.top_k,'top_overlap_old_v3':overlap,
                'old_top':old_top,'v3_top':v3_top,
            }
            out.write(json.dumps(record,ensure_ascii=False)+'\n'); out.flush()

            print('\n'+'='*118)
            print(f"[{pi}] {rec.get('id')} family={rec.get('failure_family')} unique={len(rows)}/{a.samples} exact={record['exact_discovered']} overlap(old,v3)={overlap}/{a.top_k}")
            print(f"ODE: {rec.get('equation_latex')}")
            print('\nOLD TOP:')
            for j,r in enumerate(old_top,1):
                print(f" {j:2d}. old={r['old_reward']:7.3f} v3={r['v3_reward']:8.3f} eq={r['q_equation']:6.2f} dmin={r['q_direction_min']:6.2f} G={r['independence']:.4f} Pd={r['direction_penalty']:5.2f} Pg={r['independence_penalty']:5.2f} exact={int(r['exact'])} :: {r['expr']}")
            print('\nV3 TOP:')
            for j,r in enumerate(v3_top,1):
                print(f" {j:2d}. v3={r['v3_reward']:8.3f} old={r['old_reward']:7.3f} eq={r['q_equation']:6.2f} dmin={r['q_direction_min']:6.2f} G={r['independence']:.4f} Pd={r['direction_penalty']:5.2f} Pg={r['independence_penalty']:5.2f} exact={int(r['exact'])} :: {r['expr']}")
    finally:
        out.close(); old_pool.close(); v3_pool.close()
    print(f'\nresults={a.output}')

if __name__=='__main__':
    main()
