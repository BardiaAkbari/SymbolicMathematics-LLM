#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0,ROOT)

from ttrl_lane_emden.core import load_pretrained, summarize_infos
from ttrl_ode_benchmark.common import read_jsonl, seed_all, make_problem_from_record, encode_problem, decode_batch, decode_greedy
from ttrl_ode_benchmark.exact_verifier import (
    ExactODEVerifier, ParallelVerifierPool, resolve_verifier_workers, exact_count, timeout_count,
)


def args():
    p=argparse.ArgumentParser(description='Frozen coverage scan on verified HSE ODE2 problems')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--problems',default='hse300k_ode2_verified.jsonl')
    p.add_argument('--output',default='hse300k_ode2_coverage.jsonl')
    p.add_argument('--rare-output',default='hse300k_ode2_rare_support.jsonl')
    p.add_argument('--max-problems',type=int,default=100)
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--temperature',type=float,default=1.0)
    p.add_argument('--max-len',type=int,default=128)
    p.add_argument('--candidate-timeout',type=float,default=4.0)
    p.add_argument('--verifier-workers',type=int,default=0,
                   help='CPU verifier processes; 0=auto up to 32, 1=serial')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--cpu',action='store_true')
    return p.parse_args()


def main():
    a=args(); seed_all(a.seed)
    device=torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'device={device}')
    env,encoder,decoder,params,ckpt=load_pretrained(a.checkpoint,device)
    nworkers=resolve_verifier_workers(a.verifier_workers)
    verifier_pool=ParallelVerifierPool(nworkers)
    print(f'verifier_workers={nworkers} start_method=spawn')
    rows=list(read_jsonl(a.problems))
    if a.max_problems>0: rows=rows[:a.max_problems]
    out=open(a.output,'w',encoding='utf-8'); rare=open(a.rare_output,'w',encoding='utf-8')
    counts={'easy_greedy':0,'rare_support':0,'zero_support':0,'parse_error':0}

    for pi,rec in enumerate(rows):
        t0=time.perf_counter()
        try:
            problem=make_problem_from_record(env,rec)
            prefix,enc1,src_len=encode_problem(env,encoder,problem,device)
        except BaseException as e:
            z={**rec,'coverage_class':'parse_error','coverage_error':f'{type(e).__name__}: {e}'}
            out.write(json.dumps(z,ensure_ascii=False)+'\n'); out.flush(); counts['parse_error']+=1
            print(f'[{pi:03d}] PARSE ERROR {rec.get("id")}: {e}')
            continue
        verifier=ExactODEVerifier(env,problem,a.candidate_timeout,worker_pool=verifier_pool)
        cache={}
        gg,gl=decode_greedy(decoder,enc1,src_len,a.max_len)
        gc,gi,_,_=verifier.evaluate_generated(gg,gl,cache)
        greedy_exact=bool(gi[0].verified_general)
        total_exact=total_timeout=0; seen=0; first_exact=None; best_reward=-1e30; best_expr=None
        while seen<a.samples:
            bs=min(a.batch_size,a.samples-seen)
            gen,glen=decode_batch(decoder,enc1,src_len,bs,a.max_len,a.temperature)
            cands,infos,hits,misses=verifier.evaluate_generated(gen,glen,cache)
            total_exact += exact_count(infos); total_timeout += timeout_count(infos)
            for cand,z in zip(cands,infos):
                if z.reward>best_reward:
                    best_reward=float(z.reward); best_expr=str(z.expression)
                if first_exact is None and z.verified_general:
                    first_exact={'expr':str(z.expression),'tokens':' '.join(cand[1]),'reward':float(z.reward)}
            seen+=bs
        if greedy_exact: cls='easy_greedy'
        elif total_exact>0: cls='rare_support'
        else: cls='zero_support'
        counts[cls]+=1
        result={**rec,
            'coverage_class':cls,'greedy_exact':greedy_exact,'greedy_expr':str(gi[0].expression),
            'sample_exact':int(total_exact),'samples':int(a.samples),'sample_exact_rate':float(total_exact/max(a.samples,1)),
            'temperature':float(a.temperature),'best_reward':float(best_reward),'best_expr':best_expr,
            'first_exact':first_exact,'timeouts':int(total_timeout),'prefix_checkpoint':' '.join(prefix),
            'coverage_seconds':float(time.perf_counter()-t0),
        }
        out.write(json.dumps(result,ensure_ascii=False)+'\n'); out.flush()
        if cls=='rare_support':
            # Build a physically label-free adaptation input. Do not save either the dataset
            # reference answer or the exact trajectory discovered by coverage; TTRL must
            # rediscover correctness using only the ODE verifier under its own counted budget.
            clean=dict(result); clean['first_exact']=None
            for k in ('answer_latex','reference_sympy','reference_verified','reference_rank'):
                clean.pop(k,None)
            rare.write(json.dumps(clean,ensure_ascii=False)+'\n'); rare.flush()
        print(f'[{pi:03d}] {cls:12s} exact={total_exact}/{a.samples} greedy={int(greedy_exact)} best={best_reward:.2f} timeout={total_timeout} :: {rec.get("id")}')
    out.close(); rare.close(); verifier_pool.close()
    print('\n=== coverage summary ===')
    print(json.dumps(counts,indent=2))
    print(f'output={a.output}')
    print(f'rare_support_output={a.rare_output}')

if __name__=='__main__': main()
