#!/usr/bin/env python3
from __future__ import annotations

import argparse,json,os,sys
import torch

ROOT=os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:sys.path.insert(0,ROOT)

from ttrl_lane_emden.core import load_pretrained,generated_to_candidates
from ttrl_ode_benchmark.common import read_jsonl,seed_all,make_problem_from_record,encode_problem,decode_batch
from ttrl_ode_benchmark.exact_verifier import ParallelVerifierPool,resolve_verifier_workers
from ttrl_ode_benchmark.rich_reward_v3 import ParallelRewardV3Pool
from ttrl_ode_benchmark.rich_reward_v31 import ParallelRewardV31Pool

def parse_args():
    p=argparse.ArgumentParser(description='Compare frozen Reward-v3 vs Reward-v3.1 on the SAME candidates for 50 failures')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--problems',default='validation50_balanced_blind.jsonl')
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--temperature',type=float,default=1.0)
    p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--max-len',type=int,default=128)
    p.add_argument('--candidate-timeout',type=float,default=4.0)
    p.add_argument('--reward-timeout',type=float,default=6.0)
    p.add_argument('--verifier-workers',type=int,default=32)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--probe-seed',type=int,default=314159)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--output',default='validation50_reward_v31_ranking.jsonl')
    return p.parse_args()

def key(ids):return tuple(int(x) for x in ids)

def main():
    a=parse_args();seed_all(a.seed)
    device=torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'device={device}')
    env,encoder,decoder,params,ckpt=load_pretrained(a.checkpoint,device);decoder.eval()
    nw=resolve_verifier_workers(a.verifier_workers);print(f'verifier_workers={nw}')
    exact_pool=ParallelVerifierPool(nw);v3_pool=ParallelRewardV3Pool(nw);v31_pool=ParallelRewardV31Pool(nw)
    out=open(a.output,'w',encoding='utf-8')
    try:
        for pi,rec in enumerate(read_jsonl(a.problems)):
            sample_seed=a.seed+1009*pi;seed_all(sample_seed)
            problem=make_problem_from_record(env,rec)
            prefix,enc1,src_len=encode_problem(env,encoder,problem,device)
            uniq={};seen=0
            while seen<a.samples:
                bs=min(a.batch_size,a.samples-seen)
                gen,glen=decode_batch(decoder,enc1,src_len,bs,a.max_len,a.temperature)
                for ids,toks in generated_to_candidates(env,gen,glen):uniq.setdefault(key(ids),(list(map(int,ids)),toks))
                seen+=bs
            ids_list=[v[0] for v in uniq.values()];tok_list=[v[1] for v in uniq.values()]
            exact_infos=exact_pool.score_many(env,problem,a.candidate_timeout,ids_list)
            v3_infos=v3_pool.score_many(env,problem,a.reward_timeout,ids_list)
            pseed=a.probe_seed+10007*pi
            v31_infos=v31_pool.score_many(env,problem,a.reward_timeout,ids_list,probe_seed=pseed)
            rows=[]
            for ids,toks,old,v3,v31 in zip(ids_list,tok_list,exact_infos,v3_infos,v31_infos):
                rows.append({
                    'expr':str(old.expression) if old.expression is not None else (v31.expression or v3.expression),
                    'tokens':' '.join(toks),'token_len':len(ids),'exact':bool(old.verified_general),
                    'old_reward':float(old.reward),'rank':int(old.generality_rank),
                    'v3_reward':float(v3.pre_reward),'v31_reward':float(v31.pre_reward),
                    'v3_eq':float(v3.equation_score),'v3_dmin':float(v3.direction_min_score),'v3_G':float(v3.independence_score),
                    'v31_eq':float(v31.equation_score),'v31_eq_domains':list(v31.equation_domain_scores),
                    'v31_dmin':float(v31.direction_min_score),'v31_directions':list(v31.direction_scores),
                    'v31_direction_domains':list(v31.direction_domain_scores),
                    'v31_base':float(v31.base_score),'v31_base_domains':list(v31.base_domain_scores),
                    'v31_G_diag':float(v31.independence_score),'v31_gate':bool(v31.generality_gate_pass),
                    'v31_wronskian_zero':bool(v31.wronskian_zero),'v31_wronskian':v31.wronskian_expression,
                    'v31_direction_penalty':float(v31.direction_penalty),
                    'v31_hard_generality_penalty':float(v31.hard_generality_penalty),
                    'v31_invalid_penalty':float(v31.invalid_penalty),'v31_validity':float(v31.validity),
                    'v31_error':v31.error,
                })
            v3_top=sorted(rows,key=lambda r:(r['v3_reward'],r['exact']),reverse=True)[:a.top_k]
            v31_top=sorted(rows,key=lambda r:(r['v31_reward'],r['exact']),reverse=True)[:a.top_k]
            overlap=len({r['tokens'] for r in v3_top}&{r['tokens'] for r in v31_top})
            record={'id':rec.get('id'),'equation_latex':rec.get('equation_latex'),'failure_family':rec.get('failure_family'),
                    'samples':a.samples,'unique_candidates':len(rows),'temperature':a.temperature,
                    'sample_seed':sample_seed,'probe_seed':pseed,'exact_discovered':int(sum(r['exact'] for r in rows)),
                    'top_k':a.top_k,'top_overlap_v3_v31':overlap,'v3_top':v3_top,'v31_top':v31_top}
            out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush()
            print('\n'+'='*126)
            print(f"[{pi}] {rec.get('id')} family={rec.get('failure_family')} unique={len(rows)}/{a.samples} exact={record['exact_discovered']} overlap(v3,v31)={overlap}/{a.top_k}")
            print(f"ODE: {rec.get('equation_latex')}")
            print('\nV3 TOP:')
            for j,r in enumerate(v3_top,1):
                print(f" {j:2d}. v3={r['v3_reward']:8.3f} v31={r['v31_reward']:8.3f} rank={r['rank']} G3={r['v3_G']:.4g} gate31={int(r['v31_gate'])} :: {r['expr']}")
            print('\nV3.1 TOP:')
            for j,r in enumerate(v31_top,1):
                dom=r['v31_eq_domains'] if r['v31_eq_domains'] else (r['v31_direction_domains'][0] if r['v31_direction_domains'] else [])
                print(f" {j:2d}. v31={r['v31_reward']:8.3f} v3={r['v3_reward']:8.3f} rank={r['rank']} eq31={r['v31_eq']:6.2f} dmin31={r['v31_dmin']:6.2f} base31={r['v31_base']:6.2f} gate={int(r['v31_gate'])} W0={int(r['v31_wronskian_zero'])} dom={','.join(f'{z:.2f}' for z in dom)} :: {r['expr']}")
    finally:
        out.close();exact_pool.close();v3_pool.close();v31_pool.close()
    print(f'\nresults={a.output}')

if __name__=='__main__':main()
