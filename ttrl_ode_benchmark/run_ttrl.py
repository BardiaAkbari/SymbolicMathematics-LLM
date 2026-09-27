#!/usr/bin/env python3
"""Search-then-adapt TTRL on HSE ODE2 exact-general-solution problems.

Reference answers are never used here. Correctness comes only from substituting the
model's candidate back into the ODE plus a two-constant generality check.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
import numpy as np
import torch

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0,ROOT)

from ttrl_lane_emden.core import (
    configure_trainable, load_pretrained, sequence_logprobs, token_sequence_logprobs,
)
from ttrl_ode_benchmark.common import (
    read_jsonl, seed_all, capture_rng_state, restore_rng_state,
    make_problem_from_record, encode_problem, decode_batch, decode_greedy,
)
from ttrl_ode_benchmark.exact_verifier import (
    ExactODEVerifier, ParallelVerifierPool, resolve_verifier_workers, exact_count, timeout_count,
)


def parse_args():
    p=argparse.ArgumentParser(description='Verifier-guided search-then-adapt TTRL on HSE ODE2')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--problems',default='hse300k_ode2_rare_support.jsonl')
    p.add_argument('--max-problems',type=int,default=10)
    p.add_argument('--output',default='hse300k_ttrl_results.jsonl')
    p.add_argument('--summary',default='hse300k_ttrl_summary.json')
    p.add_argument('--warmup-samples',type=int,default=512)
    p.add_argument('--warmup-batch-size',type=int,default=64)
    p.add_argument('--warmup-temperature',type=float,default=1.0)
    p.add_argument('--steps',type=int,default=15)
    p.add_argument('--rollouts',type=int,default=64)
    p.add_argument('--temperature',type=float,default=1.0)
    p.add_argument('--eval-rollouts',type=int,default=256)
    p.add_argument('--eval-every',type=int,default=5)
    p.add_argument('--eval-temperature',type=float,default=1.0)
    p.add_argument('--max-len',type=int,default=128)
    p.add_argument('--scope',choices=['proj','last_layer','decoder'],default='last_layer')
    p.add_argument('--lr',type=float,default=3e-6)
    p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--length-normalize',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--replay-updates-new-exact',type=int,default=20)
    p.add_argument('--replay-updates',type=int,default=2)
    p.add_argument('--replay-weight',type=float,default=1.0)
    p.add_argument('--max-exact-buffer',type=int,default=32)
    p.add_argument('--max-approx-buffer',type=int,default=8)
    p.add_argument('--candidate-timeout',type=float,default=4.0)
    p.add_argument('--verifier-workers',type=int,default=0,
                   help='CPU verifier processes; 0=auto up to 32, 1=serial')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--cpu',action='store_true')
    return p.parse_args()


def candidate_entry(cand,z,step):
    return {'ids':list(map(int,cand[0])),'tokens':' '.join(cand[1]),'expr':str(z.expression),
            'reward':float(z.reward),'exact':bool(z.verified_general),'mse':float(z.numerical_mse),
            'rank':int(z.generality_rank),'step':int(step),'length':len(cand[0])}


def trim_buffer(buf,max_exact,max_approx):
    # Exact candidates all have the same correctness status; shorter programs are useful replay targets.
    exact=sorted([e for e in buf if e['exact']],key=lambda e:(e['length'],-e['reward'],e['step']))[:max_exact]
    approx=sorted([e for e in buf if not e['exact']],key=lambda e:(-e['reward'],e['length'],e['step']))[:max_approx]
    buf[:]=exact+approx


def update_buffer(buf,cands,infos,step,max_exact,max_approx):
    existing={tuple(e['ids']) for e in buf}; new_exact=[]
    for c,z in zip(cands,infos):
        if not z.valid_parse or not np.isfinite(float(z.reward)): continue
        key=tuple(map(int,c[0]))
        if key in existing: continue
        e=candidate_entry(c,z,step); buf.append(e); existing.add(key)
        if e['exact']: new_exact.append(e)
    trim_buffer(buf,max_exact,max_approx)
    keep={tuple(e['ids']) for e in buf}
    return [e for e in new_exact if tuple(e['ids']) in keep]


def replay(env,decoder,enc1,src_len,buf,optimizer,trainable,device,updates,weight,clip):
    if not buf or updates<=0: return None
    exact=[e for e in buf if e['exact']]
    selected=exact if exact else buf[:1]  # before exact discovery, do not imitate broad mediocre modes
    seqs=[e['ids'] for e in selected]
    losses=[]; decoder.train()
    for _ in range(updates):
        lp=token_sequence_logprobs(env,decoder,enc1.detach(),src_len,seqs,device,length_normalize=False)
        loss=-float(weight)*lp.mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,clip); optimizer.step()
        losses.append(float(loss.item()))
    decoder.eval()
    return {'updates':updates,'n_replayed':len(selected),'loss_first':losses[0],'loss_last':losses[-1]}


def eval_policy(decoder,enc1,src_len,verifier,n,max_len,temp,cache):
    if n<=0: return {'n':0,'exact':0,'rate':0.0,'timeouts':0}
    gen,glen=decode_batch(decoder,enc1,src_len,n,max_len,temp)
    cands,infos,h,m=verifier.evaluate_generated(gen,glen,cache)
    examples=[]
    for c,z in zip(cands,infos):
        if z.verified_general and len(examples)<3:
            examples.append({'expr':str(z.expression),'tokens':' '.join(c[1]),'reward':float(z.reward)})
    ne=exact_count(infos)
    return {'n':n,'exact':ne,'rate':float(ne/n),'timeouts':timeout_count(infos),
            'best_reward':float(max(z.reward for z in infos)),'examples':examples,'cache_hits':h,'cache_misses':m}


def main():
    a=parse_args(); seed_all(a.seed)
    device=torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'device={device}')
    env,encoder,decoder,params,ckpt=load_pretrained(a.checkpoint,device)
    nworkers=resolve_verifier_workers(a.verifier_workers)
    verifier_pool=ParallelVerifierPool(nworkers)
    print(f'verifier_workers={nworkers} start_method=spawn')
    trainable=configure_trainable(decoder,a.scope)
    base=[p.detach().cpu().clone() for p in trainable]
    print(f'trainable_parameters={sum(p.numel() for p in trainable):,} scope={a.scope}')
    problems=list(read_jsonl(a.problems))
    if a.max_problems>0: problems=problems[:a.max_problems]
    fout=open(a.output,'w',encoding='utf-8'); summaries=[]

    for pi,rec in enumerate(problems):
        # independent task adaptation: restore exactly the pretrained trainable slice.
        with torch.no_grad():
            for p,b in zip(trainable,base): p.copy_(b.to(device=p.device,dtype=p.dtype))
        decoder.eval(); seed_all(a.seed+1009*pi)
        optimizer=torch.optim.Adam(trainable,lr=a.lr)
        problem=make_problem_from_record(env,rec)
        prefix,enc1,src_len=encode_problem(env,encoder,problem,device)
        verifier=ExactODEVerifier(env,problem,a.candidate_timeout,worker_pool=verifier_pool)
        train_cache={}; eval_cache={}; buf=[]; search_calls=0; search_unique_verifier_evals=0; heldout_calls=0; heldout_unique_verifier_evals=0
        print('\n'+'='*100)
        print(f'problem {pi+1}/{len(problems)} id={rec.get("id")}')
        print(f'equation={rec.get("equation_latex")}')
        print(f'prefix={" ".join(prefix)}')

        # baseline held-out, non-invasive
        st=capture_rng_state(); baseline=eval_policy(decoder,enc1,src_len,verifier,a.eval_rollouts,a.max_len,a.eval_temperature,eval_cache); restore_rng_state(st)
        heldout_calls+=a.eval_rollouts; heldout_unique_verifier_evals+=baseline.get('cache_misses',0)
        gg,gl=decode_greedy(decoder,enc1,src_len,a.max_len); _,ginfo,_,_=verifier.evaluate_generated(gg,gl,{})
        baseline['greedy_exact']=bool(ginfo[0].verified_general)
        print(f"baseline: exact={baseline['exact']}/{baseline['n']} ({100*baseline['rate']:.2f}%) greedy={int(ginfo[0].verified_general)}")

        # frozen warmup search
        warm_exact=0; seen=0; warm_timeouts=0; first_exact_at=None
        while seen<a.warmup_samples:
            bs=min(a.warmup_batch_size,a.warmup_samples-seen)
            gen,glen=decode_batch(decoder,enc1,src_len,bs,a.max_len,a.warmup_temperature)
            cands,infos,_hits,_misses=verifier.evaluate_generated(gen,glen,train_cache)
            search_unique_verifier_evals += _misses
            ne=exact_count(infos); warm_exact+=ne; warm_timeouts+=timeout_count(infos)
            new_exact=update_buffer(buf,cands,infos,-1,a.max_exact_buffer,a.max_approx_buffer)
            if ne and first_exact_at is None:
                # index within verifier-call stream; exact position inside batch is not needed for paper metric yet
                first_exact_at=seen+bs
            seen+=bs; search_calls+=bs
            print(f'  warmup {seen}/{a.warmup_samples}: exact={warm_exact} buffer_exact={sum(e["exact"] for e in buf)}')
        warm_replay=None
        if any(e['exact'] for e in buf):
            warm_replay=replay(env,decoder,enc1,src_len,buf,optimizer,trainable,device,
                               a.replay_updates_new_exact,a.replay_weight,a.grad_clip)
        print(f'warmup summary: exact={warm_exact}/{a.warmup_samples} first_exact_by={first_exact_at} replay={warm_replay}')

        step_rows=[]
        for step in range(a.steps):
            gen,glen=decode_batch(decoder,enc1,src_len,a.rollouts,a.max_len,a.temperature)
            cands,infos,_hits,_misses=verifier.evaluate_generated(gen,glen,train_cache); search_calls+=a.rollouts; search_unique_verifier_evals+=_misses
            rewards_np=np.asarray([z.reward for z in infos],dtype=np.float32)
            ne=exact_count(infos); new_exact=update_buffer(buf,cands,infos,step,a.max_exact_buffer,a.max_approx_buffer)
            have_exact=any(e['exact'] for e in buf)
            rl_loss=None
            # Search-then-adapt: before first exact discovery, keep policy frozen instead of
            # suppressing the rare correct mode with approximate imitation/RL.
            if have_exact and float(rewards_np.std())>1e-8:
                rewards=torch.tensor(rewards_np,device=device)
                adv=(rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-6)
                decoder.train()
                lp=sequence_logprobs(decoder,enc1.detach().expand(a.rollouts,-1,-1).contiguous(),
                    src_len.expand(a.rollouts).contiguous(),gen,glen,length_normalize=a.length_normalize)
                loss=-(adv.detach()*lp).mean(); optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable,a.grad_clip); optimizer.step(); decoder.eval(); rl_loss=float(loss.item())
            nup=a.replay_updates_new_exact if new_exact else (a.replay_updates if have_exact else 0)
            rep=replay(env,decoder,enc1,src_len,buf,optimizer,trainable,device,nup,a.replay_weight,a.grad_clip)
            gg,gl=decode_greedy(decoder,enc1,src_len,a.max_len); _,ginfo,_,_=verifier.evaluate_generated(gg,gl,{})
            ev=None
            if a.eval_every>0 and (step+1)%a.eval_every==0:
                st=capture_rng_state(); ev=eval_policy(decoder,enc1,src_len,verifier,a.eval_rollouts,a.max_len,a.eval_temperature,eval_cache); restore_rng_state(st)
                heldout_calls+=a.eval_rollouts; heldout_unique_verifier_evals+=ev.get('cache_misses',0)
            print(f"  step={step:02d} train_exact={ne}/{a.rollouts} new_exact={len(new_exact)} greedy={int(ginfo[0].verified_general)}"+
                  (f" fresh={ev['exact']}/{ev['n']} ({100*ev['rate']:.1f}%)" if ev else ''))
            step_rows.append({'step':step,'train_exact':ne,'new_exact':len(new_exact),'greedy_exact':bool(ginfo[0].verified_general),
                              'rl_loss':rl_loss,'replay':rep,'eval':ev,'timeouts':timeout_count(infos),'search_calls':search_calls})

        st=capture_rng_state(); final=eval_policy(decoder,enc1,src_len,verifier,a.eval_rollouts,a.max_len,a.eval_temperature,eval_cache); restore_rng_state(st)
        heldout_calls+=a.eval_rollouts; heldout_unique_verifier_evals+=final.get('cache_misses',0)
        gg,gl=decode_greedy(decoder,enc1,src_len,a.max_len); gc,ginfo,_,_=verifier.evaluate_generated(gg,gl,{})
        summary={'id':rec.get('id'),'equation_latex':rec.get('equation_latex'),'coverage_rate':rec.get('sample_exact_rate'),
                 'baseline':baseline,'warmup_exact':warm_exact,'warmup_samples':a.warmup_samples,'first_exact_by':first_exact_at,
                 'final':final,'final_greedy_exact':bool(ginfo[0].verified_general),'final_greedy_expr':str(ginfo[0].expression),
                 'search_candidate_samples':search_calls,'search_unique_verifier_evals':search_unique_verifier_evals,
                 'heldout_candidate_samples':heldout_calls,'heldout_unique_verifier_evals':heldout_unique_verifier_evals,'steps':step_rows}
        summaries.append(summary); fout.write(json.dumps(summary,ensure_ascii=False)+'\n'); fout.flush()
        print(f"FINAL exact={final['exact']}/{final['n']} ({100*final['rate']:.2f}%) greedy={int(summary['final_greedy_exact'])} search_samples={search_calls} unique_verifier={search_unique_verifier_evals}")

    fout.close(); verifier_pool.close()
    agg={
        'n_problems':len(summaries),
        'mean_baseline_exact_rate':float(np.mean([s['baseline']['rate'] for s in summaries])) if summaries else 0.0,
        'mean_final_exact_rate':float(np.mean([s['final']['rate'] for s in summaries])) if summaries else 0.0,
        'greedy_exact_before':int(sum(bool(s['baseline'].get('greedy_exact',False)) for s in summaries)),
        'greedy_exact_after':int(sum(s['final_greedy_exact'] for s in summaries)),
        'warmup_discovery_rate':float(np.mean([s['warmup_exact']>0 for s in summaries])) if summaries else 0.0,
        'per_problem':[{k:v for k,v in s.items() if k!='steps'} for s in summaries],
    }
    with open(a.summary,'w',encoding='utf-8') as f: json.dump(agg,f,ensure_ascii=False,indent=2)
    print('\n=== aggregate ==='); print(json.dumps({k:v for k,v in agg.items() if k!='per_problem'},indent=2))
    print(f'results={a.output}\nsummary={a.summary}')

if __name__=='__main__': main()
