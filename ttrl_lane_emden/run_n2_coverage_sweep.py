#!/usr/bin/env python3
"""Coverage sweep for Lane--Emden n=2 across algebraic forms and temperatures."""
import argparse, os, random, sys, time
import numpy as np
import torch

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0,ROOT)

from ttrl_lane_emden.core import encode_tokens, equation_to_tokens, load_pretrained, make_problem
from ttrl_lane_emden.ivp_n2 import LaneEmdenN2Verifier, summarize_ivp

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--forms',default='standard,cleared')
    p.add_argument('--temperatures',default='0.5,0.7,1.0,1.3')
    p.add_argument('--samples-per-setting',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--max-len',type=int,default=128)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--cpu',action='store_true')
    return p.parse_args()

def main():
    a=args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    dev=torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    print('device='+str(dev))
    env,enc,dec,_,_=load_pretrained(a.checkpoint,dev)
    verifier=LaneEmdenN2Verifier(env)
    forms=[s.strip() for s in a.forms.split(',') if s.strip()]
    temps=[float(s) for s in a.temperatures.split(',') if s.strip()]
    print('certify: ode<=%.3g ref<=%.3g anchor<=%.3g' % (verifier.certify_ode_rel,verifier.certify_ref_nrmse,verifier.certify_anchor_rmse))
    print('elite:   ode<=%.3g ref<=%.3g anchor<=%.3g' % (verifier.elite_ode_rel,verifier.elite_ref_nrmse,verifier.elite_anchor_rmse))
    results=[]
    for form in forms:
        prob=make_problem(env,'lane_emden',n=2,mode='general',lane_form=form)
        tok=equation_to_tokens(env,prob.equation)
        x,slen=encode_tokens(env,tok,dev)
        with torch.no_grad(): enc1=enc('fwd',x=x,lengths=slen,causal=False).transpose(0,1)
        print('\n=== form=%s ===' % form)
        print('prefix='+' '.join(tok))
        for temp in temps:
            cache={}; infos_all=[]; cands_all=[]; seen=0; t0=time.perf_counter()
            while seen<a.samples_per_setting:
                bs=min(a.batch_size,a.samples_per_setting-seen)
                with torch.no_grad():
                    gen,glen=dec.generate(enc1.expand(bs,-1,-1).contiguous(),slen.expand(bs).contiguous(),max_len=a.max_len,sample_temperature=temp)
                cands,infos,_,_=verifier.evaluate_generated(gen,glen,cache)
                cands_all.extend(cands); infos_all.extend(infos); seen+=bs
            cert=sum(z.certified_ivp for z in infos_all); elite=sum(z.elite_eligible for z in infos_all)
            rows=summarize_ivp(cands_all,infos_all,top_k=1); b=rows[0]
            print('temp=%.2f certified=%d/%d elite=%d/%d best_reward=%.3f ode=%.3e ref=%.3e anchor=%.3e time=%.1fs' % (
                temp,cert,a.samples_per_setting,elite,a.samples_per_setting,b['reward'],b['ode_rel'],b['ref_nrmse'],b['anchor_rmse'],time.perf_counter()-t0))
            print('  best=%s' % b['expr'])
            print('  fitted=%s' % b['fitted_expr'])
            results.append((cert,elite,b['reward'],form,temp,b))
    print('\n=== overall best by joint reward ===')
    for cert,elite,reward,form,temp,b in sorted(results,key=lambda r:r[2],reverse=True)[:5]:
        print('form=%s temp=%.2f cert=%d elite=%d reward=%.3f ode=%.3e ref=%.3e anchor=%.3e :: %s' % (
            form,temp,cert,elite,reward,b['ode_rel'],b['ref_nrmse'],b['anchor_rmse'],b['expr']))

if __name__=='__main__': main()
