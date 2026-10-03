#!/usr/bin/env python3
"""Reference-free TTRL for Lane-Emden n=5."""
import argparse, json, os, random, sys, time
import numpy as np
import torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from ttrl_lane_emden.core import configure_trainable,encode_tokens,equation_to_tokens,load_pretrained,sequence_logprobs,token_sequence_logprobs
from ttrl_lane_emden.ivp_n5 import LaneEmdenN5Verifier

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True); p.add_argument('--steps',type=int,default=100); p.add_argument('--rollouts',type=int,default=128)
    p.add_argument('--temperature',type=float,default=1.5); p.add_argument('--temperature-end',type=float,default=1.0)
    p.add_argument('--max-len',type=int,default=64); p.add_argument('--lr',type=float,default=3e-6)
    p.add_argument('--scope',choices=['proj','last_layer','decoder'],default='last_layer'); p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--elite-size',type=int,default=4); p.add_argument('--elite-updates',type=int,default=5); p.add_argument('--replay-updates',type=int,default=1)
    p.add_argument('--length-penalty',type=float,default=0.01); p.add_argument('--seed',type=int,default=0); p.add_argument('--cpu',action='store_true')
    p.add_argument('--save-jsonl',default='ttrl_lane_emden_n5_results.jsonl'); p.add_argument('--save-decoder',default='')
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda'); print('device=',device)
    env,encoder,decoder,_,_=load_pretrained(a.checkpoint,device)
    import sympy as sp
    x=env.local_dict['x']; y=env.local_dict['f'](x)
    equation=x*sp.diff(y,x,2)+2*sp.diff(y,x)+x*y**5
    verifier=LaneEmdenN5Verifier(env,length_penalty=a.length_penalty)
    tokens=equation_to_tokens(env,equation)
    print('\n=== Lane-Emden n=5 TTRL ==='); print('equation:',equation); print('input:', ' '.join(tokens)); print('scope:',a.scope)
    for q in encoder.parameters(): q.requires_grad_(False)
    trainable=configure_trainable(decoder,a.scope); opt=torch.optim.Adam(trainable,lr=a.lr)
    print('trainable_parameters=',sum(q.numel() for q in trainable))
    src,src_len=encode_tokens(env,tokens,device)
    with torch.no_grad(): enc=encoder('fwd',x=src,lengths=src_len,causal=False).transpose(0,1)
    elites=[]; cache={}; out=open(a.save_jsonl,'w',encoding='utf-8')
    for step in range(a.steps):
        t=time.perf_counter(); alpha=step/max(a.steps-1,1); temp=a.temperature+alpha*(a.temperature_end-a.temperature)
        decoder.eval()
        with torch.no_grad():
            generated,gen_len=decoder.generate(enc.expand(a.rollouts,-1,-1).contiguous(),src_len.expand(a.rollouts).contiguous(),max_len=a.max_len,sample_temperature=temp)
        candidates,infos,_,_=verifier.evaluate_generated(generated,gen_len,cache)
        rewards=np.asarray([z.reward for z in infos],dtype=np.float32); order=np.argsort(-rewards); best=infos[int(order[0])]
        print(f'\n[step {step:03d}] T={temp:.3f} mean={rewards.mean():.3f} max={rewards.max():.3f} exact={sum(z.exact for z in infos)}')
        for i in order[:3]:
            z=infos[int(i)]; print(f'  reward={z.reward:8.3f} ode={z.ode_residual:.3e} anchor={z.anchor_error:.3e} exact={int(z.exact)} :: {z.expression}')
        existing={tuple(e['ids']) for e in elites}
        for (ids,words),z in zip(candidates,infos):
            k=tuple(int(i) for i in ids)
            if z.expression is not None and k not in existing:
                elites.append({'ids':list(k),'reward':float(z.reward),'expression':str(z.expression),'length':len(k)})
                existing.add(k)
        elites.sort(key=lambda e:(-e['reward'],e['length'])); elites=elites[:a.elite_size]
        rt=torch.tensor(rewards,device=device); rl_loss=None
        if float(rt.std(unbiased=False))>1e-8:
            adv=(rt-rt.mean())/(rt.std(unbiased=False)+1e-6); decoder.train()
            lp=sequence_logprobs(decoder,enc.detach().expand(a.rollouts,-1,-1).contiguous(),src_len.expand(a.rollouts).contiguous(),generated,gen_len,length_normalize=True)
            loss=-(adv.detach()*lp).mean(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,a.grad_clip); opt.step(); rl_loss=float(loss.item()); decoder.eval()
        if elites:
            decoder.train()
            ids=[e['ids'] for e in elites]
            for _ in range(a.elite_updates if step==0 else a.replay_updates):
                lp=token_sequence_logprobs(env,decoder,enc.detach(),src_len,ids,device,length_normalize=False); loss=-lp.mean(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,a.grad_clip); opt.step()
            decoder.eval()
        with torch.no_grad(): g,gl=decoder.generate(enc,src_len,max_len=a.max_len,sample_temperature=None)
        _,gi,_,_=verifier.evaluate_generated(g,gl,cache); greedy=gi[0]
        print(f'  greedy reward={greedy.reward:.3f} ode={greedy.ode_residual:.3e} anchor={greedy.anchor_error:.3e} exact={int(greedy.exact)}')
        out.write(json.dumps({'step':step,'temperature':temp,'reward_mean':float(rewards.mean()),'reward_max':float(rewards.max()),'exact_count':int(sum(z.exact for z in infos)),'best_expression':str(best.expression),'best_ode':best.ode_residual,'best_anchor':best.anchor_error,'greedy_expression':str(greedy.expression),'greedy_reward':greedy.reward,'greedy_exact':greedy.exact,'rl_loss':rl_loss,'time_s':time.perf_counter()-t})+'\n'); out.flush()
    out.close()
    if a.save_decoder: torch.save(decoder.state_dict(),a.save_decoder)

if __name__=='__main__': main()
