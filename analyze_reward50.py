#!/usr/bin/env python3
import argparse, json, math
from collections import Counter, defaultdict

def main():
    p=argparse.ArgumentParser()
    p.add_argument('path', nargs='?', default='validation50_reward_v3_ranking.jsonl')
    a=p.parse_args()
    recs=[json.loads(x) for x in open(a.path,encoding='utf-8') if x.strip()]
    fam=Counter(r.get('failure_family') for r in recs)
    top1=[]
    flags=[]
    for r in recs:
        if not r.get('v3_top'): continue
        t=r['v3_top'][0]; top1.append(t['v3_reward'])
        f=[]
        if t.get('rank',2)<2: f.append('TOP1_RANK_LT_2')
        if t.get('independence',1.0)<1e-4 and t.get('v3_reward',-99)>0: f.append('TOP1_NEAR_DEPENDENT_POSITIVE')
        if t.get('v3_reward',-99)>6 and not t.get('exact'): f.append('TOP1_NONEXACT_GT_6')
        if r.get('failure_family')=='homogeneous' and t.get('q_equation',-99)>8 and t.get('q_direction_min',99)<4: f.append('HOM_EQ_HIDES_BAD_DIRECTION')
        if r.get('failure_family')!='homogeneous' and t.get('direction_penalty',0)>0 and t.get('v3_reward',-99)>0: f.append('INHOM_POSITIVE_WITH_BAD_DIRECTION')
        # flat top-10 plateau may make GRPO advantages nearly zero
        vals=[x['v3_reward'] for x in r.get('v3_top',[])[:10]]
        if len(vals)>=5 and max(vals)-min(vals)<1e-6: f.append('TOP10_FLAT_PLATEAU')
        if f: flags.append((r['id'],r['equation_latex'],r['failure_family'],t['v3_reward'],f,t['expr']))
    print('n_records=',len(recs))
    print('families=',dict(fam))
    if top1:
        print('top1_reward_mean=',sum(top1)/len(top1),'max=',max(top1),'min=',min(top1))
    print('flagged_records=',len(flags))
    c=Counter(z for *_,fs,_ in flags for z in fs)
    print('flag_counts=',dict(c))
    for i,(rid,eq,ff,rw,fs,expr) in enumerate(flags,1):
        print(f'[{i}] {rid} family={ff} top1_v3={rw:.6f} flags={",".join(fs)}')
        print(' ODE:',eq)
        print(' TOP1:',expr)

if __name__=='__main__': main()
