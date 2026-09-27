#!/usr/bin/env python3
"""Create mathematically equivalent representation-shift variants.

All non-scalar factors used here are nonzero for real x, so multiplying/dividing
by them preserves the real solution set (unlike multiplying by x).
"""
import argparse, json, os, sys
import sympy as sp

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from ttrl_lane_emden.core import default_env_for_verifier, equation_to_tokens
from ttrl_ode_benchmark.common import read_jsonl, make_problem_from_record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--problems',default='hse300k_ode2_verified.jsonl')
    p.add_argument('--output',default='hse300k_ode2_representation_shift.jsonl')
    p.add_argument('--max-problems',type=int,default=1000)
    p.add_argument('--variants',default='original,neg,scale7,mul_exp,mul_quad,div_quad')
    a=p.parse_args()
    names=[x.strip() for x in a.variants.split(',') if x.strip()]
    env=default_env_for_verifier(); x=env.local_dict['x']
    rows=list(read_jsonl(a.problems))
    if a.max_problems>0: rows=rows[:a.max_problems]
    n=0; rejected=0
    with open(a.output,'w',encoding='utf-8') as f:
        for rec in rows:
            try: base=make_problem_from_record(env,rec).equation
            except Exception: rejected+=1; continue
            variants={
                'original':base,
                'neg':-base,
                'scale7':7*base,
                'mul_exp':sp.exp(x)*base,
                'mul_quad':(x**2+1)*base,
                'div_quad':base/(x**2+1),
            }
            for name in names:
                if name not in variants: continue
                eq=variants[name]
                try: prefix=equation_to_tokens(env,eq)
                except Exception: continue
                out=dict(rec)
                out['base_id']=rec.get('id'); out['id']=f"{rec.get('id')}::repr={name}"
                out['representation_variant']=name
                out['equation_sympy_override']=str(eq)
                out['prefix_variant']=' '.join(prefix)
                f.write(json.dumps(out,ensure_ascii=False)+'\n'); n+=1
    print(f'wrote={n} rejected_base={rejected} output={a.output}')

if __name__=='__main__': main()
