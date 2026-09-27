#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import (
    Problem, default_env_for_verifier, equation_to_tokens, exact_generality,
    exact_zero, numeric_generality_rank, residual_expression,
)
from ttrl_ode_benchmark.dataset import iter_dataset_rows
from ttrl_ode_benchmark.exact_verifier import candidate_time_limit, CandidateVerificationTimeout
from ttrl_ode_benchmark.latex_ode import (
    LatexODEParseError, detect_order, has_conditions, parse_ode2_equation,
    parse_reference_solution,
)

REPO_URL='https://github.com/hse-scila/dif_equations_fine_tune.git'


def parse_args():
    p=argparse.ArgumentParser(description='Prepare verified ODE2 subset from HSE 300k ODE corpus')
    p.add_argument('--dataset-dir', default='hse_dif_equations_fine_tune',
                   help='HSE repo root or its data directory')
    p.add_argument('--clone', action='store_true', help='git clone the public HSE repo if dataset-dir does not exist')
    p.add_argument('--output', default='hse300k_ode2_verified.jsonl')
    p.add_argument('--reject-log', default='hse300k_ode2_rejects.jsonl')
    p.add_argument('--max-rows', type=int, default=0, help='0 = scan all rows')
    p.add_argument('--max-accepted', type=int, default=0, help='0 = keep all compatible verified rows')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip-reference-check', action='store_true',
                   help='Parse/filter only; not recommended for paper benchmark construction')
    p.add_argument('--reference-timeout', type=float, default=4.0,
                   help='Hard wall-clock timeout per reference parse+verification (seconds; 0 disables)')
    return p.parse_args()


def maybe_clone(path):
    if os.path.exists(path): return
    parent=os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(parent,exist_ok=True)
    print(f'cloning {REPO_URL} -> {path}')
    subprocess.run(['git','clone','--depth','1',REPO_URL,path],check=True)


def main():
    a=parse_args()
    if a.clone: maybe_clone(a.dataset_dir)
    env=default_env_for_verifier()
    rng=random.Random(a.seed)
    scanned=accepted=0
    reasons={}
    out=open(a.output,'w',encoding='utf-8')
    rej=open(a.reject_log,'w',encoding='utf-8')

    def reject(row, reason, detail=''):
        nonlocal reasons
        reasons[reason]=reasons.get(reason,0)+1
        rej.write(json.dumps({**row,'reject_reason':reason,'detail':str(detail)[:1000]},ensure_ascii=False)+'\n')

    for row in iter_dataset_rows(a.dataset_dir):
        scanned += 1
        if a.max_rows and scanned>a.max_rows: break
        eq_tex=str(row.get('equation_latex',''))
        ans_tex=str(row.get('answer_latex',''))
        if detect_order(eq_tex)!=2:
            reject(row,'not_ode2'); continue
        if has_conditions(eq_tex):
            reject(row,'has_ic_bc'); continue
        try:
            eq=parse_ode2_equation(eq_tex,env)
        except BaseException as e:
            reject(row,'equation_parse',e); continue
        try:
            prefix=equation_to_tokens(env,eq)
        except BaseException as e:
            reject(row,'model_vocab',e); continue

        ref=None; ref_verified=False; rank=0
        if not a.skip_reference_check:
            try:
                with candidate_time_limit(a.reference_timeout):
                    ref=parse_reference_solution(ans_tex,env,order=2)
                    residual=residual_expression(env,eq,ref)
                    zero,_=exact_zero(residual,seconds=2)
                    rank=numeric_generality_rank(env,ref,2)
                    ref_verified=bool(zero and rank>=2 and exact_generality(env,ref,2,seconds=1))
                if not ref_verified:
                    reject(row,'reference_not_verified',f'zero={zero} rank={rank} ref={ref}'); continue
            except CandidateVerificationTimeout as e:
                reject(row,'reference_timeout',e); continue
            except BaseException as e:
                reject(row,'reference_parse_or_verify',e); continue

        rec={
            'id':f"{row['source_file']}:{row['source_row']}",
            'source_file':row['source_file'],'source_row':row['source_row'],
            'category':row.get('category',''),
            'equation_latex':eq_tex,'answer_latex':ans_tex,
            'equation_sympy':str(eq),
            'reference_sympy':str(ref) if ref is not None else None,
            'reference_verified':bool(ref_verified),'reference_rank':int(rank),
            'prefix':' '.join(prefix),'order':2,
        }
        out.write(json.dumps(rec,ensure_ascii=False)+'\n')
        accepted+=1
        if accepted<=10:
            print(f"[accept {accepted}] {rec['id']} :: {eq_tex}")
            print(f"  sympy={eq}")
            print(f"  ref={ref}")
            print(f"  prefix={rec['prefix']}")
        if a.max_accepted and accepted>=a.max_accepted: break
        if scanned%5000==0:
            print(f'scanned={scanned} accepted={accepted}')

    out.close(); rej.close()
    print('\n=== preparation summary ===')
    print(f'scanned={scanned} accepted_verified_ode2={accepted}')
    print(f'output={a.output}')
    print(f'reject_log={a.reject_log}')
    print('reject_reasons='+json.dumps(dict(sorted(reasons.items(),key=lambda kv:-kv[1])),indent=2))

if __name__=='__main__': main()
