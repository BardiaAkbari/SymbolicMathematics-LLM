#!/usr/bin/env python3
import argparse,json,sys,os
from types import SimpleNamespace
import sympy as sp
ROOT=os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:sys.path.insert(0,ROOT)
from ttrl_ode_benchmark.latex_ode import parse_ode2_equation
from ttrl_ode_benchmark.rich_reward_v31 import rich_pre_exact_reward_v31

x=sp.Symbol('x',real=True);f=sp.Function('f');a8,a9=sp.symbols('a8 a9',real=True)
env=SimpleNamespace(local_dict={'x':x,'f':f,'a8':a8,'a9':a9},coefficients={'a8':a8,'a9':a9})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('log');ap.add_argument('--output',default='logged_v3top_rescored_v31.jsonl');a=ap.parse_args()
 out=open(a.output,'w')
 with open(a.log,encoding='utf-8') as fh:
  for i,line in enumerate(fh):
   if not line.strip():continue
   rec=json.loads(line);eq=parse_ode2_equation(rec['equation_latex'],env);problem=SimpleNamespace(equation=eq)
   rows=[]
   for j,r in enumerate(rec.get('v3_top',[])):
    try:
     h=sp.sympify(r['expr'],locals=env.local_dict)
     z=rich_pre_exact_reward_v31(env,problem,h,probe_seed=314159+10007*i)
     rows.append({'rank_v3':j+1,'expr':r['expr'],'old_v3':r.get('v3_reward'),'v31':z.pre_reward,'gate':z.generality_gate_pass,'W0':z.wronskian_zero,'eqdom':z.equation_domain_scores,'dirdom':z.direction_domain_scores,'base':z.base_score,'error':z.error})
    except Exception as e: rows.append({'rank_v3':j+1,'expr':r.get('expr'),'error':f'{type(e).__name__}: {e}'})
   out.write(json.dumps({'id':rec['id'],'family':rec.get('failure_family'),'equation_latex':rec['equation_latex'],'rows':rows})+'\n')
 out.close();print(a.output)
if __name__=='__main__':main()
