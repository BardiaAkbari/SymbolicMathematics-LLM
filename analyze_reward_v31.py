#!/usr/bin/env python3
import argparse,json

def main():
 p=argparse.ArgumentParser();p.add_argument('path');a=p.parse_args()
 n=0; exact=0; gate_top=0; ranklow_top=0; high6=0; high4=0; changed=0
 fam={}
 for line in open(a.path,encoding='utf-8'):
  if not line.strip():continue
  r=json.loads(line);n+=1;exact+=int(r.get('exact_discovered',0)>0)
  f=r.get('failure_family','?');fam.setdefault(f,{'n':0,'gate_fail':0,'ranklow':0,'high6':0});fam[f]['n']+=1
  t=(r.get('v31_top') or [{}])[0]
  if not t.get('v31_gate',True):gate_top+=1;fam[f]['gate_fail']+=1
  if int(t.get('rank',2))<2:ranklow_top+=1;fam[f]['ranklow']+=1
  if not t.get('exact',False) and float(t.get('v31_reward',-99))>6:high6+=1;fam[f]['high6']+=1
  if not t.get('exact',False) and float(t.get('v31_reward',-99))>4:high4+=1
  v3=(r.get('v3_top') or [{}])[0]
  if t.get('tokens')!=v3.get('tokens'):changed+=1
 print(f'problems={n} exact_discovered_in_fresh_sample={exact}')
 print(f'top1_gate_fail={gate_top} top1_rank_lt2={ranklow_top}')
 print(f'nonexact_top1_reward_gt6={high6} nonexact_top1_reward_gt4={high4}')
 print(f'top1_changed_v3_to_v31={changed}/{n}')
 for k,v in sorted(fam.items()):print(k,v)
 print('\nThese are mechanical flags only. Final reward judgment still requires manual mathematical inspection.')
if __name__=='__main__':main()
