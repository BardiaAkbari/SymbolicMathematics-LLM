#!/usr/bin/env python3
import argparse, json


def load(path):
    with open(path,encoding='utf-8') as f: return json.load(f)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--grpo',default='grpo4_v31_summary.json')
    p.add_argument('--frozen',default='frozen4_control_summary.json')
    a=p.parse_args(); g=load(a.grpo); f=load(a.frozen)
    gm={r['id']:r for r in g['per_problem']}; fm={r['id']:r for r in f['per_problem']}
    ids=sorted(set(gm)&set(fm))
    print('id\tfrozen_discovered\tgrpo_discovered\tfrozen_first\tgrpo_first\tfrozen_greedy\tgrpo_greedy')
    for i in ids:
        x,y=fm[i],gm[i]
        print(f"{i}\t{int(x['first_exact_by_search_calls'] is not None)}\t{int(y['first_exact_by_search_calls'] is not None)}\t{x['first_exact_by_search_calls']}\t{y['first_exact_by_search_calls']}\t{int(x['final_greedy_exact'])}\t{int(y['final_greedy_exact'])}")
    print('\naggregate')
    for name,z in [('frozen',f),('grpo',g)]:
        print(name, {k:z.get(k) for k in ['n_problems','n_discovered','discovery_rate','greedy_exact_after','mean_final_exact_rate','mean_first_exact_calls_discovered']})

if __name__=='__main__': main()
