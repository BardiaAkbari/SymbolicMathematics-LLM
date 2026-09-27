from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch
import sympy as sp

from ttrl_lane_emden.core import Problem, encode_tokens, equation_to_tokens
from ttrl_ode_benchmark.latex_ode import parse_ode2_equation


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    d={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.random.get_rng_state()}
    if torch.cuda.is_available(): d['cuda']=torch.cuda.get_rng_state_all()
    return d


def restore_rng_state(d):
    random.setstate(d['python']); np.random.set_state(d['numpy']); torch.random.set_rng_state(d['torch'])
    if 'cuda' in d and torch.cuda.is_available(): torch.cuda.set_rng_state_all(d['cuda'])


def read_jsonl(path):
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def make_problem_from_record(env, rec):
    expr_text = rec.get('equation_sympy_override') or rec.get('equation_sympy')
    if expr_text:
        try:
            eq = sp.sympify(expr_text, locals=env.local_dict)
        except Exception:
            eq = parse_ode2_equation(rec['equation_latex'], env)
    else:
        eq=parse_ode2_equation(rec['equation_latex'],env)
    # Additive terms give the scale-resistant residual shaping decomposition.
    terms=tuple(sp.Add.make_args(sp.expand(eq)))
    return Problem(name=str(rec.get('id','hse_ode2')), equation=eq, order=2, mode='general',
                   expected=None, residual_terms=terms, input_form='hse_latex')


def encode_problem(env, encoder, problem, device):
    tokens=equation_to_tokens(env,problem.equation)
    x,lengths=encode_tokens(env,tokens,device)
    with torch.no_grad():
        enc=encoder('fwd',x=x,lengths=lengths,causal=False).transpose(0,1)
    return tokens,enc,lengths


def decode_batch(decoder, enc1, src_len_1, n, max_len, temperature):
    with torch.no_grad():
        return decoder.generate(
            enc1.expand(n,-1,-1).contiguous(), src_len_1.expand(n).contiguous(),
            max_len=max_len, sample_temperature=temperature,
        )


def decode_greedy(decoder, enc1, src_len_1, max_len):
    with torch.no_grad():
        return decoder.generate(enc1,src_len_1,max_len=max_len,sample_temperature=None)
