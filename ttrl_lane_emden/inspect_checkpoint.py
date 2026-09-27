#!/usr/bin/env python3
import argparse
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import trusted_torch_load

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
a = p.parse_args()
ckpt = trusted_torch_load(a.checkpoint, map_location='cpu')
print('top-level keys:', list(ckpt.keys()))
params = ckpt.get('params', {})
for k in ['tasks','emb_dim','n_enc_layers','n_dec_layers','n_heads','n_variables','n_coefficients','operators','max_ops','max_int','max_len','clean_prefix_expr','share_inout_emb']:
    print(f'{k}: {params.get(k)}')
for name in ['encoder','decoder']:
    sd = ckpt.get(name, {})
    print(name, 'tensors=', len(sd), 'parameters=', sum(v.numel() for v in sd.values() if torch.is_tensor(v)))
