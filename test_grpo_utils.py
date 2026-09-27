#!/usr/bin/env python3
import torch
from ttrl_ode_benchmark.grpo_utils import group_normalized_advantages, grpo_clipped_loss


def main():
    # 4 sequences x 3 tokens. Make current logp a leaf so gradient causality is explicit.
    cur = torch.tensor([
        [-1.0,-1.2,-0.8],[-1.1,-1.0,-1.3],[-0.9,-1.4,-1.1],[-1.2,-0.9,-1.0]
    ], requires_grad=True)
    old = cur.detach().clone()
    ref = old - 0.2
    ent = (-cur).exp()  # differentiable proxy; only gradient causality matters in this unit test
    mask = torch.ones_like(cur, dtype=torch.bool)
    adv = group_normalized_advantages(torch.tensor([0.0,1.0,2.0,4.0]))
    loss,info = grpo_clipped_loss(cur,old,ref,ent,mask,adv,kl_coef=0.02,entropy_coef=0.002)
    loss.backward()
    assert torch.isfinite(cur.grad).all() and float(cur.grad.abs().sum()) > 0

    # Entropy term must itself change gradients when policy advantage is zero.
    cur2 = cur.detach().clone().requires_grad_(True)
    ent2 = (-cur2).exp()
    zadv = torch.zeros(4)
    loss2,_ = grpo_clipped_loss(cur2,cur2.detach(),None,ent2,mask,zadv,kl_coef=0.0,entropy_coef=0.01)
    loss2.backward()
    assert float(cur2.grad.abs().sum()) > 0

    # KL must itself change gradients if current != reference.
    cur3 = cur.detach().clone().requires_grad_(True)
    loss3,_ = grpo_clipped_loss(cur3,cur3.detach(),ref,torch.zeros_like(cur3),mask,zadv,kl_coef=0.1,entropy_coef=0.0)
    loss3.backward()
    assert float(cur3.grad.abs().sum()) > 0

    print('GRPO UTILS TESTS PASSED')
    print(info.to_dict())

if __name__=='__main__': main()
