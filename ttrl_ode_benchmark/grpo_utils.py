from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import torch
import torch.nn.functional as F


@dataclass
class GRPOLossInfo:
    loss: float
    policy_loss: float
    kl_loss: float
    entropy: float
    clip_fraction: float
    mean_ratio: float
    approx_old_kl: float

    def to_dict(self):
        return asdict(self)


def _masked_per_sequence_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x, mask: [B,T]. Return [B], giving every sequence equal weight."""
    m = mask.to(dtype=x.dtype)
    return (x * m).sum(-1) / m.sum(-1).clamp_min(1.0)


def teacher_forced_token_stats(
    decoder,
    src_enc: torch.Tensor,
    src_len: torch.Tensor,
    generated: torch.Tensor,
    gen_len: torch.Tensor,
    *,
    temperature: float = 1.0,
    require_entropy: bool = True,
):
    """Differentiable token log-probs / entropy for sampled generated sequences.

    Returns tensors shaped [B,T] for token_logp, entropy, mask.  The same
    temperature used during generation should be supplied so PPO ratios are
    defined for the actual sampling distribution.
    """
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0:
        raise ValueError("temperature must be finite and > 0")

    slen, bs = generated.shape
    decoded = decoder(
        "fwd", x=generated, lengths=gen_len, causal=True,
        src_enc=src_enc, src_len=src_len,
    )
    logits = decoder.proj(decoded[:-1]).float() / temp
    targets = generated[1:]
    logp_all = F.log_softmax(logits, dim=-1)
    token_lp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    if require_entropy:
        probs = logp_all.exp()
        entropy = -(probs * logp_all).sum(-1)
    else:
        entropy = torch.zeros_like(token_lp)

    positions = torch.arange(slen - 1, device=generated.device)[:, None]
    mask = positions < (gen_len[None, :] - 1)
    # transpose to [B,T]
    return token_lp.transpose(0, 1), entropy.transpose(0, 1), mask.transpose(0, 1)


def group_normalized_advantages(rewards, *, eps: float = 1e-6, clip: float | None = 5.0):
    if not torch.is_tensor(rewards):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    rewards = rewards.float()
    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + float(eps))
    if clip is not None and clip > 0:
        adv = adv.clamp(-float(clip), float(clip))
    return adv


def k3_kl_from_logps(policy_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Non-negative sampled k3 KL estimator for pi || ref.

    Samples are from pi/old-pi.  Let log_r = log(ref/pi). Then
    exp(log_r) - log_r - 1 is non-negative and has the right local geometry.
    """
    log_r = (ref_logp - policy_logp).clamp(-20.0, 20.0)
    return torch.exp(log_r) - log_r - 1.0


def grpo_clipped_loss(
    current_logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor | None,
    entropy: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.02,
    entropy_coef: float = 0.002,
):
    """Token-level clipped GRPO/PPO objective with one scalar advantage per sequence."""
    if current_logp.shape != old_logp.shape or current_logp.shape != entropy.shape or current_logp.shape != mask.shape:
        raise ValueError("token tensors must have identical [B,T] shapes")
    if advantages.ndim != 1 or advantages.shape[0] != current_logp.shape[0]:
        raise ValueError("advantages must have shape [B]")

    log_ratio = (current_logp - old_logp).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    a = advantages.to(current_logp.device, dtype=current_logp.dtype)[:, None]
    unclipped = ratio * a
    clipped = ratio.clamp(1.0 - float(clip_eps), 1.0 + float(clip_eps)) * a
    surrogate = torch.minimum(unclipped, clipped)
    policy_obj = _masked_per_sequence_mean(surrogate, mask).mean()
    policy_loss = -policy_obj

    if ref_logp is not None and float(kl_coef) != 0.0:
        kl_tok = k3_kl_from_logps(current_logp, ref_logp)
        kl_loss = _masked_per_sequence_mean(kl_tok, mask).mean()
    else:
        kl_loss = current_logp.sum() * 0.0

    entropy_mean = _masked_per_sequence_mean(entropy, mask).mean()
    loss = policy_loss + float(kl_coef) * kl_loss - float(entropy_coef) * entropy_mean

    with torch.no_grad():
        m = mask
        clipped_bool = ((ratio < 1.0 - float(clip_eps)) | (ratio > 1.0 + float(clip_eps))) & m
        clip_fraction = clipped_bool.float().sum() / m.float().sum().clamp_min(1.0)
        mean_ratio = _masked_per_sequence_mean(ratio, mask).mean()
        old_kl = _masked_per_sequence_mean(k3_kl_from_logps(current_logp, old_logp), mask).mean()

    info = GRPOLossInfo(
        loss=float(loss.detach().item()),
        policy_loss=float(policy_loss.detach().item()),
        kl_loss=float(kl_loss.detach().item()),
        entropy=float(entropy_mean.detach().item()),
        clip_fraction=float(clip_fraction.detach().item()),
        mean_ratio=float(mean_ratio.detach().item()),
        approx_old_kl=float(old_kl.detach().item()),
    )
    return loss, info


def mean_sampled_kl(current_logp: torch.Tensor, reference_logp: torch.Tensor, mask: torch.Tensor) -> float:
    with torch.no_grad():
        kl = k3_kl_from_logps(current_logp, reference_logp)
        return float(_masked_per_sequence_mean(kl, mask).mean().item())
