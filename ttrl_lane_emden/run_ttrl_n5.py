#!/usr/bin/env python3
"""Verifier-guided TTRL for the regular n=5 Lane-Emden IVP.

Run from the SymbolicMathematics-LLM repository root:
    python ttrl_lane_emden/run_ttrl_n5.py --checkpoint ode2.pth ...

The model input is always the cleared equation:
    x*y'' + 2*y' + x*y**5 = 0
with y(0)=1 and y'(0)=0.  The known closed form is used only by the
numeric reward verifier; a candidate is counted as an exact solution only
when SymPy verifies the ODE and both initial conditions symbolically.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import sympy as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrl_lane_emden.core import (  # noqa: E402
    encode_tokens,
    equation_to_tokens,
    exact_zero,
    generated_to_candidates,
    ids_to_sympy,
    load_pretrained,
    make_problem,
    residual_expression,
    sequence_logprobs,
    token_ids_to_generated_batch,
)

ENCODER_SCOPES = {"last_encoder_decoder", "full"}


@dataclass
class IVPInfo:
    reward: float
    valid_parse: bool
    exact_residual_zero: bool
    generality_rank: int
    numerical_mse: float
    expression: Optional[sp.Expr]
    residual: Optional[sp.Expr] = None
    verified_general: bool = False  # Means exact IVP verification in this runner.
    error: Optional[str] = None
    exact_checked: bool = False
    exact_skipped: bool = False
    target_mse: float = 1e6
    residual_mse: float = 1e6
    ic_mse: float = 1e6
    token_length: int = 0


def parse_args():
    p = argparse.ArgumentParser(
        description="TTRL for Lane-Emden n=5 using the cleared equation and IVP verifier"
    )
    p.add_argument("--checkpoint", required=True, help="Path to the original ode2.pth checkpoint")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--rollouts", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--temperature-end", type=float, default=1.0)
    p.add_argument(
        "--scope",
        choices=["last_layer", "decoder", "last_encoder_decoder", "full"],
        default="last_layer",
        help=(
            "last_layer = last decoder block + output projection; decoder = full decoder; "
            "last_encoder_decoder = last encoder block + full decoder; full = full encoder + decoder"
        ),
    )
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--sample-batch-size", type=int, default=32)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--length-normalize", action=argparse.BooleanOptionalAction, default=True)

    # Reward components.  Pointwise target error provides a dense signal; the
    # cleared ODE residual and IVP conditions prevent answer-only overfitting.
    p.add_argument("--target-weight", type=float, default=1.0)
    p.add_argument("--residual-weight", type=float, default=1.0)
    p.add_argument("--ic-weight", type=float, default=0.75)
    p.add_argument("--length-penalty", type=float, default=0.01)
    p.add_argument("--exact-bonus", type=float, default=20.0)
    p.add_argument("--x-min", type=float, default=1.0)
    p.add_argument("--x-max", type=float, default=5.0)
    p.add_argument("--n-points", type=int, default=20)
    p.add_argument("--parameter-draws", type=int, default=3)
    p.add_argument(
        "--symbolic-timeout",
        type=float,
        default=0.35,
        help="Timeout in seconds per symbolic simplification pass during exact verification",
    )
    p.add_argument(
        "--exact-verify-top-k",
        type=int,
        default=8,
        help="During training, exact-check the top K unique candidates per step (0 = all)",
    )

    # Verified-elite replay preserves discovered exact symbolic solutions.
    p.add_argument("--elite-updates", type=int, default=4)
    p.add_argument("--elite-replay-updates", type=int, default=1)
    p.add_argument("--elite-weight", type=float, default=1.0)
    p.add_argument("--max-elites", type=int, default=4)

    # Evaluation is kept separate from the training rollouts and their loss.
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-rollouts", type=int, default=32)
    p.add_argument("--eval-temperature", type=float, default=None)
    p.add_argument("--save-jsonl", default=None)
    p.add_argument("--save-checkpoint", default=None)
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_trainable(module, yes: bool):
    for parameter in module.parameters():
        parameter.requires_grad_(yes)


def configure_scope(encoder, decoder, scope: str):
    """Freeze everything, then enable only parameters for the selected ablation."""
    set_trainable(encoder, False)
    set_trainable(decoder, False)

    def enable(modules):
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    if scope == "last_layer":
        # Mirrors the original core.configure_trainable(..., "last_layer").
        enable([
            decoder.attentions[-1],
            decoder.layer_norm1[-1],
            decoder.layer_norm15[-1],
            decoder.encoder_attn[-1],
            decoder.ffns[-1],
            decoder.layer_norm2[-1],
            decoder.proj,
        ])
    elif scope == "decoder":
        set_trainable(decoder, True)
    elif scope == "last_encoder_decoder":
        enable([
            encoder.attentions[-1],
            encoder.layer_norm1[-1],
            encoder.ffns[-1],
            encoder.layer_norm2[-1],
        ])
        set_trainable(decoder, True)
    elif scope == "full":
        set_trainable(encoder, True)
        set_trainable(decoder, True)
    else:  # argparse protects this; retain a useful error for direct calls.
        raise ValueError(f"Unsupported scope: {scope}")

    # A tied decoder projection/embedding can expose the same Parameter twice.
    params, seen = [], set()
    for module in (encoder, decoder):
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                params.append(parameter)
                seen.add(id(parameter))
    return params


def encode_source(encoder, x, src_len, grad: bool):
    with torch.set_grad_enabled(grad):
        return encoder("fwd", x=x, lengths=src_len, causal=False).transpose(0, 1)


def generate_batch(env, decoder, src_enc_one, src_len_one, n_samples, temperature,
                   max_len, batch_size, device):
    """Sample in chunks to bound peak generation memory, then pad across chunks."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    chunks, length_chunks = [], []
    decoder.eval()
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            size = min(batch_size, n_samples - start)
            enc_k = src_enc_one.expand(size, -1, -1).contiguous()
            src_len_k = src_len_one.expand(size).contiguous()
            generated, gen_len = decoder.generate(
                enc_k, src_len_k, max_len=max_len, sample_temperature=temperature
            )
            chunks.append(generated)
            length_chunks.append(gen_len)

    max_time = max(chunk.shape[0] for chunk in chunks)
    padded = torch.full(
        (max_time, n_samples), env.pad_index, dtype=torch.long, device=device
    )
    offset = 0
    for chunk in chunks:
        width = chunk.shape[1]
        padded[:chunk.shape[0], offset:offset + width] = chunk
        offset += width
    return padded, torch.cat(length_chunks, dim=0)


def _safe_real_array(value, shape):
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim == 0:
        arr = np.full(shape, arr, dtype=np.complex128)
    arr = np.broadcast_to(arr, shape)
    if not np.all(np.isfinite(arr)) or np.max(np.abs(arr.imag)) > 1e-7:
        raise ValueError("non-finite or complex-valued candidate")
    return np.asarray(arr.real, dtype=np.float64)


def _parameter_draws(symbols: Sequence[sp.Symbol], draws: int):
    if not symbols:
        return [()]
    rng = np.random.RandomState(1729 + len(symbols))
    return [tuple(float(v) for v in row) for row in rng.uniform(-1.7, 1.7, size=(max(1, draws), len(symbols)))]


def numeric_target_mse(env, hyp, problem, xs, target_values, parameter_draws):
    x = env.local_dict["x"]
    symbols = sorted(hyp.free_symbols - {x}, key=lambda s: s.name)
    try:
        fn = sp.lambdify([x] + symbols, hyp, modules=["numpy"])
        values = []
        for params in _parameter_draws(symbols, parameter_draws):
            pred = _safe_real_array(fn(xs, *params), xs.shape)
            relative = (pred - target_values) / np.maximum(np.abs(target_values), 1e-6)
            values.append(float(np.mean(np.square(np.clip(relative, -1e6, 1e6)))))
        result = float(np.mean(values))
        return result if math.isfinite(result) else 1e6
    except Exception:
        return 1e6


def numeric_residual_mse(env, problem, hyp, xs, parameter_draws):
    """Dimensionless residual score for the cleared ODE, evaluated at x in [1, 5]."""
    x = env.local_dict["x"]
    f = env.local_dict["f"]
    symbols = sorted(hyp.free_symbols - {x}, key=lambda s: s.name)
    try:
        terms = [term.subs(f(x), hyp).doit() for term in problem.residual_terms]
        funcs = [sp.lambdify([x] + symbols, term, modules=["numpy"]) for term in terms]
        errors = []
        for params in _parameter_draws(symbols, parameter_draws):
            term_values = [_safe_real_array(fn(xs, *params), xs.shape) for fn in funcs]
            arr = np.stack(term_values, axis=0)
            arr = np.clip(arr, -1e100, 1e100)
            residual = arr.sum(axis=0)
            denominator = np.square(arr).sum(axis=0)
            valid = denominator > 1e-24
            if not np.any(valid):
                return 1e6
            ratios = np.square(residual[valid]) / np.maximum(denominator[valid], 1e-300)
            errors.extend(np.clip(ratios, 0.0, 1e6).tolist())
        return float(np.mean(errors)) if errors else 1e6
    except Exception:
        return 1e6


def numeric_ic_mse(env, hyp, parameter_draws):
    x = env.local_dict["x"]
    symbols = sorted(hyp.free_symbols - {x}, key=lambda s: s.name)
    try:
        y0_expr = hyp.subs(x, 0)
        yp0_expr = sp.diff(hyp, x).subs(x, 0)
        y0_fn = sp.lambdify(symbols, y0_expr, modules=["numpy"])
        yp0_fn = sp.lambdify(symbols, yp0_expr, modules=["numpy"])
        errors = []
        for params in _parameter_draws(symbols, parameter_draws):
            y0 = complex(y0_fn(*params)) if symbols else complex(y0_fn())
            yp0 = complex(yp0_fn(*params)) if symbols else complex(yp0_fn())
            if not (np.isfinite(y0.real) and np.isfinite(y0.imag) and
                    np.isfinite(yp0.real) and np.isfinite(yp0.imag)):
                return 1e6
            if abs(y0.imag) > 1e-7 or abs(yp0.imag) > 1e-7:
                return 1e6
            errors.append((y0.real - 1.0) ** 2 + yp0.real ** 2)
        value = float(np.mean(errors)) if errors else 1e6
        return value if math.isfinite(value) else 1e6
    except Exception:
        return 1e6


def _log_score(mse):
    if not math.isfinite(mse) or mse >= 1e6:
        return -6.0
    return float(np.clip(-math.log10(max(mse, 0.0) + 1e-12), -6.0, 6.0))


def score_candidate_numeric(env, problem, token_ids, xs, target_values, args):
    try:
        hyp = ids_to_sympy(env, token_ids)
        target_mse = numeric_target_mse(env, hyp, problem, xs, target_values, args.parameter_draws)
        residual_mse = numeric_residual_mse(env, problem, hyp, xs, args.parameter_draws)
        ic_mse = numeric_ic_mse(env, hyp, args.parameter_draws)
        reward = (
            args.target_weight * _log_score(target_mse)
            + args.residual_weight * _log_score(residual_mse)
            + args.ic_weight * _log_score(ic_mse)
            - args.length_penalty * len(token_ids)
        )
        return IVPInfo(
            reward=float(reward), valid_parse=True, exact_residual_zero=False,
            generality_rank=0, numerical_mse=float(target_mse + residual_mse + ic_mse),
            expression=hyp, exact_checked=False, target_mse=float(target_mse),
            residual_mse=float(residual_mse), ic_mse=float(ic_mse), token_length=len(token_ids),
        )
    except Exception as exc:
        return IVPInfo(
            reward=-25.0, valid_parse=False, exact_residual_zero=False, generality_rank=0,
            numerical_mse=1e6, expression=None, error=f"{type(exc).__name__}: {exc}",
            exact_checked=True, exact_skipped=True, target_mse=1e6, residual_mse=1e6, ic_mse=1e6,
            token_length=len(token_ids),
        )


def _symbolically_zero(expr, timeout_s):
    try:
        if expr.has(sp.nan, sp.oo, -sp.oo, sp.zoo, sp.I):
            return False
        ok, _ = exact_zero(expr, seconds=timeout_s)
        return bool(ok)
    except Exception:
        return False


def exact_verify_candidate(env, problem, info: IVPInfo, args):
    """Exact check of ODE + y(0)=1 + y'(0)=0, without general-solution rank checks."""
    if info.exact_checked:
        return info
    info.exact_checked = True
    if info.token_length > args.max_exact_len:
        info.exact_skipped = True
        info.error = f"exact verification skipped: length {info.token_length} > --max-exact-len {args.max_exact_len}"
        return info
    hyp = info.expression
    if hyp is None:
        return info
    x = env.local_dict["x"]
    try:
        residual = residual_expression(env, problem.equation, hyp)
        res_ok = _symbolically_zero(residual, args.symbolic_timeout)
        y0_ok = _symbolically_zero(hyp.subs(x, 0) - 1, args.symbolic_timeout)
        yp0_ok = _symbolically_zero(sp.diff(hyp, x).subs(x, 0), args.symbolic_timeout)
        no_free_parameters = len(hyp.free_symbols - {x}) == 0
        info.residual = residual
        info.exact_residual_zero = bool(res_ok)
        info.verified_general = bool(res_ok and y0_ok and yp0_ok and no_free_parameters)
        if info.verified_general:
            info.reward += args.exact_bonus
    except Exception as exc:
        info.error = f"exact verification: {type(exc).__name__}: {exc}"
    return info


def score_rollouts(env, problem, generated, gen_len, xs, target_values, args, cache):
    candidates = generated_to_candidates(env, generated, gen_len)
    infos = []
    hits = misses = 0
    for token_ids, _words in candidates:
        key = tuple(int(t) for t in token_ids)
        if key in cache:
            hits += 1
            info = cache[key]
        else:
            misses += 1
            info = score_candidate_numeric(env, problem, token_ids, xs, target_values, args)
            cache[key] = info
        infos.append(info)
    return candidates, infos, hits, misses


def verify_best(candidates, infos, cache, env, problem, args, top_k):
    """Exact-check top K unique training candidates; top_k=0 means check every unique one."""
    indices = sorted(range(len(infos)), key=lambda i: infos[i].reward, reverse=True)
    ordered_keys, seen = [], set()
    for i in indices:
        key = tuple(int(t) for t in candidates[i][0])
        if key in seen:
            continue
        seen.add(key)
        if not cache[key].exact_checked:
            ordered_keys.append(key)
    limit = len(ordered_keys) if top_k == 0 else min(top_k, len(ordered_keys))
    for key in ordered_keys[:limit]:
        exact_verify_candidate(env, problem, cache[key], args)
    # infos reference the mutable cached objects; updated rewards are visible.


def summarize(candidates, infos, top_k=5):
    order = sorted(range(len(infos)), key=lambda i: infos[i].reward, reverse=True)
    out = []
    for i in order[:top_k]:
        token_ids, words = candidates[i]
        info = infos[i]
        out.append({
            "idx": int(i), "reward": float(info.reward),
            "target_mse": float(info.target_mse), "residual_mse": float(info.residual_mse),
            "ic_mse": float(info.ic_mse),
            "exact_checked": bool(info.exact_checked and not info.exact_skipped),
            "exact_skipped": bool(info.exact_skipped),
            "residual_zero": bool(info.exact_residual_zero) if info.exact_checked else None,
            "exact_ivp": bool(info.verified_general), "length": len(token_ids),
            "expr": str(info.expression) if info.expression is not None else "<parse error>",
            "tokens": " ".join(words), "error": info.error,
        })
    return out


def print_top(rows):
    for row in rows:
        print(
            f"  idx={row['idx']:>3} reward={row['reward']:>8.3f} "
            f"target={row['target_mse']:.2e} residual={row['residual_mse']:.2e} "
            f"IC={row['ic_mse']:.2e} checked={int(row['exact_checked'])} "
            f"skipped={int(row['exact_skipped'])} exact_IVP={int(row['exact_ivp'])} "
            f"len={row['length']:>3} :: {row['expr']}"
        )


def update_elite_buffer(elites, candidates, infos, step, max_elites):
    existing = {tuple(e["ids"]) for e in elites}
    new = []
    for (ids, words), info in zip(candidates, infos):
        if not (info.exact_checked and info.verified_general):
            continue
        key = tuple(int(x) for x in ids)
        if key in existing:
            continue
        elite = {
            "ids": list(key), "words": list(words), "expr": str(info.expression),
            "reward": float(info.reward), "discovered_step": int(step), "length": len(key),
        }
        elites.append(elite)
        new.append(elite)
        existing.add(key)
    elites.sort(key=lambda e: (e["length"], e["discovered_step"]))
    if len(elites) > max_elites:
        kept = {tuple(e["ids"]) for e in elites[:max_elites]}
        new = [e for e in new if tuple(e["ids"]) in kept]
        del elites[max_elites:]
    return new


def _elite_logprobs(env, encoder, decoder, src_x, src_len_one, elites, device,
                    encoder_trainable, frozen_enc):
    generated, lengths = token_ids_to_generated_batch(env, [e["ids"] for e in elites], device)
    if encoder_trainable:
        src_enc = encode_source(encoder, src_x, src_len_one, grad=True)
    else:
        src_enc = frozen_enc
    batch_size = generated.shape[1]
    enc_k = src_enc.expand(batch_size, -1, -1).contiguous()
    src_len = src_len_one.expand(batch_size).contiguous()
    return sequence_logprobs(
        decoder, enc_k, src_len, generated, lengths, length_normalize=False
    )


def run_elite_updates(env, encoder, decoder, src_x, src_len_one, frozen_enc, elites,
                      optimizer, trainable, device, encoder_trainable,
                      n_updates, weight, grad_clip):
    if not elites or n_updates <= 0:
        return None
    values = []
    encoder.eval()
    decoder.eval()
    for _ in range(n_updates):
        lp = _elite_logprobs(
            env, encoder, decoder, src_x, src_len_one, elites, device,
            encoder_trainable, frozen_enc,
        )
        loss = -float(weight) * lp.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        values.append(float(loss.item()))
    return {"updates": int(n_updates), "loss_first": values[0], "loss_last": values[-1]}


def policy_update(env, encoder, decoder, src_x, src_len_one, frozen_enc,
                  generated, gen_len, advantages, optimizer, trainable, device,
                  encoder_trainable, batch_size, length_normalize, grad_clip):
    """Accumulate rollout gradients over microbatches, take one optimizer step."""
    n = generated.shape[1]
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    encoder.eval()
    decoder.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        g = generated[:, start:end]
        gl = gen_len[start:end]
        b = end - start
        if encoder_trainable:
            src_enc = encode_source(encoder, src_x, src_len_one, grad=True)
        else:
            src_enc = frozen_enc
        enc_k = src_enc.expand(b, -1, -1).contiguous()
        slen = src_len_one.expand(b).contiguous()
        sequence_lp = sequence_logprobs(
            decoder, enc_k, slen, g, gl, length_normalize=length_normalize
        )
        loss = -(advantages[start:end].detach() * sequence_lp).sum() / n
        loss.backward()
        loss_total += float(loss.detach().item())
    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
    optimizer.step()
    return loss_total


def greedy_info(env, encoder, decoder, src_x, src_len_one, frozen_enc, encoder_trainable,
                problem, xs, target_values, args, cache, device):
    enc = encode_source(encoder, src_x, src_len_one, grad=False) if encoder_trainable else frozen_enc
    generated, lengths = generate_batch(
        env, decoder, enc, src_len_one, 1, None, args.max_len, 1, device
    )
    candidates, infos, _, _ = score_rollouts(
        env, problem, generated, lengths, xs, target_values, args, cache
    )
    verify_best(candidates, infos, cache, env, problem, args, top_k=0)
    return candidates[0], infos[0]


def fresh_eval(env, encoder, decoder, src_x, src_len_one, frozen_enc, encoder_trainable,
               problem, xs, target_values, args, cache, device, n_samples, temperature):
    if n_samples <= 0:
        return None
    t0 = time.perf_counter()
    enc = encode_source(encoder, src_x, src_len_one, grad=False) if encoder_trainable else frozen_enc
    generated, lengths = generate_batch(
        env, decoder, enc, src_len_one, n_samples, temperature,
        args.max_len, args.sample_batch_size, device,
    )
    generation_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    candidates, infos, hits, misses = score_rollouts(
        env, problem, generated, lengths, xs, target_values, args, cache
    )
    verify_best(candidates, infos, cache, env, problem, args, top_k=0)
    verification_s = time.perf_counter() - t1
    exact = [info.exact_checked and info.verified_general for info in infos]
    rows = summarize(candidates, infos, top_k=1)
    rewards = [info.reward for info in infos]
    return {
        "n": int(n_samples), "successes": int(sum(exact)),
        "exact_rate": float(np.mean(exact)), "best_reward": float(max(rewards)),
        "top_expr": rows[0]["expr"] if rows else None,
        "cache_hits": int(hits), "cache_misses": int(misses),
        "generation_s": float(generation_s), "verification_s": float(verification_s),
    }


def save_checkpoint(path, checkpoint, encoder, decoder):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = dict(checkpoint)
    out["encoder"] = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
    out["decoder"] = {k: v.detach().cpu() for k, v in decoder.state_dict().items()}
    tmp = path + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    if args.steps < 1 or args.rollouts < 2:
        raise ValueError("--steps must be >= 1 and --rollouts must be >= 2")
    if args.sample_batch_size < 1 or args.train_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.n_points < 2 or args.x_max <= args.x_min:
        raise ValueError("Need n_points >= 2 and x_max > x_min")
    if args.temperature <= 0 or args.temperature_end <= 0:
        raise ValueError("Temperatures must be positive")
    if args.symbolic_timeout <= 0:
        raise ValueError("--symbolic-timeout must be positive")

    seed_all(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}")
    print("representation=cleared (x*y'' + 2*y' + x*y^5 = 0); fixed IVP y(0)=1, y'(0)=0")

    env, encoder, decoder, params, checkpoint = load_pretrained(args.checkpoint, device)
    problem = make_problem(env, "lane_emden", n=5, mode="ivp", lane_form="cleared")
    if problem.input_form != "cleared":
        raise RuntimeError("Refusing to train: n=5 must use the cleared Lane-Emden representation")
    src_tokens = equation_to_tokens(env, problem.equation)
    src_x, src_len_one = encode_tokens(env, src_tokens, device)
    x_symbol = env.local_dict["x"]
    xs = np.linspace(args.x_min, args.x_max, args.n_points, dtype=np.float64)
    target_fn = sp.lambdify(x_symbol, problem.expected, modules=["numpy"])
    target_values = np.asarray(target_fn(xs), dtype=np.float64)

    print(f"problem={problem.name}; mode={problem.mode}; input_form={problem.input_form}")
    print(f"equation={problem.equation}")
    print(f"input_prefix={' '.join(src_tokens)}")
    print(f"known_ivp_solution={problem.expected}")
    print(
        f"checkpoint_arch emb={params.emb_dim} enc={params.n_enc_layers} "
        f"dec={params.n_dec_layers} heads={params.n_heads}"
    )

    trainable = configure_scope(encoder, decoder, args.scope)
    encoder_trainable = args.scope in ENCODER_SCOPES
    n_trainable = sum(p.numel() for p in trainable)
    print(f"scope={args.scope}; trainable_parameters={n_trainable:,}")
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    # Deterministic encoder cache for decoder-only scopes. When the encoder is
    # trainable, recompute it per sampling step and per training microbatch.
    frozen_enc = None
    if not encoder_trainable:
        frozen_enc = encode_source(encoder, src_x, src_len_one, grad=False)

    stem = Path(args.checkpoint).stem
    if args.save_checkpoint is None:
        args.save_checkpoint = f"{stem}_ttrl_n5_{args.scope}.pth"
    if args.save_jsonl is None:
        args.save_jsonl = f"ttrl_n5_{args.scope}_results.jsonl"
    eval_temperature = args.temperature_end if args.eval_temperature is None else args.eval_temperature
    cache = {}
    elites = []

    print("\n[baseline greedy]")
    (g_ids, _g_words), ginfo = greedy_info(
        env, encoder, decoder, src_x, src_len_one, frozen_enc, encoder_trainable,
        problem, xs, target_values, args, cache, device,
    )
    print(
        f"reward={ginfo.reward:.3f} exact_checked={ginfo.exact_checked} "
        f"residual_zero={ginfo.exact_residual_zero} exact_ivp={ginfo.verified_general} "
        f"target_mse={ginfo.target_mse:.3e}"
    )
    print(f"expr={ginfo.expression}")

    os.makedirs(os.path.dirname(os.path.abspath(args.save_jsonl)), exist_ok=True)
    with open(args.save_jsonl, "w", encoding="utf-8") as fout:
        if args.eval_rollouts > 0:
            print(f"\n[baseline fresh eval: {args.eval_rollouts} samples]")
            ev = fresh_eval(
                env, encoder, decoder, src_x, src_len_one, frozen_enc, encoder_trainable,
                problem, xs, target_values, args, cache, device,
                args.eval_rollouts, eval_temperature,
            )
            print(
                f"exact_IVP={ev['successes']}/{ev['n']} rate={100*ev['exact_rate']:.3f}% "
                f"best_reward={ev['best_reward']:.3f} "
                f"time(gen={ev['generation_s']:.2f}s verify={ev['verification_s']:.2f}s)"
            )
            fout.write(json.dumps({"event": "baseline_eval", "eval": ev}) + "\n")
            fout.flush()

        for step in range(args.steps):
            step_start = time.perf_counter()
            progress = step / max(args.steps - 1, 1)
            temperature = args.temperature + (args.temperature_end - args.temperature) * progress

            # Sample under the current policy without retaining an inference graph.
            enc_sampling = (
                encode_source(encoder, src_x, src_len_one, grad=False)
                if encoder_trainable else frozen_enc
            )
            t0 = time.perf_counter()
            generated, gen_len = generate_batch(
                env, decoder, enc_sampling, src_len_one, args.rollouts, temperature,
                args.max_len, args.sample_batch_size, device,
            )
            generation_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            candidates, infos, hits, misses = score_rollouts(
                env, problem, generated, gen_len, xs, target_values, args, cache
            )
            verify_best(
                candidates, infos, cache, env, problem, args,
                top_k=args.exact_verify_top_k,
            )
            verification_s = time.perf_counter() - t0

            rewards_np = np.asarray([info.reward for info in infos], dtype=np.float32)
            successes = sum(info.exact_checked and info.verified_general for info in infos)
            top = summarize(candidates, infos, top_k=5)
            print(
                f"\n[step {step + 1:03d}/{args.steps}] T={temperature:.3f} "
                f"reward mean={rewards_np.mean():.3f} std={rewards_np.std():.3f} "
                f"max={rewards_np.max():.3f} exact_IVP_in_batch={successes} "
                f"verified_unique={len({tuple(c[0]) for c, z in zip(candidates, infos) if z.exact_checked})}"
            )
            print_top(top)

            new_elites = update_elite_buffer(elites, candidates, infos, step, args.max_elites)
            if new_elites:
                print(f"  +++ discovered {len(new_elites)} NEW exact IVP solution(s); elite_buffer={len(elites)} +++")
                for elite in new_elites:
                    print(f"      elite len={elite['length']} :: {elite['expr']}")

            # Group-relative REINFORCE, using all rollout rewards but microbatched
            # differentiable log-probability computation to limit peak memory.
            rl_loss = None
            std = float(rewards_np.std())
            rl_start = time.perf_counter()
            if std >= 1e-8:
                advantages = (rewards_np - rewards_np.mean()) / (std + 1e-6)
                advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
                rl_loss = policy_update(
                    env, encoder, decoder, src_x, src_len_one, frozen_enc,
                    generated, gen_len, advantages_t, optimizer, trainable, device,
                    encoder_trainable, args.train_batch_size, args.length_normalize, args.grad_clip,
                )
                print(f"  policy_loss={rl_loss:.6f}")
            else:
                print("  rewards have near-zero variance; skipping policy update")
            rl_s = time.perf_counter() - rl_start

            elite_updates = args.elite_updates if new_elites else args.elite_replay_updates
            elite_start = time.perf_counter()
            elite_stats = run_elite_updates(
                env, encoder, decoder, src_x, src_len_one, frozen_enc, elites,
                optimizer, trainable, device, encoder_trainable, elite_updates,
                args.elite_weight, args.grad_clip,
            )
            elite_s = time.perf_counter() - elite_start
            if elite_stats:
                print(
                    f"  elite_replay updates={elite_stats['updates']} "
                    f"loss={elite_stats['loss_first']:.4f}->{elite_stats['loss_last']:.4f}"
                )

            (greedy_ids, _), greedy_info_value = greedy_info(
                env, encoder, decoder, src_x, src_len_one, frozen_enc, encoder_trainable,
                problem, xs, target_values, args, cache, device,
            )
            print(
                f"  adapted_greedy reward={greedy_info_value.reward:.3f} "
                f"exact_IVP={int(greedy_info_value.verified_general)} "
                f"target_mse={greedy_info_value.target_mse:.2e} :: {greedy_info_value.expression}"
            )

            eval_stats = None
            if (args.eval_rollouts > 0 and args.eval_every > 0 and
                    ((step + 1) % args.eval_every == 0 or bool(new_elites))):
                eval_stats = fresh_eval(
                    env, encoder, decoder, src_x, src_len_one, frozen_enc, encoder_trainable,
                    problem, xs, target_values, args, cache, device,
                    args.eval_rollouts, eval_temperature,
                )
                print(
                    f"  [fresh eval] exact_IVP={eval_stats['successes']}/{eval_stats['n']} "
                    f"rate={100*eval_stats['exact_rate']:.3f}% "
                    f"best={eval_stats['best_reward']:.3f} "
                    f"time(gen={eval_stats['generation_s']:.2f}s verify={eval_stats['verification_s']:.2f}s)"
                )

            total_s = time.perf_counter() - step_start
            print(
                f"  timing: gen={generation_s:.2f}s verify={verification_s:.2f}s "
                f"RL={rl_s:.2f}s elite={elite_s:.2f}s total={total_s:.2f}s "
                f"cache(hit={hits},miss={misses},size={len(cache)})"
            )
            row = {
                "step": step + 1, "temperature": float(temperature),
                "reward_mean": float(rewards_np.mean()), "reward_std": float(rewards_np.std()),
                "reward_max": float(rewards_np.max()), "n_verified_exact_ivp": int(successes),
                "top": top, "elite_buffer_size": len(elites),
                "new_elites": [{k: v for k, v in e.items() if k != "ids"} for e in new_elites],
                "policy_loss": rl_loss, "elite_train": elite_stats,
                "greedy": {
                    "reward": float(greedy_info_value.reward),
                    "exact_checked": bool(greedy_info_value.exact_checked),
                    "residual_zero": bool(greedy_info_value.exact_residual_zero),
                    "exact_ivp": bool(greedy_info_value.verified_general),
                    "target_mse": float(greedy_info_value.target_mse),
                    "expr": str(greedy_info_value.expression),
                },
                "eval": eval_stats,
                "timing": {"generation_s": generation_s, "verification_s": verification_s,
                           "rl_s": rl_s, "elite_s": elite_s, "total_s": total_s},
                "cache": {"hits": hits, "misses": misses, "size": len(cache)},
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    print("\n[final greedy]")
    (_, _), final_info = greedy_info(
        env, encoder, decoder, src_x, src_len_one, frozen_enc, encoder_trainable,
        problem, xs, target_values, args, cache, device,
    )
    print(
        f"reward={final_info.reward:.3f} exact_checked={final_info.exact_checked} "
        f"residual_zero={final_info.exact_residual_zero} exact_IVP={final_info.verified_general}"
    )
    print(f"expr={final_info.expression}")
    print(f"elite_buffer_size={len(elites)}")
    save_checkpoint(args.save_checkpoint, checkpoint, encoder, decoder)
    print(f"saved_checkpoint={args.save_checkpoint}")
    print(f"result_log={args.save_jsonl}")


if __name__ == "__main__":
    main()
