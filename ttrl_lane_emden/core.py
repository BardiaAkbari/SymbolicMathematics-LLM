import copy
import math
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
import torch
import torch.nn.functional as F

from src.envs.char_sp import CharSPEnvironment, InvalidPrefixExpression
from src.envs.sympy_utils import simplify as timed_simplify
from src.model.transformer import TransformerModel


@dataclass
class Problem:
    name: str
    equation: sp.Expr
    order: int
    mode: str = "general"
    expected: Optional[sp.Expr] = None
    residual_terms: Optional[Tuple[sp.Expr, ...]] = None
    input_form: str = "standard"


@dataclass
class RewardInfo:
    reward: float
    valid_parse: bool
    exact_residual_zero: bool
    generality_rank: int
    numerical_mse: float
    expression: Optional[sp.Expr]
    residual: Optional[sp.Expr]
    verified_general: bool = False
    error: Optional[str] = None


def trusted_torch_load(path: str, map_location="cpu"):
    """Load an official/trusted Lample--Charton checkpoint on old or new PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def load_pretrained(checkpoint_path: str, device: torch.device):
    ckpt = trusted_torch_load(checkpoint_path, map_location="cpu")
    if "params" not in ckpt or "encoder" not in ckpt or "decoder" not in ckpt:
        raise ValueError("Checkpoint must contain params, encoder, and decoder.")

    params_dict = dict(ckpt["params"])
    # Fields below are irrelevant to direct inference, but older checkpoints may omit them.
    params_dict.setdefault("max_len", 512)
    params_dict.setdefault("clean_prefix_expr", True)
    params_dict.setdefault("rewrite_functions", "")
    params = Namespace(**params_dict)

    env = CharSPEnvironment(params)
    encoder = TransformerModel(params, env.id2word, is_encoder=True, with_output=False)
    decoder = TransformerModel(params, env.id2word, is_encoder=False, with_output=True)
    encoder.load_state_dict(_strip_module_prefix(ckpt["encoder"]), strict=True)
    decoder.load_state_dict(_strip_module_prefix(ckpt["decoder"]), strict=True)
    encoder.to(device).eval()
    decoder.to(device).eval()
    return env, encoder, decoder, params, ckpt


def default_env_for_verifier() -> CharSPEnvironment:
    """Environment matching the public README generator settings; no checkpoint required."""
    p = Namespace(
        max_int=5,
        max_ops=15,
        max_ops_G=4,
        int_base=10,
        balanced=False,
        positive=True,
        precision=10,
        n_variables=1,
        n_coefficients=0,
        max_len=512,
        clean_prefix_expr=True,
        operators=(
            "add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,"
            "ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,"
            "tanh:1,asinh:1,acosh:1,atanh:1"
        ),
        rewrite_functions="",
        leaf_probs="0.75,0,0.25,0",
    )
    return CharSPEnvironment(p)


def make_problem(env: CharSPEnvironment, name: str, n: int = 1, mode: str = "general",
                 lane_form: str = "cleared") -> Problem:
    x = env.local_dict["x"]
    f = env.local_dict["f"]
    a0 = env.local_dict["a0"]
    a1 = env.local_dict["a1"]

    if name == "harmonic":
        d2 = sp.diff(f(x), x, 2)
        y = f(x)
        eq = d2 + y
        expected = a0 * sp.sin(x) + a1 * sp.cos(x)
        return Problem("harmonic", eq, order=2, mode="general", expected=expected,
                       residual_terms=(d2, y), input_form="standard")

    if name != "lane_emden":
        raise ValueError(f"Unknown problem: {name}")
    if lane_form not in {"standard", "cleared"}:
        raise ValueError("lane_form must be 'standard' or 'cleared'")

    y = f(x)
    yp = sp.diff(y, x)
    ypp = sp.diff(y, x, 2)
    nonlinear = y ** int(n)

    # Both equations are equivalent for x != 0.  The cleared form is much closer
    # to the algebraic forms seen by the Lample--Charton ODE generator and avoids
    # presenting the model with an explicit x**-1 factor.
    if lane_form == "standard":
        terms = (ypp, 2 * yp / x, nonlinear)
    else:
        terms = (x * ypp, 2 * yp, x * nonlinear)
    eq = sp.Add(*terms)

    expected = None
    if n == 0:
        expected = a0 + a1 / x - x**2 / 6 if mode == "general" else 1 - x**2 / 6
    elif n == 1:
        expected = (a0 * sp.sin(x) + a1 * sp.cos(x)) / x if mode == "general" else sp.sin(x) / x
    elif n == 5 and mode == "ivp":
        expected = 1 / sp.sqrt(1 + x**2 / 3)
    return Problem(f"lane_emden_n{n}_{lane_form}", eq, order=2, mode=mode, expected=expected,
                   residual_terms=terms, input_form=lane_form)


def equation_to_tokens(env: CharSPEnvironment, equation: sp.Expr) -> List[str]:
    prefix = env.sympy_to_prefix(equation)
    prefix = env.clean_prefix(prefix)
    missing = [w for w in prefix if w not in env.word2id]
    if missing:
        raise ValueError(f"Equation contains tokens absent from model vocabulary: {missing}")
    return prefix


def encode_tokens(env: CharSPEnvironment, tokens: Sequence[str], device: torch.device):
    ids = torch.LongTensor([env.word2id[w] for w in tokens])
    x, lengths = env.batch_sequences([ids])
    return x.to(device), lengths.to(device)


def ids_to_sympy(env: CharSPEnvironment, ids: Sequence[int]) -> sp.Expr:
    prefix = [env.id2word[int(i)] for i in ids]
    prefix = env.unclean_prefix(prefix)
    infix = env.prefix_to_infix(prefix)
    return sp.S(infix, locals=env.local_dict)


def generated_to_candidates(env, generated: torch.Tensor, lengths: torch.Tensor):
    candidates = []
    for i in range(generated.shape[1]):
        L = int(lengths[i].item())
        token_ids = generated[1:L - 1, i].detach().cpu().tolist()
        token_words = [env.id2word[t] for t in token_ids]
        candidates.append((token_ids, token_words))
    return candidates


def sample_rollouts(env, encoder, decoder, equation: sp.Expr, n_samples: int, temperature: float,
                    max_len: int, device: torch.device):
    src_tokens = equation_to_tokens(env, equation)
    x, lengths = encode_tokens(env, src_tokens, device)
    with torch.no_grad():
        enc = encoder("fwd", x=x, lengths=lengths, causal=False).transpose(0, 1)  # (1, slen, dim)
        enc_k = enc.expand(n_samples, -1, -1).contiguous()
        len_k = lengths.expand(n_samples).contiguous()
        generated, gen_len = decoder.generate(
            enc_k, len_k, max_len=max_len, sample_temperature=temperature
        )
    return src_tokens, generated, gen_len


def _coefficient_symbols(env, expr: sp.Expr) -> List[sp.Symbol]:
    coeff_set = set(env.coefficients.values())
    return sorted([s for s in expr.free_symbols if s in coeff_set], key=lambda s: s.name)


def residual_expression(env: CharSPEnvironment, equation: sp.Expr, hyp: sp.Expr) -> sp.Expr:
    x = env.local_dict["x"]
    f = env.local_dict["f"]
    return equation.subs(f(x), hyp).doit()


def exact_zero(expr: sp.Expr, seconds: int = 2) -> Tuple[bool, sp.Expr]:
    try:
        s = timed_simplify(expr, seconds=seconds)
        if s == 0:
            return True, s
        # together/cancel catches many rational identities cheaply.
        try:
            s2 = sp.cancel(sp.together(s))
            s2 = timed_simplify(s2, seconds=seconds)
            return bool(s2 == 0), s2
        except Exception:
            return False, s
    except BaseException:
        return False, expr


def numeric_generality_rank(env: CharSPEnvironment, hyp: sp.Expr, order: int,
                            x_values=(0.37, 0.83, 1.41, 2.17)) -> int:
    """Max numeric rank of the jet Jacobian d(h,h',...)/d(constants)."""
    x = env.local_dict["x"]
    coeffs = _coefficient_symbols(env, hyp)
    if not coeffs:
        return 0
    rows = [sp.diff(hyp, x, k) for k in range(order)]
    J = sp.Matrix([[sp.diff(row, c) for c in coeffs] for row in rows])
    max_rank = 0
    for xv in x_values:
        # Give every coefficient a deterministic nonzero value.
        subs = {x: xv}
        subs.update({c: 0.7 + 0.31 * (j + 1) for j, c in enumerate(coeffs)})
        try:
            arr = np.array(J.subs(subs).evalf(), dtype=np.complex128)
            if not np.all(np.isfinite(arr)):
                continue
            rank = int(np.linalg.matrix_rank(arr, tol=1e-8))
            max_rank = max(max_rank, rank)
        except Exception:
            continue
    return min(max_rank, order)


def exact_generality(env: CharSPEnvironment, hyp: sp.Expr, order: int, seconds: int = 1) -> bool:
    """For ODE1/ODE2, check whether the candidate contains order independent constants."""
    x = env.local_dict["x"]
    coeffs = _coefficient_symbols(env, hyp)
    if len(coeffs) < order:
        return False
    if order == 1:
        for c in coeffs:
            z, _ = exact_zero(sp.diff(hyp, c), seconds=seconds)
            if not z:
                return True
        return False
    if order == 2:
        hp = sp.diff(hyp, x)
        for i in range(len(coeffs)):
            for j in range(i + 1, len(coeffs)):
                c1, c2 = coeffs[i], coeffs[j]
                det = sp.diff(hyp, c1) * sp.diff(hp, c2) - sp.diff(hyp, c2) * sp.diff(hp, c1)
                z, _ = exact_zero(det, seconds=seconds)
                if not z:
                    return True
        return False
    # We only need ODE1/ODE2 for this project.
    return numeric_generality_rank(env, hyp, order) >= order


def numerical_residual_mse(env: CharSPEnvironment, residual: sp.Expr,
                           x_min: float = 0.25, x_max: float = 4.0,
                           n_points: int = 16, n_coeff_draws: int = 3,
                           seed: int = 0) -> float:
    """Legacy absolute residual MSE (kept for diagnostics)."""
    x = env.local_dict["x"]
    coeffs = _coefficient_symbols(env, residual)
    xs = np.linspace(x_min, x_max, n_points, dtype=np.float64)
    rng = np.random.RandomState(seed)
    try:
        fn = sp.lambdify([x] + coeffs, residual, modules=["numpy"])
        vals_all = []
        draws = max(1, n_coeff_draws if coeffs else 1)
        for _ in range(draws):
            cvals = rng.uniform(-1.7, 1.7, size=len(coeffs)).tolist()
            vals = fn(xs, *cvals)
            vals = np.asarray(vals, dtype=np.complex128)
            if vals.ndim == 0:
                vals = np.full_like(xs, vals, dtype=np.complex128)
            vals = np.broadcast_to(vals, xs.shape)
            if not np.all(np.isfinite(vals)):
                return 1e12
            if np.max(np.abs(vals.imag)) > 1e-7:
                return 1e12
            vals_all.append(vals.real)
        vals = np.concatenate(vals_all)
        mse = float(np.mean(np.square(np.clip(vals, -1e6, 1e6))))
        return mse if math.isfinite(mse) else 1e12
    except Exception:
        return 1e12


def relative_residual_mse(env: CharSPEnvironment, problem: Problem, hyp: sp.Expr,
                          x_min: float = 0.25, x_max: float = 4.0,
                          n_points: int = 24, n_coeff_draws: int = 4,
                          seed: int = 0) -> float:
    """Scale/reparameterization-resistant residual for reward shaping.

    For each collocation point we score
        |sum_j T_j[h]|^2 / sum_j |T_j[h]|^2
    where T_j are the ODE terms.  This prevents a wrong family from obtaining an
    arbitrarily good reward merely by multiplying the whole expression by a tiny
    constant (e.g. exp(-5)), which is especially important when the output has
    arbitrary integration constants.
    """
    x = env.local_dict["x"]
    f = env.local_dict["f"]
    if problem.residual_terms is None:
        raw_terms = tuple(sp.Add.make_args(problem.equation))
    else:
        raw_terms = problem.residual_terms
    terms = [t.subs(f(x), hyp).doit() for t in raw_terms]
    coeffs = _coefficient_symbols(env, hyp)
    xs = np.linspace(x_min, x_max, n_points, dtype=np.float64)
    rng = np.random.RandomState(seed)

    try:
        fns = [sp.lambdify([x] + coeffs, t, modules=["numpy"]) for t in terms]
        ratios = []
        draws = max(1, n_coeff_draws if coeffs else 1)
        for _ in range(draws):
            cvals = rng.uniform(-1.7, 1.7, size=len(coeffs)).tolist()
            term_vals = []
            for fn in fns:
                v = np.asarray(fn(xs, *cvals), dtype=np.complex128)
                if v.ndim == 0:
                    v = np.full(xs.shape, v, dtype=np.complex128)
                v = np.broadcast_to(v, xs.shape)
                if not np.all(np.isfinite(v)) or np.max(np.abs(v.imag)) > 1e-7:
                    return 1e12
                term_vals.append(v.real)
            arr = np.stack(term_vals, axis=0)
            arr = np.clip(arr, -1e100, 1e100)
            res = np.sum(arr, axis=0)
            num = np.square(res)
            den = np.sum(np.square(arr), axis=0)
            # If every term is numerically zero, this sample provides no useful
            # dense signal.  Exact symbolic zero is handled separately.
            valid = den > 1e-24
            if not np.any(valid):
                return 1e12
            ratios.extend((num[valid] / np.maximum(den[valid], 1e-300)).tolist())
        if not ratios:
            return 1e12
        value = float(np.mean(np.clip(np.asarray(ratios), 0.0, 1e12)))
        return value if math.isfinite(value) else 1e12
    except Exception:
        return 1e12


def score_general_candidate(env: CharSPEnvironment, problem: Problem, hyp: sp.Expr,
                            exact_bonus: float = 20.0, rank_weight: float = 6.0) -> RewardInfo:
    try:
        residual = residual_expression(env, problem.equation, hyp)
        is_zero, residual_s = exact_zero(residual, seconds=2)
        # IMPORTANT: use a relative residual, not absolute MSE.  Absolute MSE is
        # reward-hackable by multiplying a wrong two-constant family by exp(-K).
        mse = 0.0 if is_zero else relative_residual_mse(env, problem, hyp)
        rank = numeric_generality_rank(env, hyp, problem.order)

        # Dense shaping from the *relative* residual.  Exact correctness is still
        # determined symbolically below.
        residual_score = float(np.clip(-math.log10(mse + 1e-12), -6.0, 6.0))
        rank_term = rank_weight * (rank - problem.order)  # 0 when complete, negative otherwise
        reward = residual_score + rank_term

        full_exact = False
        if is_zero and rank >= problem.order:
            full_exact = exact_generality(env, hyp, problem.order)
            if full_exact:
                reward += exact_bonus

        return RewardInfo(
            reward=float(reward), valid_parse=True,
            exact_residual_zero=bool(is_zero), generality_rank=int(rank),
            numerical_mse=float(mse), expression=hyp, residual=residual_s,
            verified_general=bool(full_exact),
        )
    except BaseException as e:
        return RewardInfo(-25.0, False, False, 0, 1e12, hyp, None, False, f"{type(e).__name__}: {e}")


def score_candidate_ids(env: CharSPEnvironment, problem: Problem, token_ids: Sequence[int]) -> RewardInfo:
    try:
        hyp = ids_to_sympy(env, token_ids)
    except BaseException as e:
        return RewardInfo(-25.0, False, False, 0, 1e12, None, None, False, f"{type(e).__name__}: {e}")
    if problem.mode != "general":
        raise NotImplementedError("The first POC intentionally implements general-solution TTRL first.")
    return score_general_candidate(env, problem, hyp)


def sequence_logprobs(decoder, src_enc: torch.Tensor, src_len: torch.Tensor,
                      generated: torch.Tensor, gen_len: torch.Tensor,
                      length_normalize: bool = True):
    """Teacher-force sampled sequences and return differentiable per-sequence log P."""
    slen, bs = generated.shape
    decoded = decoder(
        "fwd", x=generated, lengths=gen_len, causal=True,
        src_enc=src_enc, src_len=src_len
    )
    logits = decoder.proj(decoded[:-1])  # positions 0..L-2 predict 1..L-1
    targets = generated[1:]
    logp = F.log_softmax(logits.float(), dim=-1)
    token_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    positions = torch.arange(slen - 1, device=generated.device)[:, None]
    mask = positions < (gen_len[None, :] - 1)
    token_lp = token_lp * mask
    seq_lp = token_lp.sum(0)
    if length_normalize:
        seq_lp = seq_lp / (gen_len - 1).clamp_min(1).float()
    return seq_lp


def configure_trainable(decoder, scope: str):
    for p in decoder.parameters():
        p.requires_grad_(False)

    if scope == "proj":
        for p in decoder.proj.parameters():
            p.requires_grad_(True)
    elif scope == "last_layer":
        modules = [
            decoder.attentions[-1], decoder.layer_norm1[-1], decoder.layer_norm15[-1],
            decoder.encoder_attn[-1], decoder.ffns[-1], decoder.layer_norm2[-1], decoder.proj,
        ]
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(True)
    elif scope == "decoder":
        for p in decoder.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError("scope must be one of: proj, last_layer, decoder")

    # shared proj/embedding weights can make embeddings trainable when share_inout_emb=True.
    params = []
    seen = set()
    for p in decoder.parameters():
        if p.requires_grad and id(p) not in seen:
            params.append(p)
            seen.add(id(p))
    return params


def evaluate_rollout_batch(env, problem, generated, gen_len):
    candidates = generated_to_candidates(env, generated, gen_len)
    infos = [score_candidate_ids(env, problem, ids) for ids, _ in candidates]
    return candidates, infos


def summarize_infos(candidates, infos, top_k=5):
    order = sorted(range(len(infos)), key=lambda i: infos[i].reward, reverse=True)
    rows = []
    for i in order[:top_k]:
        ids, words = candidates[i]
        info = infos[i]
        rows.append({
            "idx": i,
            "reward": info.reward,
            "mse": info.numerical_mse,
            "rank": info.generality_rank,
            "res0": info.exact_residual_zero,
            "verified_general": info.verified_general,
            "expr": str(info.expression) if info.expression is not None else "<parse error>",
            "tokens": " ".join(words),
        })
    return rows


def token_ids_to_generated_batch(env: CharSPEnvironment, token_sequences: Sequence[Sequence[int]],
                                 device: torch.device):
    """Rebuild generated tensors (<EOS> tokens <EOS>) from raw candidate token IDs."""
    if len(token_sequences) == 0:
        raise ValueError("token_sequences must be non-empty")
    seqs = [torch.LongTensor(list(s)) for s in token_sequences]
    generated, lengths = env.batch_sequences(seqs)
    return generated.to(device), lengths.to(device)


def token_sequence_logprobs(env: CharSPEnvironment, decoder, src_enc_1: torch.Tensor,
                            src_len_1: torch.Tensor, token_sequences: Sequence[Sequence[int]],
                            device: torch.device, length_normalize: bool = False):
    """Differentiable log P for arbitrary saved candidate token sequences."""
    generated, gen_len = token_ids_to_generated_batch(env, token_sequences, device)
    bs = generated.shape[1]
    enc_k = src_enc_1.expand(bs, -1, -1).contiguous()
    src_len = src_len_1.expand(bs).contiguous()
    return sequence_logprobs(
        decoder, enc_k, src_len, generated, gen_len,
        length_normalize=length_normalize,
    )


def evaluate_rollout_batch_cached(env, problem, generated, gen_len, cache=None):
    """Evaluate candidates while reusing verifier results for repeated token sequences.

    SymPy verification dominates runtime in this POC.  Generated sequences repeat often,
    so a per-problem cache avoids paying for identical symbolic work more than once.
    """
    candidates = generated_to_candidates(env, generated, gen_len)
    if cache is None:
        cache = {}
    infos = []
    hits = 0
    misses = 0
    for ids, _ in candidates:
        key = tuple(int(x) for x in ids)
        if key in cache:
            info = cache[key]
            hits += 1
        else:
            info = score_candidate_ids(env, problem, ids)
            cache[key] = info
            misses += 1
        infos.append(info)
    return candidates, infos, hits, misses
