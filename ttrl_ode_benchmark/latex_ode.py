from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import sympy as sp
from sympy.parsing.latex import parse_latex
try:
    from lark import Tree
except Exception:  # pragma: no cover
    Tree = ()


class LatexODEParseError(ValueError):
    pass


def _first_lark_branch(obj):
    """SymPy's Lark backend sometimes returns an ambiguity tree.

    For the elementary expressions used by AGDES/HSE, the first branch is the
    conventional parse (e.g. a*sin(2*x)+b*cos(2*x)).
    """
    if Tree and isinstance(obj, Tree):
        if not obj.children:
            raise LatexODEParseError("empty Lark ambiguity tree")
        return _first_lark_branch(obj.children[0])
    return obj


def _strip_tex_noise(s: str) -> str:
    s = str(s or '').strip().replace('$', '')
    # spacing / sizing commands
    for t in (r'\left', r'\right'):
        s = s.replace(t, '')
    for t in (r'\,', r'\;', r'\!', r'\quad', r'\qquad', r'\:', r'\ '):
        s = s.replace(t, ' ')
    # common aliases in the HSE textbook / AGDES lineage
    s = s.replace(r'\mathrm{tg}', r'\tan').replace(r'\operatorname{tg}', r'\tan')
    s = s.replace(r'\operatorname{tan}', r'\tan')
    s = s.replace(r'\operatorname{sh}', r'\sinh').replace(r'\operatorname{ch}', r'\cosh')
    s = s.replace(r'\ln', r'\log')
    s = s.replace(r'\cdot', ' ')
    # Lark needs an explicit multiplication sign in patterns like x^2 e^{3x}.
    s = re.sub(r'(\^\{[^{}]+\}|\^\d+)\s+e\^', r'\1 \\cdot e^', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _parse_latex_expr(s: str) -> sp.Expr:
    s = _strip_tex_noise(s)
    try:
        out = _first_lark_branch(parse_latex(s, backend='lark'))
    except BaseException as e:
        raise LatexODEParseError(f"LaTeX parse failed: {type(e).__name__}: {e}") from e
    if not isinstance(out, sp.Basic):
        raise LatexODEParseError(f"LaTeX parser returned {type(out).__name__}, not a SymPy expression")
    # Bare `e` in this corpus means Euler's number.  Lark parses it as a Symbol.
    for sym in list(out.free_symbols):
        if str(sym) == 'e':
            out = out.subs(sym, sp.E)
    return out


def _drop_trailing_conditions(equation_latex: str) -> str:
    """Keep only the differential equation before an IC/BC suffix.

    We still reject IVP/BVP rows for the exact-general benchmark; this helper
    prevents a comma in a textual suffix from confusing equation parsing.
    """
    s = str(equation_latex)
    # A generated general equation normally contains no comma.  Textbook rows
    # often append `, y(0)=...` or `, y'(0)=...`.
    parts = s.split(',')
    if len(parts) > 1:
        tail = ','.join(parts[1:])
        if re.search(r'\by\s*(?:\\prime\s*)*\(', tail) or re.search(r'\by\s*\(', tail):
            return parts[0]
    return s


def has_conditions(equation_latex: str) -> bool:
    s = str(equation_latex or '')
    # explicit initial/boundary values: y(0), y'(0), y(1), ...
    return bool(
        re.search(r'\by\s*\([^)]*\)\s*=', s)
        or re.search(r'y\\prime(?:\\prime)?\s*\([^)]*\)\s*=', s)
        or re.search(r'y\^\{\\prime(?:\\prime)?\}\s*\([^)]*\)\s*=', s)
    )


def detect_order(equation_latex: str) -> int:
    s = str(equation_latex or '')
    if re.search(r'y(?:\\prime){3}', s) or re.search(r'y\^\{(?:\\prime){3}\}', s) or "y'''" in s:
        return 3
    if (r'y\prime\prime' in s or r'y^{\prime\prime}' in s or "y''" in s
            or re.search(r'\\frac\s*\{d\^?\{?2\}?\s*y\}\s*\{d\s*x\^?\{?2\}?\}', s)):
        return 2
    if r'y\prime' in s or r'y^{\prime}' in s or "y'" in s or r'\frac{dy}{dx}' in s:
        return 1
    return 0


def _replace_derivatives(s: str) -> str:
    s = _strip_tex_noise(s)
    # Fractions first.
    s = re.sub(r'\\frac\s*\{d\^?\{?2\}?\s*y\}\s*\{d\s*x\^?\{?2\}?\}', 'q', s)
    s = s.replace(r'\frac{d^2y}{dx^2}', 'q').replace(r'\frac{d^{2}y}{dx^{2}}', 'q')
    s = s.replace(r'\frac{dy}{dx}', 'p')
    # Prime notation used by the HSE CSVs.
    s = s.replace(r'y\prime\prime', 'q').replace(r'y^{\prime\prime}', 'q').replace("y''", 'q')
    s = s.replace(r'y\prime', 'p').replace(r'y^{\prime}', 'p').replace("y'", 'p')
    # Remaining dependent variable -> z placeholder.
    s = re.sub(r'(?<![A-Za-z])y(?![A-Za-z])', 'z', s)
    return s


def parse_ode2_equation(equation_latex: str, env) -> sp.Expr:
    """Parse a scalar second-order HSE/AGDES LaTeX ODE into Lample's f(x) form."""
    if detect_order(equation_latex) != 2:
        raise LatexODEParseError("not a second-order scalar ODE")
    core = _drop_trailing_conditions(equation_latex)
    s = _replace_derivatives(core)
    if '=' not in s:
        raise LatexODEParseError("equation has no '='")
    lhs_s, rhs_s = s.split('=', 1)
    lhs = _parse_latex_expr(lhs_s)
    rhs = _parse_latex_expr(rhs_s)
    expr = sp.expand(lhs - rhs)

    x = env.local_dict['x']
    f = env.local_dict['f']
    # Lark creates its own Symbol('x'); canonicalize by name to env.local_dict['x'].
    for sym in list(expr.free_symbols):
        if str(sym) == 'x' and sym != x:
            expr = expr.subs(sym, x)
    symbols = {str(v): v for v in expr.free_symbols}
    subs = {}
    if 'q' in symbols:
        subs[symbols['q']] = sp.diff(f(x), x, 2)
    if 'p' in symbols:
        subs[symbols['p']] = sp.diff(f(x), x)
    if 'z' in symbols:
        subs[symbols['z']] = f(x)
    expr = sp.expand(expr.subs(subs))

    # Canonicalize accidental independent-variable t -> x for simple generated rows.
    t_syms = [v for v in expr.free_symbols if str(v) == 't']
    if t_syms and x not in expr.free_symbols:
        expr = expr.subs(t_syms[0], x)

    # No unknown symbols beyond x are allowed in the equation itself.
    allowed = {x}
    bad = [s for s in expr.free_symbols if s not in allowed]
    if bad:
        raise LatexODEParseError(f"unsupported free symbols in equation: {[str(s) for s in bad]}")
    if not expr.has(sp.diff(f(x), x, 2)):
        raise LatexODEParseError("parsed expression lost y''")
    return expr


def _strip_solution_lhs(answer_latex: str) -> str:
    s = str(answer_latex or '').strip().replace('$', '')
    if '=' not in s:
        return s
    lhs, rhs = s.split('=', 1)
    lhs_compact = re.sub(r'\s+', '', lhs)
    if lhs_compact in {'y', 'y(x)', 'f(x)'}:
        return rhs
    # Some rows put an intermediate transform before the final equality; choose
    # the final RHS only when the first lhs clearly denotes y.
    if lhs_compact.startswith('y'):
        return rhs
    return s


def parse_reference_solution(answer_latex: str, env, order: int = 2) -> sp.Expr:
    s = _strip_solution_lhs(answer_latex)
    out = _parse_latex_expr(s)

    # Canonicalize x/t and Euler e.
    x = env.local_dict['x']
    for sym in list(out.free_symbols):
        if str(sym) == 'x' and sym != x:
            out = out.subs(sym, x)
        elif str(sym) == 't':
            out = out.subs(sym, x)
        elif str(sym) == 'e':
            out = out.subs(sym, sp.E)

    # Map arbitrary integration-constant spellings C_1, c_1, C_{1}, etc. to
    # the model's a0, a1, ... symbols.  Do not map x.
    consts = []
    for sym in out.free_symbols:
        name = str(sym)
        if re.match(r'^[Cc](?:_?\{?\d+\}?)?$', name) or re.match(r'^[Cc]_?\{?\d+\}?$', name):
            consts.append(sym)
    consts = sorted(set(consts), key=lambda s: str(s))
    if len(consts) < order:
        raise LatexODEParseError(f"reference has only {len(consts)} integration constants; expected >= {order}")
    coeffs = [env.local_dict[f'a{i}'] for i in range(min(len(consts), 10))]
    out = out.subs({c: a for c, a in zip(consts, coeffs)})

    bad = [s for s in out.free_symbols if s != x and s not in set(env.coefficients.values())]
    if bad:
        raise LatexODEParseError(f"unsupported free symbols in solution: {[str(s) for s in bad]}")
    return out


@dataclass
class ParsedPair:
    equation: sp.Expr
    reference: sp.Expr
