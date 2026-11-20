"""Parser for the verified subset of ``mgcv`` formula syntax — preview epic Slice P3
(``docs/PLAN_gam_parity_preview.md``, ADR-251).

Parses a formula **string** into :class:`ParsedFormula`, a list of inert term records
(:class:`SmoothCall`, :class:`TensorCall`, :class:`ParametricCall`). It builds nothing
numeric and does not look at data: whether a name is a factor or a number is decided
later by :mod:`hgam.frontend.api`, which holds the DataFrame.

**The grammar is the verified subset, nothing wider** (``MGCV_FEATURE_COVERAGE.md``)::

    response ~ term (+ term)*          response := column | column / column
    term     := s(x [, bs="tp"] [, k=K])  |  s(x, bs="cr", k=K [, by=z])  |  s(f, bs="re")
              | s(f, x, bs="sz", k=K, xt=list(bs="cr"))
              | s(x, f, bs="fs", k=K, xt="cr")
              | ti(x, z, bs="cr" [, k=K | c(K1, K2)])
              | offset(column) | factor | factor:factor

Anything else raises :class:`~hgam.errors.HgamValidationError` whose
message names the construct and the coverage row that explains the refusal.
**A bare ``s(x)`` is ``tp``, as in ``mgcv``** (Slice R4, ADR-261; before it, the parser
refused a bare ``s(x)`` rather than map it to ``cr``, maintainer decision 2026-10-06, because
the oracle defines the default and ``tp`` was unverified). ``tp`` is accepted for a single
covariate, ``k`` optional (default 10), with no ``by=``; ``ti()`` still needs ``bs="cr"``.
"""

import re
from dataclasses import dataclass, field

from hgam.errors import HgamNotImplementedError, HgamValidationError

__all__ = [
    "FsCall",
    "ParametricCall",
    "ParsedFormula",
    "SmoothCall",
    "SzCall",
    "TensorCall",
    "parse_formula",
]

_IDENT = re.compile(r"^[A-Za-z_.][A-Za-z0-9_.]*$")
_INT = re.compile(r"^[0-9]+$")
_STRING = re.compile(r"""^(["'])(.*)\1$""")

_CR_DEFAULT_K = 10
"""``mgcv``'s default ``k`` for a ``cr`` smooth (``k = 10``)."""
_TP_DEFAULT_K = 10
"""``mgcv``'s default ``k`` for a one-covariate ``tp`` smooth: ``M + 8`` with the null-space
dimension ``M = 2`` (``smooth.construct.tp.smooth.spec``'s ``def.k``)."""
_TI_DEFAULT_K = 5
"""``mgcv``'s default ``k`` per margin of a ``ti`` smooth (``k = 5``)."""

_COVERAGE = "docs/MGCV_FEATURE_COVERAGE.md"

_REFUSED_BASES: dict[str, str] = {
    "ts": f"{_COVERAGE} §2.1 / ladder rung L7 (shrinkage thin-plate; only plain tp is verified)",
    "cs": f"{_COVERAGE} §2.1 (shrinkage cr; not in the target form)",
    "cc": f"{_COVERAGE} §2.1 (cyclic cubic; not in the target form)",
    "cp": f"{_COVERAGE} §2.1 (cyclic P-spline; not in the target form)",
    "ps": f"{_COVERAGE} §2.1 (P-spline; not in the target form)",
    "ds": f"{_COVERAGE} §2.1 (Duchon spline; not in the target form)",
    "gp": f"{_COVERAGE} §2.1 (Gaussian process; not in the target form)",
    "mrf": f"{_COVERAGE} §2.1 (Markov random field; not in the target form)",
    "so": f"{_COVERAGE} §2.1 (soap film; not in the target form)",
    "sos": f"{_COVERAGE} §2.1 (sphere; not in the target form)",
}

_SZ_XT = re.compile(r"""^list\(\s*bs\s*=\s*(["'])cr\1\s*\)$""")
"""The one ``xt`` an ``sz`` smooth accepts: ``list(bs="cr")``, a cr base smooth (the exit test's form)."""

_FS_XT = re.compile(r"""^(?:(["'])cr\1|list\(\s*bs\s*=\s*(["'])cr\2\s*\))$""")
"""The ``xt`` an ``fs`` smooth accepts: ``"cr"`` or ``list(bs="cr")``, a cr base smooth
(``smooth.construct.fs.smooth.spec`` reads either form)."""


@dataclass(frozen=True)
class SmoothCall:
    """A parsed ``s(variable, bs=..., k=..., by=...)`` call."""

    variable: str
    basis: str
    k: int | None
    by: str | None = None

    @property
    def text(self) -> str:
        return f"s({self.variable})"


@dataclass(frozen=True)
class SzCall:
    """A parsed ``s(f, x, bs="sz", k=K, xt=list(bs="cr"))`` call: a factor and the smoothed covariate, in
    written order. Which name is the factor is checked against the data by :mod:`hgam.frontend.api`."""

    variables: tuple[str, str]
    k: int

    @property
    def text(self) -> str:
        return f"s({','.join(self.variables)})"


@dataclass(frozen=True)
class FsCall:
    """A parsed ``s(x, f, bs="fs", k=K, xt="cr")`` call: two variables in written order, one of which is a
    factor. mgcv finds the factor by type (``smooth.construct.fs.smooth.spec``), so
    :mod:`hgam.frontend.api` decides which against the data."""

    variables: tuple[str, str]
    k: int

    @property
    def text(self) -> str:
        return f"s({','.join(self.variables)})"


@dataclass(frozen=True)
class TensorCall:
    """A parsed ``ti(x, z, bs="cr", k=...)`` or ``te(x, z, bs="cr", k=...)`` call (two margins).
    ``kind`` is the function written, ``"ti"`` or ``"te"``."""

    variables: tuple[str, ...]
    k: tuple[int, ...]
    kind: str = "ti"

    @property
    def text(self) -> str:
        return f"{self.kind}({','.join(self.variables)})"


@dataclass(frozen=True)
class ParametricCall:
    """A parsed parametric term: one factor (``a``) or a two-factor interaction
    (``a:b``). Whether the names really are factors is decided against the data."""

    variables: tuple[str, ...]

    @property
    def text(self) -> str:
        return ":".join(self.variables)


type TermCall = SmoothCall | SzCall | FsCall | TensorCall | ParametricCall


@dataclass(frozen=True)
class ParsedFormula:
    """A parsed formula: the response column, the terms in written order, and the
    offset column named by ``offset(.)`` if any."""

    response: str
    terms: tuple[TermCall, ...]
    offset: str | None = None
    response_denominator: str | None = None
    """The ``b`` of a response written ``a / b`` (the response is column ``a`` divided by column ``b``)."""
    source: str = field(default="", compare=False)


def _refuse(construct: str, why: str) -> HgamNotImplementedError:
    return HgamNotImplementedError(f"gam(): {construct} is not supported — {why}")


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split ``text`` on ``sep`` outside parentheses and quotes."""
    parts: list[str] = []
    depth = 0
    quote = ""
    current: list[str] = []
    for ch in text:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in ("'", '"'):
            quote = ch
            current.append(ch)
        elif ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise HgamValidationError(f"gam(): unbalanced ')' in formula fragment {text!r}.")
            current.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if depth != 0 or quote:
        raise HgamValidationError(f"gam(): unbalanced parentheses or quotes in {text!r}.")
    parts.append("".join(current))
    return parts


def _call(text: str) -> tuple[str, list[str], dict[str, str]] | None:
    """``name(args)`` -> ``(name, positional, keyword)`` with raw argument text, or
    ``None`` if ``text`` is not a call."""
    match = re.match(r"^([A-Za-z_.][A-Za-z0-9_.]*)\s*\((.*)\)$", text.strip(), flags=re.DOTALL)
    if match is None:
        return None
    name, inner = match.group(1), match.group(2)
    positional: list[str] = []
    keyword: dict[str, str] = {}
    if inner.strip():
        for raw in _split_top_level(inner, ","):
            arg = raw.strip()
            if not arg:
                raise HgamValidationError(f"gam(): empty argument in {text!r}.")
            parts = _split_top_level(arg, "=")
            if len(parts) == 1:
                if keyword:
                    raise HgamValidationError(
                        f"gam(): positional argument {arg!r} after a named one in {text!r}."
                    )
                positional.append(arg)
            elif len(parts) == 2 and _IDENT.match(parts[0].strip()):
                key = parts[0].strip()
                if key in keyword:
                    raise HgamValidationError(f"gam(): argument {key!r} repeated in {text!r}.")
                keyword[key] = parts[1].strip()
            else:
                raise HgamValidationError(f"gam(): cannot parse argument {arg!r} in {text!r}.")
    return name, positional, keyword


def _identifier(value: str, what: str, term: str) -> str:
    if not _IDENT.match(value):
        raise _refuse(
            f"{what} {value!r} in {term}",
            "only a plain column name is accepted (compute transformed columns before "
            "calling gam(), e.g. df.with_columns(...))",
        )
    return value


def _string(value: str, what: str, term: str) -> str:
    match = _STRING.match(value)
    if match is None:
        raise HgamValidationError(f"gam(): {what} in {term} must be a quoted string.")
    return match.group(2)


def _int_k(value: str, term: str) -> int:
    if not _INT.match(value):
        raise HgamValidationError(f"gam(): k in {term} must be a positive integer, got {value!r}.")
    return int(value)


def _k_vector(value: str, term: str) -> tuple[int, ...]:
    call = _call(value)
    if call is not None and call[0] == "c" and not call[2]:
        return tuple(_int_k(a, term) for a in call[1])
    return (_int_k(value, term),)


def _parse_sz(text: str, positional: list[str], keyword: dict[str, str]) -> SzCall:
    """``s(f, x, bs="sz", ...)`` (mgcv ``smooth.construct.sz.smooth.spec``, smooth.r, mgcv 1.9-4). Only a cr
    base smooth of one covariate is ported: mgcv's default base (no ``xt``) is ``tp``, which is refused."""
    for key in keyword:
        if key not in ("bs", "k", "xt"):
            raise _refuse(
                f"argument {key!r} in {text}",
                f'only k= and xt=list(bs="cr") are ported for bs="sz" ({_COVERAGE} §2.1)',
            )
    if "xt" not in keyword:
        raise _refuse(
            f"{text} without xt=",
            'mgcv\'s default sz base smooth is tp, which is not ported for sz; write xt=list(bs="cr")',
        )
    if not _SZ_XT.match(keyword["xt"].strip()):
        raise _refuse(f"xt={keyword['xt']} in {text}", 'only xt=list(bs="cr") is ported for bs="sz"')
    if len(positional) != 2:
        raise _refuse(
            f"{text} with {len(positional)} variable(s)",
            "sz is ported for one factor and one smoothed covariate, s(f, x, ...)",
        )
    variables = (
        _identifier(positional[0], "variable", text),
        _identifier(positional[1], "variable", text),
    )
    if variables[0] == variables[1]:
        raise HgamValidationError(f"gam(): {text} names the same variable twice.")
    k = _int_k(keyword["k"], text) if "k" in keyword else _CR_DEFAULT_K
    return SzCall(variables=variables, k=k)


def _parse_fs(text: str, positional: list[str], keyword: dict[str, str]) -> FsCall:
    """``s(x, f, bs="fs", ...)`` (mgcv ``smooth.construct.fs.smooth.spec``, smooth.r, mgcv 1.9-4). Only a cr
    base smooth of one covariate is ported: mgcv's default base (no ``xt``) is ``tp``, which is refused."""
    for key in keyword:
        if key not in ("bs", "k", "xt"):
            raise _refuse(
                f"argument {key!r} in {text}",
                f'only k= and xt="cr" are ported for bs="fs" ({_COVERAGE} §2.1)',
            )
    if "xt" not in keyword:
        raise _refuse(
            f"{text} without xt=",
            'mgcv\'s default fs base smooth is tp, which is not ported for fs; write xt="cr"',
        )
    if not _FS_XT.match(keyword["xt"].strip()):
        raise _refuse(
            f"xt={keyword['xt']} in {text}", 'only xt="cr" (or xt=list(bs="cr")) is ported for bs="fs"'
        )
    if len(positional) != 2:
        raise _refuse(
            f"{text} with {len(positional)} variable(s)",
            "fs is ported for one factor and one smoothed covariate, s(x, f, ...)",
        )
    variables = (
        _identifier(positional[0], "variable", text),
        _identifier(positional[1], "variable", text),
    )
    if variables[0] == variables[1]:
        raise HgamValidationError(f"gam(): {text} names the same variable twice.")
    k = _int_k(keyword["k"], text) if "k" in keyword else _CR_DEFAULT_K
    return FsCall(variables=variables, k=k)


def _parse_smooth(text: str, positional: list[str], keyword: dict[str, str]) -> SmoothCall | SzCall | FsCall:
    if "bs" in keyword:
        early = _string(keyword["bs"], "bs", text)
        if early in _REFUSED_BASES:
            raise _refuse(f"bs={early!r} in {text}", _REFUSED_BASES[early])
        if early == "sz":
            return _parse_sz(text, positional, keyword)
        if early == "fs":
            return _parse_fs(text, positional, keyword)
    allowed = {"bs", "k", "by"}
    for key in keyword:
        if key not in allowed:
            hint = {
                "xt": 'xt= is supported only as xt=list(bs="cr") on a bs="sz" or bs="fs" smooth',
                "sp": "a fixed smoothing parameter (sp=) is not supported",
                "fx": "fixed-df smooths (fx=) are not supported",
                "m": "a non-default penalty order (m=) is not supported",
                "id": "linked smoothing parameters (id=) are not supported",
                "pc": "point constraints (pc=) are not supported",
                "d": "tensor margin dimensions (d=) are not supported",
            }.get(key, "this argument is outside the verified subset")
            raise _refuse(f"argument {key!r} in {text}", f"{hint} ({_COVERAGE} §2.1)")
    if len(positional) == 0:
        raise HgamValidationError(f"gam(): {text} names no variable.")
    if len(positional) > 1:
        raise _refuse(
            f"multi-variable smooth {text}",
            "s() with two or more variables needs a basis (tp/te) that is not verified; "
            f'use ti(x, z, bs="cr") for an interaction ({_COVERAGE} §2.1, rungs L7/L8)',
        )
    variable = _identifier(positional[0], "variable", text)
    basis = _string(keyword["bs"], "bs", text) if "bs" in keyword else "tp"
    if basis in _REFUSED_BASES:
        raise _refuse(f"bs={basis!r} in {text}", _REFUSED_BASES[basis])
    if basis not in ("cr", "re", "tp"):
        raise _refuse(f"bs={basis!r} in {text}", f"unknown or unverified basis ({_COVERAGE} §2.1)")
    by = None
    if "by" in keyword:
        by = _identifier(keyword["by"], "by", text)
    if basis == "tp":
        if by is not None:
            raise _refuse(
                f"by= on a thin-plate smooth {text}",
                f'not verified ({_COVERAGE} §2.1 tp row); write bs="cr" for a by-smooth',
            )
        k_tp = _int_k(keyword["k"], text) if "k" in keyword else _TP_DEFAULT_K
        return SmoothCall(variable=variable, basis="tp", k=k_tp)
    if basis == "re":
        if "k" in keyword:
            raise HgamNotImplementedError(
                f"gam(): {text} with bs='re' takes no k (its width is the factor's level count)."
            )
        if by is not None:
            raise _refuse(f"by= on a random-effect smooth {text}", f"not verified ({_COVERAGE} §2.1)")
        return SmoothCall(variable=variable, basis="re", k=None)
    k = _int_k(keyword["k"], text) if "k" in keyword else _CR_DEFAULT_K
    return SmoothCall(variable=variable, basis="cr", k=k, by=by)


def _parse_tensor(text: str, positional: list[str], keyword: dict[str, str], kind: str = "ti") -> TensorCall:
    """``ti(...)`` or ``te(...)`` (``kind``) with two cr margins. mgcv's ``te`` defaults to ``bs = "cr"``
    (``te()``, smooth.r), so ``bs=`` may be left out there; ``k`` defaults to 5 per margin for both."""
    for key in keyword:
        if key not in ("bs", "k"):
            raise _refuse(
                f"argument {key!r} in {text}",
                f"only bs= and k= are verified for {kind}() ({_COVERAGE} §2.1)",
            )
    if len(positional) != 2:
        raise _refuse(
            f"{text} with {len(positional)} margin(s)",
            f"{kind}() is verified for exactly two margins ({_COVERAGE} §2.1; t2 is rung L8)",
        )
    variables = tuple(_identifier(v, "variable", text) for v in positional)
    if len(set(variables)) != 2:
        raise HgamValidationError(f"gam(): {text} repeats a variable (mgcv stops here too).")
    if "bs" not in keyword and kind == "ti":
        raise _refuse(
            f"{text} without bs=",
            "mgcv's default margin basis for ti() is 'tp', and tp as a ti margin is not built "
            '(ADR-261); write bs="cr"',
        )
    basis = _string(keyword["bs"], "bs", text) if "bs" in keyword else "cr"
    if basis != "cr":
        raise _refuse(f"bs={basis!r} in {text}", f'only bs="cr" margins are verified ({_COVERAGE} §2.1)')
    k = _k_vector(keyword["k"], text) if "k" in keyword else (_TI_DEFAULT_K,)
    if kind == "te" and (len(k) not in (1, 2) or min(k) < 3):
        # te() (smooth.r): "one or more supplied k too small" or a k vector of the wrong length makes
        # mgcv warn and reset k to 5^d; that reset is not ported.
        raise _refuse(
            f"k={k!r} in {text}",
            "mgcv warns and resets k to its default 5 per margin when a k is below 3 or the k vector's "
            "length is not the number of margins; that reset is not ported, write k >= 3 per margin",
        )
    if len(k) == 1:
        k = k * 2
    if len(k) != 2:
        raise HgamValidationError(f"gam(): k in {text} must be one value or c(k1, k2).")
    return TensorCall(variables=variables, k=k, kind=kind)


_REFUSED_CALLS: dict[str, str] = {
    "t2": f"alternative tensor decomposition — {_COVERAGE} §2.1 / ladder rung L8",
    "factor": "wrap-free factors only: pass the column as a string/Enum column instead",
    "as.factor": "wrap-free factors only: pass the column as a string/Enum column instead",
    "poly": "no parametric numeric terms are verified",
    "I": "no parametric numeric terms are verified",
    "log": "compute transformed columns before calling gam()",
    "exp": "compute transformed columns before calling gam()",
    "ns": "no parametric numeric terms are verified",
    "bs": 'use s(x, bs="cr")',
}


def parse_formula(formula: str) -> ParsedFormula:
    """Parse an ``mgcv`` formula string from the verified subset.

    Raises:
        HgamValidationError: for any construct outside the subset (naming the
            construct and the coverage row), or for malformed syntax.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise HgamValidationError("gam(): formula must be a non-empty string.")
    sides = _split_top_level(formula, "~")
    if len(sides) != 2:
        raise HgamValidationError(f"gam(): formula {formula!r} needs exactly one '~' (response ~ terms).")
    response = sides[0].strip()
    if not response:
        raise HgamValidationError(f"gam(): formula {formula!r} has no response (response ~ terms).")
    # R's model.frame (stats) evaluates the left-hand side as an expression; the one expression ported
    # is a ratio of two columns, a / b (the exit formula's DthCnt / ExposCnt, a proportion response).
    ratio = [part.strip() for part in _split_top_level(response, "/")]
    denominator: str | None = None
    if len(ratio) == 2 and all(_IDENT.match(part) for part in ratio):
        response, denominator = ratio
    elif not _IDENT.match(response):
        raise _refuse(
            f"response {response!r}",
            "the response must be a plain column name or a ratio of two columns, a / b (cbind(), "
            "transformations and other expressions are outside the verified subset)",
        )
    rhs = sides[1].strip()
    if not rhs:
        raise HgamValidationError(f"gam(): formula {formula!r} has no terms.")

    terms: list[TermCall] = []
    offset: str | None = None
    seen: set[str] = set()
    for raw in _split_top_level(rhs, "+"):
        piece = raw.strip()
        if not piece:
            raise HgamValidationError(f"gam(): empty term in formula {formula!r}.")
        if piece in ("0", "-1") or piece.startswith("-") or piece == "1":
            raise _refuse(
                f"{piece!r}",
                "removing or restating the intercept is not supported; the intercept is always included",
            )
        if len(_split_top_level(piece, "-")) > 1:
            raise _refuse(
                f"'-' in {piece!r}",
                "removing a term or the intercept (- 1) is not supported; the intercept is always included",
            )
        if "*" in piece or "^" in piece or "/" in piece or "|" in piece or "%" in piece:
            raise _refuse(
                f"operator in {piece!r}",
                "write interactions explicitly as a + b + a:b (only '+' and ':' are supported)",
            )
        call = _call(piece)
        term: TermCall | None = None
        if call is not None:
            name, positional, keyword = call
            if name == "s":
                term = _parse_smooth(piece, positional, keyword)
            elif name in ("ti", "te"):
                term = _parse_tensor(piece, positional, keyword, kind=name)
            elif name == "offset":
                if keyword or len(positional) != 1:
                    raise HgamValidationError(
                        f"gam(): offset() takes exactly one column name, got {piece!r}."
                    )
                if offset is not None:
                    raise _refuse("more than one offset()", "combine them into one column")
                offset = _identifier(positional[0], "offset", piece)
                continue
            elif name in _REFUSED_CALLS:
                raise _refuse(f"{name}() in {piece!r}", _REFUSED_CALLS[name])
            else:
                raise _refuse(f"{name}() in {piece!r}", f"not in the verified subset ({_COVERAGE})")
        else:
            names = [n.strip() for n in piece.split(":")]
            if not all(_IDENT.match(n) for n in names):
                raise HgamValidationError(f"gam(): cannot parse term {piece!r}.")
            if len(names) > 2:
                raise _refuse(
                    f"interaction {piece!r}",
                    f"only two-factor interactions are verified ({_COVERAGE} §2.1 parametric row)",
                )
            if len(set(names)) != len(names):
                raise HgamValidationError(f"gam(): interaction {piece!r} repeats a variable.")
            term = ParametricCall(variables=tuple(names))
        assert term is not None
        key = term.text if not isinstance(term, SmoothCall) else f"{term.text}:{term.by}"
        if key in seen:
            raise HgamValidationError(f"gam(): term {piece!r} appears twice in the formula.")
        seen.add(key)
        terms.append(term)
    if not terms:
        raise HgamValidationError(f"gam(): formula {formula!r} has no model terms.")
    return ParsedFormula(
        response=response,
        terms=tuple(terms),
        offset=offset,
        response_denominator=denominator,
        source=formula,
    )
