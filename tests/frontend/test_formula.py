"""``hgam.frontend.formula`` — the parser and its refusal list (preview Slice P3, ADR-251).

One test per refused construct: each must raise ``HgamValidationError`` naming the
construct (and, where the refusal is a coverage decision, the coverage row)."""

import pytest

from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend.formula import (
    ParametricCall,
    SmoothCall,
    TensorCall,
    parse_formula,
)


def test_parses_the_verified_subset() -> None:
    p = parse_formula(
        'y ~ A + B + A:B + s(x, bs="cr", k=8) + s(x, by=w, bs="cr") + s(f, bs="re") '
        '+ ti(x, z, bs="cr", k=c(6, 5)) + offset(off)'
    )
    assert p.response == "y"
    assert p.offset == "off"
    assert p.terms == (
        ParametricCall(("A",)),
        ParametricCall(("B",)),
        ParametricCall(("A", "B")),
        SmoothCall("x", "cr", 8),
        SmoothCall("x", "cr", 10, by="w"),  # mgcv's default k for cr
        SmoothCall("f", "re", None),
        TensorCall(("x", "z"), (6, 5)),
    )


def test_te_parses_with_cr_default_and_k() -> None:
    """mgcv's te() defaults to bs="cr" and k = 5 per margin (te, smooth.r); the label is te(x,z)."""
    (term,) = parse_formula("y ~ te(x, z)").terms
    assert term == TensorCall(("x", "z"), (5, 5), kind="te")
    assert term.text == "te(x,z)"
    assert parse_formula('y ~ te(x, z, bs="cr", k=c(5, 4))').terms == (
        TensorCall(("x", "z"), (5, 4), kind="te"),
    )


def test_ti_defaults_and_scalar_k_recycle() -> None:
    assert parse_formula('y ~ ti(x, z, bs="cr")').terms == (TensorCall(("x", "z"), (5, 5)),)
    assert parse_formula('y ~ ti(x, z, bs="cr", k=7)').terms == (TensorCall(("x", "z"), (7, 7)),)


def test_single_quotes_and_whitespace_and_dotted_names() -> None:
    p = parse_formula("  y.rate~ s( age.at , bs = 'cr' ,k=5 )  ")
    assert p.response == "y.rate"
    assert p.terms == (SmoothCall("age.at", "cr", 5),)


def test_a_bare_smooth_is_thin_plate_with_mgcvs_default_k() -> None:
    """Slice R4 (ADR-261) lifts the 2026-10-07 refusal: ``tp`` is verified, so a bare ``s(x)``
    parses to it as in ``mgcv`` (default ``k = 10``); the explicit ``cr`` form is unchanged."""
    assert parse_formula("y ~ s(x)").terms == (SmoothCall("x", "tp", 10),)
    assert parse_formula('y ~ s(x, bs="tp")').terms == (SmoothCall("x", "tp", 10),)
    assert parse_formula("y ~ s(x, k=7)").terms == (SmoothCall("x", "tp", 7),)
    assert parse_formula('y ~ s(x, bs="cr")').terms == (SmoothCall("x", "cr", 10),)


def test_a_thin_plate_smooth_takes_no_by() -> None:
    with pytest.raises(HgamNotImplementedError, match=r"by= on a thin-plate smooth"):
        parse_formula("y ~ s(x, by=g)")


@pytest.mark.parametrize(
    ("formula", "needle"),
    [
        ('y ~ s(x, bs="ts")', "bs='ts'"),
        ('y ~ s(x, bs="cc")', "bs='cc'"),
        ('y ~ s(x, bs="ps")', "bs='ps'"),
        ('y ~ s(f, x, bs="fs")', "without xt="),
        ('y ~ s(x, f, bs="fs", xt="tp")', 'xt="tp"'),
        ('y ~ s(f, x, bs="sz", k=8)', "without xt="),
        ('y ~ s(x, z, bs="cr")', "multi-variable smooth"),
        ('y ~ s(x, bs="cr", sp=1)', "argument 'sp'"),
        ('y ~ s(x, bs="cr", fx=TRUE)', "argument 'fx'"),
        ('y ~ s(x, bs="cr", m=1)', "argument 'm'"),
        ('y ~ s(f, bs="re", k=3)', "takes no k"),
        ('y ~ s(f, by=g, bs="re")', "by= on a random-effect"),
        ("y ~ te(x, z, w)", "3 margin"),
        ("y ~ t2(x, z, w)", "3 margin"),
        ("y ~ t2(x, z, full=TRUE)", "argument 'full'"),
        ("y ~ t2(x, z, ord=1)", "argument 'ord'"),
        ('y ~ ti(x, z, w, bs="cr")', "3 margin"),
        ("y ~ ti(x, z)", "without bs="),
        ('y ~ ti(x, z, bs="tp")', "bs='tp'"),
        ('y ~ ti(x, z, bs="cr", by=w)', "argument 'by'"),
        ('y ~ te(x, z, bs="tp")', "bs='tp'"),
        ("y ~ te(x, z, by=w)", "argument 'by'"),
        ("y ~ te(x, z, d=2)", "argument 'd'"),
        ("y ~ te(x, z, k=2)", "resets k"),
        ("y ~ te(x, z, k=c(5, 4, 3))", "resets k"),
        ("y ~ a*b", "operator"),
        ("y ~ a^2", "operator"),
        ("y ~ a:b:c", "interaction"),
        ("y ~ x - 1", "'-'"),
        ("y ~ 0 + x", "'0'"),
        ("y ~ poly(x, 2)", "poly()"),
        ("y ~ log(x)", "log()"),
        ("y ~ factor(g)", "factor()"),
        ("log(y) ~ a", "response 'log(y)'"),
        ("cbind(a, b) ~ x", "response"),
        ('y ~ s(log(x), bs="cr")', "variable 'log(x)'"),
        ("y ~ offset(log(e))", "offset"),
        ("y ~ a + offset(o1) + offset(o2)", "more than one offset"),
        ("y ~ weird(x)", "weird()"),
    ],
)
def test_every_refused_construct_is_refused_by_name(formula: str, needle: str) -> None:
    with pytest.raises(HgamNotImplementedError) as info:
        parse_formula(formula)
    assert needle in str(info.value)


@pytest.mark.parametrize(
    "formula",
    ["", "   ", "y", "y ~", "~ x", "y ~ x ~ z", 'y ~ s(x, bs="cr"', "y ~ a + + b", "y ~ a + a"],
)
def test_malformed_formulas_raise(formula: str) -> None:
    with pytest.raises(HgamValidationError):
        parse_formula(formula)


@pytest.mark.parametrize("formula", ['y ~ s(x, bs="ps")', "y ~ t2(x, x2, full=TRUE)"])
def test_an_unported_construct_raises_not_implemented(formula: str) -> None:
    """CLAUDE.md: a term, family or option that is not ported raises NotImplementedError naming it."""
    with pytest.raises(NotImplementedError):
        parse_formula(formula)
