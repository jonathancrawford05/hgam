"""Model specifications shared by the engine tests.

Copied from the polaris-re conformance modules that built them (``gam_multiterm_conformance``,
``gam_gaussian_conformance``, ``gam_quasipoisson_conformance``); the harness itself stayed behind.
"""

from hgam.engine.gam_term_spec import ModelSpec, TermSpec


def _three_terms(
    age_knots: tuple[float, ...], year_knots: tuple[float, ...], by_label: str
) -> tuple[TermSpec, ...]:
    return (
        TermSpec(
            label="s(AttdAge)",
            variables=("AttdAge",),
            basis="cr",
            k=(len(age_knots),),
            knots=(("AttdAge", age_knots),),
        ),
        TermSpec(
            label=by_label,
            variables=("AttdAge",),
            basis="cr",
            k=(len(age_knots),),
            knots=(("AttdAge", age_knots),),
            by="StudyYear_C",
        ),
        TermSpec(
            label="ti(AttdAge,PolYear)",
            variables=("AttdAge", "PolYear"),
            basis="ti",
            k=(len(age_knots), len(year_knots)),
            knots=(("AttdAge", age_knots), ("PolYear", year_knots)),
        ),
    )


def multiterm_model_spec(age_knots: tuple[float, ...], year_knots: tuple[float, ...]) -> ModelSpec:
    """``s(AttdAge) + s(AttdAge, by=StudyYear_C) + ti(AttdAge, PolYear)``, binomial(cloglog)."""
    return ModelSpec(
        family="binomial",
        link="cloglog",
        terms=_three_terms(age_knots, year_knots, "s(AttdAge,by=StudyYear_C)"),
        weights_column="ExposCnt",
    )


def quasipoisson_model_spec(age_knots: tuple[float, ...], year_knots: tuple[float, ...]) -> ModelSpec:
    """The same three terms under quasipoisson(log), no prior weights."""
    return ModelSpec(
        family="quasipoisson",
        link="log",
        terms=_three_terms(age_knots, year_knots, "s(AttdAge):StudyYear_C"),
    )
