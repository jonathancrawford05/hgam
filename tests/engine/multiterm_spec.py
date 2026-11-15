"""The three-term binomial-cloglog ``ModelSpec`` several engine tests fit.

Copied from polaris-re's ``analytics/gam_multiterm_conformance._multiterm_model_spec``; the rest
of that conformance module stayed behind.
"""

from hgam.engine.gam_term_spec import ModelSpec, TermSpec


def multiterm_model_spec(age_knots: tuple[float, ...], year_knots: tuple[float, ...]) -> ModelSpec:
    """``s(AttdAge) + s(AttdAge, by=StudyYear_C) + ti(AttdAge, PolYear)``, weights ``ExposCnt``."""
    ref_term = TermSpec(
        label="s(AttdAge)",
        variables=("AttdAge",),
        basis="cr",
        k=(len(age_knots),),
        knots=(("AttdAge", age_knots),),
    )
    by_term = TermSpec(
        label="s(AttdAge,by=StudyYear_C)",
        variables=("AttdAge",),
        basis="cr",
        k=(len(age_knots),),
        knots=(("AttdAge", age_knots),),
        by="StudyYear_C",
    )
    ti_term = TermSpec(
        label="ti(AttdAge,PolYear)",
        variables=("AttdAge", "PolYear"),
        basis="ti",
        k=(len(age_knots), len(year_knots)),
        knots=(("AttdAge", age_knots), ("PolYear", year_knots)),
    )
    return ModelSpec(
        family="binomial", link="cloglog", terms=(ref_term, by_term, ti_term), weights_column="ExposCnt"
    )
