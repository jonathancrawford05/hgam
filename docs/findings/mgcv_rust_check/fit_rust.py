"""Try the hgam_formula target on mgcv-rust, then compare the expressible subset with mgcv."""
import inspect
import json
import time
import traceback
import warnings

import numpy as np
import pandas as pd

import mgcv_rust
from mgcv_rust import GAM, Gam

warnings.simplefilter("always")
df = pd.read_csv("hgam_synth.csv")
df["FaceHigh"] = (df["FaceSize"] == "High").astype(float)
df["SmokeS"] = (df["Smoke"] == "S").astype(float)
df["FaceHigh_SmokeS"] = df["FaceHigh"] * df["SmokeS"]
y = df["y"].to_numpy(float)
w = df["ExposCnt"].to_numpy(float)
results = {"version": getattr(mgcv_rust, "__version__", None)}


def attempt(label, fn):
    """Run fn, record ok/error and any warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = fn()
            rec = {"ok": True, "detail": out}
        except BaseException as exc:  # noqa: BLE001 - we want every failure mode, incl. pyo3 panics
            rec = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        rec["warnings"] = [f"{c.category.__name__}: {str(c.message)[:300]}" for c in caught][:5]
    results[label] = rec
    print(f"[{label}] ok={rec['ok']}", rec.get("error") or rec.get("detail"), "| warnings:", rec["warnings"])
    return rec


print("predict signature:", inspect.signature(Gam.predict))
print("predict doc:", (inspect.getdoc(Gam.predict) or "")[:1200])

# ---------------------------------------------------------------- A. can the target be expressed at all?
Xs = df[["AttdAge", "PolYear"]].to_numpy(float)
Xall = df[["AttdAge", "PolYear", "StudyYear_C", "FaceHigh", "SmokeS"]].to_numpy(float)

attempt("A1_formula_ti", lambda: str(type(GAM(family="binomial").fit_formula(
    Xs, y, formula="s(0, k=13) + s(1, k=6) + ti(0, 1, k=c(13,6))", method="REML", max_iter=50))))
attempt("A2_formula_te", lambda: str(type(GAM(family="binomial").fit_formula(
    Xs, y, formula="s(0, k=13) + s(1, k=6) + te(0, 1, k=6)", method="REML", max_iter=50))))
attempt("A3_formula_by", lambda: str(type(GAM(family="binomial").fit_formula(
    Xall, y, formula="s(0, k=13) + s(0, by=2, k=13)", method="REML", max_iter=50))))
attempt("A4_formula_sz", lambda: str(type(GAM(family="binomial").fit_formula(
    Xall, y, formula="s(0, k=13) + s(3, 0, bs='sz', k=13)", method="REML", max_iter=50))))
attempt("A5_formula_bs_cr_kw", lambda: str(type(GAM(family="binomial").fit_formula(
    Xs, y, formula="s(0, k=13, bs='cr') + s(1, k=6, bs='cr')", method="REML", max_iter=50))))
for bs in ("sz", "fs", "ti", "te", "tp", "ps"):
    attempt(f"A6_basis_{bs}", lambda bs=bs: str(Gam(
        predictors=["AttdAge", "PolYear"], family="binomial",
        predictor_basis_map={"AttdAge": bs, "PolYear": "cr"}, term_k_mapping={"AttdAge": 13, "PolYear": 6},
    ).fit(df[["AttdAge", "PolYear"]], y, sample_weight=w).converged_))


def cloglog_probe():
    g = Gam(predictors=["AttdAge", "PolYear"], family="binomial", link="cloglog",
            term_k_mapping={"AttdAge": 13, "PolYear": 6})
    g.fit(df[["AttdAge", "PolYear"]], y, sample_weight=w)
    return {"link_attr": g.link, "converged": bool(g.converged_)}


attempt("A7_cloglog_link", cloglog_probe)
attempt("A8_select_kwarg", lambda: str(Gam(
    predictors=["AttdAge", "PolYear"], family="binomial", select=True,
    term_k_mapping={"AttdAge": 13, "PolYear": 6}).fit(df[["AttdAge", "PolYear"]], y, sample_weight=w).converged_))
attempt("A9_knots_kwarg", lambda: str(Gam(
    predictors=["AttdAge", "PolYear"], family="binomial",
    knots={"PolYear": [1, 2, 3, 5, 10, 21]},
    term_k_mapping={"AttdAge": 13, "PolYear": 6}).fit(df[["AttdAge", "PolYear"]], y, sample_weight=w).converged_))

# ---------------------------------------------------------------- B. the subset it can express
preds = ["AttdAge", "PolYear", "FaceHigh", "SmokeS", "FaceHigh_SmokeS"]
basis = {"AttdAge": "cr", "PolYear": "cr", "FaceHigh": "parametric", "SmokeS": "parametric",
         "FaceHigh_SmokeS": "parametric"}
ref = pd.read_csv("ref_reduced_logit.csv")
ref_meta = json.load(open("ref_reduced_logit.json"))


def link_scale(g, X):
    sig = inspect.signature(g.predict)
    if "type" in sig.parameters:
        try:
            return np.asarray(g.predict(X, type="link"), float)
        except Exception:
            pass
    p = np.clip(np.asarray(g.predict(X), float), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def reduced_fit(method, discrete):
    g = Gam(predictors=preds, family="binomial", method=method, discrete=discrete,
            predictor_basis_map=basis, term_k_mapping={"AttdAge": 13, "PolYear": 6})
    t0 = time.perf_counter()
    g.fit(df[preds], y, sample_weight=w)
    secs = time.perf_counter() - t0
    eta = link_scale(g, df[preds])
    d = eta - ref["eta"].to_numpy()
    out = {"secs": round(secs, 3), "converged": bool(g.converged_), "n_outer_iter": int(g.n_outer_iter_),
           "n_coef": int(np.size(g.coef_)) + 1, "intercept": float(g.intercept_),
           "max_abs_d_eta": float(np.max(np.abs(d))), "rms_d_eta": float(np.sqrt(np.mean(d**2))),
           "max_abs_eta_ref": float(np.max(np.abs(ref["eta"])))}
    try:
        out["edf"] = {k: float(v) for k, v in g._edf_by_user_name().items()}
    except Exception as exc:  # noqa: BLE001
        out["edf_error"] = str(exc)[:200]
    try:
        out["lambdas"] = [float(v) for v in np.ravel(g.get_lambdas())]
    except Exception as exc:  # noqa: BLE001
        out["lambda_error"] = str(exc)[:200]
    np.save(f"rust_eta_{method}_{'disc' if discrete else 'exact'}.npy", eta)
    return out


for method, discrete in (("REML", False), ("fREML", False), ("fREML", True)):
    attempt(f"B_reduced_{method}_{'discrete' if discrete else 'exact'}", lambda m=method, d=discrete: reduced_fit(m, d))

print("mgcv reference (reduced, logit): edf", ref_meta["smooth_edf"], "sp", ref_meta["sp"], "n_coef", ref_meta["n_coef"],
      "secs", ref_meta["secs"])
json.dump(results, open("rust_results.json", "w"), indent=1, default=str)
