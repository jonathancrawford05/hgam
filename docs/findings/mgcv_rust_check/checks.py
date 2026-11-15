"""Follow-up checks: discrete no-op, quasi-Poisson under fREML, quasibinomial proportions, pairwise spread."""
import itertools
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from mgcv_rust import Gam

cells = pd.read_csv("hgam_synth.csv")
cells["FaceHigh"] = (cells["FaceSize"] == "High").astype(float)
cells["SmokeS"] = (cells["Smoke"] == "S").astype(float)
cells["FaceHigh_SmokeS"] = cells["FaceHigh"] * cells["SmokeS"]
preds = ["AttdAge", "PolYear", "FaceHigh", "SmokeS", "FaceHigh_SmokeS"]
basis = {"AttdAge": "cr", "PolYear": "cr", "FaceHigh": "parametric", "SmokeS": "parametric", "FaceHigh_SmokeS": "parametric"}
kmap = {"AttdAge": 13, "PolYear": 6}
rate, expo = (cells["DthCnt"] / cells["ExposCnt"]).to_numpy(float), cells["ExposCnt"].to_numpy(float)


def fit(family, y, w, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = Gam(predictors=preds, family=family, predictor_basis_map=basis, term_k_mapping=kmap, **kw)
        t0 = time.perf_counter()
        g.fit(cells[preds], y, sample_weight=w)
        secs = time.perf_counter() - t0
    eta = np.asarray(g.predict(cells[preds], scale="link"), float)
    return g, eta, secs, [str(c.message)[:120] for c in caught]


def ref(tag):
    return pd.read_csv(f"ref2_{tag}.csv")["eta"].to_numpy()


print("== discrete flag (binomial fREML):", float(np.max(np.abs(np.load("rust2_eta_binom_fREML.npy") - np.load("rust2_eta_binom_fREML_discrete.npy")))))

print("== quasi-Poisson, rates with exposure weights")
for method in ("REML", "fREML"):
    try:
        g, eta, secs, warns = fit("quasipoisson", rate, expo, method=method)
        d = eta - ref("qpois_rate_gam")
        print(f"  {method}: converged={bool(g.converged_)} outer_iter={int(g.n_outer_iter_)} secs={secs:.2f} "
              f"max|d_eta|={np.max(np.abs(d)):.3g} rms={np.sqrt(np.mean(d**2)):.3g} "
              f"edf={[round(float(v), 4) for v in list(g._edf_by_user_name().values())[:2]]} warnings={warns[:1]}")
    except BaseException as exc:  # noqa: BLE001
        print(f"  {method}: FAILED {type(exc).__name__}: {str(exc)[:250]}")
with open("ref2_qpois_rate_gam.json") as fh:
    m = json.load(fh)
print("  mgcv quasi-Poisson reference: edf", {k: round(v, 4) for k, v in m["smooth_edf"].items()}, "scale", round(m["scale"], 4), "secs", m["secs"])

print("== proportions with weights (the target's response form)")
for family in ("binomial", "quasibinomial"):
    try:
        g, eta, secs, warns = fit(family, cells["y"].to_numpy(float), expo, method="REML")
        d = eta - ref("logit_prop_gam")
        print(f"  {family}: ok converged={bool(g.converged_)} max|d_eta| vs mgcv binomial proportion-form={np.max(np.abs(d)):.3g} warnings={warns[:1]}")
    except BaseException as exc:  # noqa: BLE001
        print(f"  {family}: REJECTED {type(exc).__name__}: {str(exc)[:200]}")

print("== pairwise max / rms |d eta| on the 30,000 cells (binomial logit, reduced model)")
series = {
    "mgcv gam REML (0/1 rows)": ref("logit_exp_gam"), "mgcv gam REML (proportions)": ref("logit_prop_gam"),
    "mgcv bam fREML": ref("logit_exp_bam"), "mgcv bam fREML discrete": ref("logit_exp_bam_discrete"),
    "rust REML": np.load("rust2_eta_binom_REML.npy"), "rust fREML": np.load("rust2_eta_binom_fREML.npy"),
}
age = cells["AttdAge"].to_numpy()
for (a, x), (b, z) in itertools.combinations(series.items(), 2):
    d = np.abs(x - z)
    i = int(np.argmax(d))
    print(f"  {a:28s} vs {b:28s} max={d.max():.2e} rms={np.sqrt(np.mean(d**2)):.2e} p99={np.quantile(d, 0.99):.2e} (max at age {age[i]:.1f})")
