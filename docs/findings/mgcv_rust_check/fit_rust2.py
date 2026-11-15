"""mgcv-rust vs mgcv on the part of the HGAM target that mgcv-rust can express."""
import json
import time
import warnings

import numpy as np
import pandas as pd

from mgcv_rust import Gam

cells = pd.read_csv("hgam_synth.csv")
ex = pd.read_csv("hgam_expanded.csv")
for d in (cells, ex):
    d["FaceHigh"] = (d["FaceSize"] == "High").astype(float)
    d["SmokeS"] = (d["Smoke"] == "S").astype(float)
    d["FaceHigh_SmokeS"] = d["FaceHigh"] * d["SmokeS"]
preds = ["AttdAge", "PolYear", "FaceHigh", "SmokeS", "FaceHigh_SmokeS"]
basis = {"AttdAge": "cr", "PolYear": "cr", "FaceHigh": "parametric", "SmokeS": "parametric", "FaceHigh_SmokeS": "parametric"}
kmap = {"AttdAge": 13, "PolYear": 6}
out = {}


def ref(tag):
    with open(f"ref2_{tag}.json") as fh:
        meta = json.load(fh)
    return pd.read_csv(f"ref2_{tag}.csv")["eta"].to_numpy(), meta


def run(label, family, X, y, w, ref_tag, **kw):
    rec = {"family": family, **{k: str(v) for k, v in kw.items()}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            g = Gam(predictors=preds, family=family, predictor_basis_map=basis, term_k_mapping=kmap, **kw)
            t0 = time.perf_counter()
            g.fit(X[preds], y, sample_weight=w)
            rec["secs"] = round(time.perf_counter() - t0, 2)
            eta = np.asarray(g.predict(cells[preds], scale="link"), float)
            eta_ref, meta = ref(ref_tag)
            d = eta - eta_ref
            rec.update(ok=True, converged=bool(g.converged_), outer_iter=int(g.n_outer_iter_),
                       max_abs_d_eta=float(np.max(np.abs(d))), rms_d_eta=float(np.sqrt(np.mean(d**2))),
                       edf={k: round(float(v), 4) for k, v in g._edf_by_user_name().items()},
                       ref_edf={k: round(float(v), 4) for k, v in meta["smooth_edf"].items()},
                       ref_secs=meta["secs"], n_coef=int(np.size(g.coef_)) + 1, ref_n_coef=meta["n_coef"])
            np.save(f"rust2_eta_{label}.npy", eta)
            rec["_gam"] = g
        except BaseException as exc:  # noqa: BLE001
            rec.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}")
        rec["warnings"] = sorted({f"{c.category.__name__}: {str(c.message)[:160]}" for c in caught})[:4]
    out[label] = {k: v for k, v in rec.items() if k != "_gam"}
    print(label, json.dumps(out[label]))
    return rec.get("_gam")


# --- binomial (logit) on 0/1 rows with weights: the only binomial response form accepted
y01, w01 = ex["y01"].to_numpy(float), ex["w01"].to_numpy(float)
g_reml = run("binom_REML", "binomial", ex, y01, w01, "logit_exp_gam", method="REML")
run("binom_fREML", "binomial", ex, y01, w01, "logit_exp_bam", method="fREML")
run("binom_fREML_discrete", "binomial", ex, y01, w01, "logit_exp_bam_discrete", method="fREML", discrete=True)

# --- are unsupported mgcv arguments rejected, or silently ignored?
base_eta = np.load("rust2_eta_binom_REML.npy")
for label, kw in (("kwarg_select", {"select": True}), ("kwarg_knots", {"knots": {"PolYear": [1, 2, 3, 5, 10, 21]}}),
                  ("kwarg_offset", {"offset": "ExposCnt"})):
    g = run(label, "binomial", ex, y01, w01, "logit_exp_gam", method="REML", **kw)
    if out[label].get("ok"):
        out[label]["max_abs_d_vs_plain_fit"] = float(np.max(np.abs(np.load(f"rust2_eta_{label}.npy") - base_eta)))
        print("   ", label, "max |d eta| vs the same fit without the argument:", out[label]["max_abs_d_vs_plain_fit"])

# --- Poisson-family: rates with exposure weights (there is no offset argument)
rate, expo = (cells["DthCnt"] / cells["ExposCnt"]).to_numpy(float), cells["ExposCnt"].to_numpy(float)
run("pois_rate_weights_REML", "poisson", cells, rate, expo, "pois_offset_gam", method="REML")
run("qpois_rate_weights_REML", "quasipoisson", cells, rate, expo, "qpois_rate_gam", method="REML")
run("pois_counts_no_offset_REML", "poisson", cells, cells["DthCnt"].to_numpy(float), None, "pois_offset_gam", method="REML")

# --- what the fitted object exposes about knots
if g_reml is not None:
    try:
        ser = g_reml.serialize()
        txt = ser if isinstance(ser, str) else json.dumps(ser, default=str)
        print("serialize keys/preview:", txt[:600])
        out["serialize_mentions_knots"] = "knot" in txt.lower()
    except BaseException as exc:  # noqa: BLE001
        print("serialize failed:", type(exc).__name__, str(exc)[:200])
    try:
        print("lambdas:", np.ravel(g_reml.get_lambdas()))
    except BaseException as exc:  # noqa: BLE001
        print("get_lambdas failed:", str(exc)[:200])
with open("rust2_results.json", "w") as fh:
    json.dump(out, fh, indent=1)
