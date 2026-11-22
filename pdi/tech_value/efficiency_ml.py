"""Machine learning on the performance-per-watt page: power gap filling, efficiency residuals
and efficiency tiers.

All three work on desktop chips with a Blender score and the specifications the Blender
residual model already uses, plus each chip's architecture and process node
(``process_nodes``). Every prediction a chip gets comes from models refitted without it and
without any chip with identical specifications, the same leave-one-out scheme as the Blender
model on the "Over & under performers" page, so no chip's own number sets its prediction.

1. **Power gap filling.** Predicts a chip's power figure from its specifications, with an 80%
   range taken from the spread of every other chip's held-out error. Today the only desktop
   power figure there is to learn from is the *rated* one, so the estimate is an estimated rated
   figure and is labelled so. When measured desktop power is collected, the same code learns
   from it instead and the estimate becomes an estimate of real draw.
2. **Efficiency residuals.** Predicts points per rated watt from cores, clocks, cache, process
   node, architecture, rated power and launch year. A chip whose measured efficiency sits outside
   the 80% range of the linear model, with the gradient-boosted model agreeing on the side, beats
   or misses what its specifications predict. Rated power is a feature, because a low power
   label alone raises efficiency and that is no achievement of the design.
3. **Tiers.** A Gaussian mixture on log points per watt of the current line-up (chips launched
   in the last three launch years, or with a current price), with the number of tiers chosen by
   BIC, so no bracket is drawn by hand. Across all years efficiency is one smooth continuum, so
   tiers are only sought among chips a buyer would compare today. BIC may choose one tier: that
   is a result, and the page says so. A chip whose tier membership is below 80% is borderline.

Each model is checked against a launch-year-only baseline (newer chips are more efficient), and
the page says when it does not beat it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LeaveOneGroupOut
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

from pdi.tech_value import blender_model, process_nodes
from pdi.tech_value import residual_evidence as rev
from pdi.tech_value.gpu_specs import NOT_ENTERED

MIN_PRODUCTS = 25
MAX_TIERS = 6
LINEUP_YEARS = 3
# A BIC lead smaller than this over the runner-up tier count is reported as uncertain.
BIC_CLEAR = 6.0
BIC_NEAR = 2.0
BORDERLINE_PROB = 0.80
BEATS = "Beats its specs"
MISSES = "Misses its specs"
IN_LINE = "In line with its specs"
# Power cannot be estimated for a name that covers several power levels.
NO_ESTIMATE = {k for k, why in NOT_ENTERED.items() if why.startswith(("Laptop", "Sold at"))}

_CPU_SPECS = ["log_cores", "log_threads", "e_share", "log_boost", "log_base", "log1p_l3_per_core",
              "log_node", "vintage", "vendor_AMD"]  # fmt: skip
# AMD cards run Blender on HIP and Intel cards on oneAPI, so the backend stands in for the maker.
_GPU_SPECS = ["log_cores", "log_mhz", "log_vram", "log_bw", "log_node", "has_rt", "vintage",
              "backend_HIP", "backend_ONEAPI", "backend_CUDA"]  # fmt: skip
POWER_FEATURES = {"cpu": _CPU_SPECS, "gpu": _GPU_SPECS}
EFFICIENCY_FEATURES = {
    "cpu": [c for c in _CPU_SPECS if c != "log_base"] + ["log_watts"],
    "gpu": [*_GPU_SPECS, "log_watts"],
}
MODEL_LABELS = {"base": "Launch year only", "ols": "Linear", "gbdt": "Gradient boosting"}


# --- Table ----------------------------------------------------------------------------------------
def frame(bench: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    """One row per benchmarked desktop chip with specifications, node, architecture and, where
    there is one, its rated power and points per rated watt."""
    parts = []
    for kind, build in (("cpu", blender_model.cpu_frame), ("gpu", blender_model.gpu_frame)):
        df = build(bench)
        if df.empty:
            continue
        df = df.assign(kind=kind)
        if kind == "cpu":
            nodes = [process_nodes.cpu_node(k) for k in df["model_key"]]
            df["arch"] = [n[0] if n else None for n in nodes]
            df["node"] = [n[1] if n else None for n in nodes]
            df["node_nm"] = [n[2] if n else np.nan for n in nodes]
        else:
            nodes = [process_nodes.gpu_node(p, a) for p, a in zip(df["processor"], df["family"],
                                                                  strict=True)]  # fmt: skip
            df["arch"] = df["family"]
            df["node"] = [n[0] if n else None for n in nodes]
            df["node_nm"] = [n[1] if n else np.nan for n in nodes]
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df[df["node_nm"].notna()].copy()
    keep = ["kind", "model_key", "name", "watts", "power_label", "score_per_watt", "price_usd"]
    rated = ranked[keep] if len(ranked) else pd.DataFrame(columns=keep)
    df = df.merge(rated, on=["kind", "model_key"], how="left")
    df["name"] = df["name"].fillna(df["processor"])
    df["year"] = df["release_year"].astype(int)
    df["log_node"] = np.log(df["node_nm"].astype(float))
    df["log_watts"] = np.log(df["watts"])
    for v in ("AMD", "Intel"):
        df[f"vendor_{v}"] = (df["vendor"] == v).astype(float)
    for b in ("HIP", "ONEAPI", "CUDA"):
        col = f"backend_{b}"
        df[col] = df[col].fillna(0.0) if col in df else 0.0
    df["has_rt"] = df["has_rt"].fillna(0.0) if "has_rt" in df else 0.0
    df["log_ppw"] = np.log(df["score_per_watt"])
    df["vendor"] = df["vendor"].str.lower()
    return df.reset_index(drop=True)


# --- Held-out fits --------------------------------------------------------------------------------
def _fit(tr: pd.DataFrame, te: pd.DataFrame, target: str, features: list[str]) -> dict:
    y = tr[target]
    cols = [c for c in features if tr[c].nunique() > 1]
    X = sm.add_constant(tr[cols].astype(float), has_constant="add")
    Xt = sm.add_constant(te[cols].astype(float), has_constant="add")
    with warnings.catch_warnings():
        # A held-out fold can leave two dummies identical (one oneAPI card left, say).
        warnings.simplefilter("ignore", SingularMatrixWarning)
        out = {"ols": sm.OLS(y, X).fit().predict(Xt).to_numpy()}
    gcols = [*cols, "arch_code"]
    gbdt = HistGradientBoostingRegressor(
        loss="absolute_error", max_depth=3, max_iter=200, min_samples_leaf=5,
        l2_regularization=1.0, categorical_features=[len(gcols) - 1], random_state=0,
    )  # fmt: skip
    gbdt.fit(tr[gcols].to_numpy(dtype=float), y)
    out["gbdt"] = gbdt.predict(te[gcols].to_numpy(dtype=float))
    Xb = sm.add_constant(tr[["vintage"]].astype(float), has_constant="add")
    Xbt = sm.add_constant(te[["vintage"]].astype(float), has_constant="add")
    out["base"] = sm.OLS(y, Xb).fit().predict(Xbt).to_numpy()
    return out


def _with_arch_code(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(arch_code=pd.Categorical(df["arch"].fillna("?")).codes.astype(float))


def held_out(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    """Leave-one-spec-group-out predictions (log) of each model, as pred_<model> columns."""
    out = _with_arch_code(df.reset_index(drop=True))
    preds = {m: np.full(len(out), np.nan) for m in MODEL_LABELS}
    for tr, te in LeaveOneGroupOut().split(out, groups=out["spec_group"]):
        for m, v in _fit(out.iloc[tr], out.iloc[te], target, features).items():
            preds[m][te] = v
    for m, v in preds.items():
        out[f"pred_{m}"] = v
        out[f"resid_{m}"] = out[target] - v
    return out


def _abs_pct(resid: pd.Series) -> float:
    return float(np.median(np.expm1(np.abs(resid))) * 100)


def validation(fit: pd.DataFrame, kind: str, target: str) -> list[dict]:
    rows = []
    for m, label in MODEL_LABELS.items():
        r = fit[f"resid_{m}"].to_numpy(dtype=float)
        lo, hi = rev.loo_interval(r)
        cover = np.nanmean(((r >= lo) & (r <= hi))[np.isfinite(lo)]) * 100
        rows.append({"kind": kind, "target": target, "model": label, "n_chips": len(r),
                     "median_abs_pct_error": _abs_pct(fit[f"resid_{m}"]),
                     "range_coverage_pct": float(cover)})  # fmt: skip
    return rows


def beats_baseline(val: pd.DataFrame, kind: str, target: str) -> bool:
    v = val[(val["kind"] == kind) & (val["target"] == target)].set_index("model")
    if v.empty:
        return False
    err = v["median_abs_pct_error"]
    return bool(
        min(err[MODEL_LABELS["ols"]], err[MODEL_LABELS["gbdt"]]) < err[MODEL_LABELS["base"]]
    )


# --- 1. Power gap filling -------------------------------------------------------------------------
def power_estimates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Estimated power for chips without a figure, and the held-out check on chips with one."""
    rows, val = [], []
    for kind, part in df.groupby("kind"):
        feats = POWER_FEATURES[kind]
        known = part[part["watts"].notna()]
        if len(known) < MIN_PRODUCTS:
            continue
        fit = held_out(known, "log_watts", feats)
        v = validation(fit, kind, "rated power")
        val += v
        best = min(("ols", "gbdt"), key=lambda m: _abs_pct(fit[f"resid_{m}"]))
        q_lo, q_hi = np.quantile(fit[f"resid_{best}"], rev.INTERVAL_QUANTILES)
        gaps = part[part["watts"].isna() & ~part["model_key"].isin(NO_ESTIMATE)]
        for r in fit.itertuples():
            est = float(np.exp(getattr(r, f"pred_{best}")))
            rows.append(_power_row(kind, r, est, q_lo, q_hi, "Held-out check", best))
        if len(gaps):
            tr = _with_arch_code(part).loc[part["watts"].notna()]
            te = _with_arch_code(part).loc[gaps.index]
            pred = _fit(tr, te, "log_watts", feats)[best]
            for r, p in zip(te.itertuples(), pred, strict=True):
                rows.append(_power_row(kind, r, float(np.exp(p)), q_lo, q_hi, "Estimated", best))
    return pd.DataFrame(rows), val


def _power_row(kind, r, est, q_lo, q_hi, status, model) -> dict:
    rated = getattr(r, "watts", np.nan)
    return {
        "kind": kind, "model_key": r.model_key, "name": r.name, "vendor": r.vendor,
        "year": int(r.year), "arch": r.arch, "node": r.node, "status": status,
        "model": MODEL_LABELS[model], "estimate_w": est, "estimate_lo_w": est * np.exp(q_lo),
        "estimate_hi_w": est * np.exp(q_hi),
        "rated_w": None if pd.isna(rated) else float(rated),
        "gap_pct": None if pd.isna(rated) else float(100 * (rated / est - 1)),
        "score": float(r.score),
        "est_score_per_watt": float(r.score) / est,
    }  # fmt: skip


# --- 2. Efficiency residuals ----------------------------------------------------------------------
def residuals(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    out, val = [], []
    for kind, part in df.groupby("kind"):
        rated = part[part["watts"].notna()]
        if len(rated) < MIN_PRODUCTS:
            continue
        fit = held_out(rated, "log_ppw", EFFICIENCY_FEATURES[kind])
        val += validation(fit, kind, "points per rated watt")
        r = fit["resid_ols"].to_numpy(dtype=float)
        lo, hi = rev.loo_interval(r)
        g = fit["resid_gbdt"].to_numpy(dtype=float)
        verdict = np.where((r > hi) & (g > 0), BEATS, np.where((r < lo) & (g < 0), MISSES, IN_LINE))
        res = pd.DataFrame(
            {
                "kind": kind, "model_key": fit["model_key"], "name": fit["name"],
                "vendor": fit["vendor"], "year": fit["year"], "arch": fit["arch"],
                "node": fit["node"], "cores": fit["cores"], "watts": fit["watts"],
                "n_runs": fit["n_runs"], "score_per_watt": fit["score_per_watt"],
                "expected_ppw": np.exp(fit["pred_ols"]),
                "expected_lo": np.exp(fit["pred_ols"] + lo),
                "expected_hi": np.exp(fit["pred_ols"] + hi),
                "residual_pct": 100 * np.expm1(r),
                "gbdt_residual_pct": 100 * np.expm1(g),
                "verdict": verdict,
            }
        )  # fmt: skip
        out.append(res.sort_values("residual_pct", ascending=False))
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), val


# --- 3. Tiers -------------------------------------------------------------------------------------
def tiers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tier per chip from a tied-variance Gaussian mixture on log points per watt.

    Tied variance keeps every tier a contiguous band: with one shared spread, a chip always
    falls in the tier whose centre is nearest on the log scale.
    """
    chips, summ = [], []
    for kind, part in df[df["watts"].notna()].groupby("kind"):
        part = part[(part["year"] > part["year"].max() - LINEUP_YEARS) | part["price_usd"].notna()]
        if len(part) < MIN_PRODUCTS:
            continue
        X = part[["log_ppw"]].to_numpy()
        fits = []
        for k in range(1, min(MAX_TIERS, len(part) // 8) + 1):
            gm = GaussianMixture(k, covariance_type="tied", n_init=5, random_state=0).fit(X)
            fits.append((gm.bic(X), k, gm))
        # Fewest tiers within BIC_NEAR of the best fit: a tier the data barely supports is
        # not added. "Clear" when every other tier count is BIC_CLEAR or more behind.
        low = min(f[0] for f in fits)
        bic, k, gm = next(f for f in fits if f[0] <= low + BIC_NEAR)
        others = [f[0] for f in fits if f[1] != k]
        lead = min(others) - bic if others else None
        order = np.argsort(-gm.means_.ravel())
        rank = {int(c): i + 1 for i, c in enumerate(order)}
        prob = gm.predict_proba(X)
        comp = prob.argmax(axis=1)
        p = part.assign(tier=[rank[int(c)] for c in comp], tier_prob=prob.max(axis=1), n_tiers=k)
        p["borderline"] = p["tier_prob"] < BORDERLINE_PROB
        chips.append(p[["kind", "model_key", "name", "vendor", "year", "score_per_watt", "tier",
                        "tier_prob", "borderline", "n_tiers"]])  # fmt: skip
        for t, g in p.groupby("tier"):
            top = g.sort_values("score_per_watt", ascending=False)
            summ.append({"kind": kind, "tier": int(t), "n_tiers": k, "n": len(g),
                         "min_ppw": float(g["score_per_watt"].min()),
                         "max_ppw": float(g["score_per_watt"].max()),
                         "median_ppw": float(g["score_per_watt"].median()),
                         "median_year": float(g["year"].median()),
                         "examples": ", ".join(top["name"].head(3)),
                         "bic_lead": lead, "clear": bool(lead is not None and lead >= BIC_CLEAR),
                         "bic_by_k": ", ".join(f"{kk}: {b:.1f}" for b, kk, _ in fits)})  # fmt: skip
    chips_df = pd.concat(chips, ignore_index=True) if chips else pd.DataFrame()
    return chips_df, pd.DataFrame(summ)


def build(bench: pd.DataFrame, ranked: pd.DataFrame) -> dict:
    df = frame(bench, ranked)
    if df.empty:
        empty = pd.DataFrame()
        return {"frame": df, "power": empty, "residuals": empty, "tiers": empty,
                "tier_summary": empty, "validation": empty}  # fmt: skip
    power, v1 = power_estimates(df)
    res, v2 = residuals(df)
    chip_tiers, tier_summary = tiers(df)
    if len(res) and len(chip_tiers):
        res = res.merge(chip_tiers[["kind", "model_key", "tier", "tier_prob", "borderline"]],
                        on=["kind", "model_key"], how="left")  # fmt: skip
    return {"frame": df, "power": power, "residuals": res, "tiers": chip_tiers,
            "tier_summary": tier_summary, "validation": pd.DataFrame(v1 + v2)}  # fmt: skip
