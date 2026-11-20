"""Over- and under-performers on Blender rendering: a second, separate residual model.

SPEC CPU 2017 and SPEC GWPG are dominated by servers and workstations, so few retail parts are
assessed from them. Blender Open Data (CC0) measures nearly every retail CPU and GPU, so it gets
its own models here. They are never mixed with the SPEC models: the target, the scale and the
verdicts are Blender's own, and a product can carry both a SPEC and a Blender assessment.

Target: log of the median Blender Benchmark score (samples per minute, summed over the three
standard scenes) in one Blender version, the version the Value ranking uses. CPUs are fitted on
CPU renders; GPUs on the backend the Value ranking shows for each card (OptiX, HIP or oneAPI),
with the backend as a feature, because a backend is a different code path and not a property
of the card.

Every product is scored by models refitted without it and without any product with identical
specifications (an "F" or "KF" variant of an Intel chip, for example), so no product's own score
sets its prediction.

Two checks decide whether the source is used for assessment, written before any result was
seen and using the thresholds the SPEC GWPG calibration already uses:

1. Held-out accuracy: the plain model's typical held-out error must be below the baseline's
   (cores or shaders times clock with one fitted constant).
2. Version stability: the whole procedure is repeated on the next two Blender versions with the
   most products. For products ranked in both, the median gap between their residuals must be
   at most ``gwpg.CALIBRATION_MAX_GAP_PCT`` (15%) over at least ``gwpg.CALIBRATION_MIN_OVERLAP``
   shared products. A product's residual that moves with the Blender version is not a property
   of the product.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from pdi.tech_value import catalogue, cpu_specs, gwpg, value
from pdi.tech_value import model as cpu_model
from pdi.tech_value import residual_evidence as rev

SOURCE = "Blender Open Data"
METRIC = "Blender Benchmark score (samples per minute, three scenes)"
# Fewer products than this in a fit, and the model is not fitted at all.
MIN_PRODUCTS = 25
# How many other Blender versions the stability check repeats the fit on.
N_CHECK_VERSIONS = 2

_NO_RT_ARCH = {
    "Kepler",
    "Maxwell",
    "Pascal",
    "GCN 1",
    "GCN 2",
    "GCN 3",
    "Polaris",
    "Vega",
    "RDNA 1",
}
_BACKENDS = ("HIP", "ONEAPI", "CUDA")  # OptiX is the reference level

CPU_FEATURES = {
    "plain": [
        "log_cores",
        "log_threads",
        "e_share",
        "log_boost",
        "log_base",
        "log1p_l3_per_core",
        "log_tdp",
        "vintage",
        "vendor_AMD",
    ],
}
CPU_FEATURES["structural"] = [
    *CPU_FEATURES["plain"],
    "log_cores_sq",
    "cores_x_tdp",
    "vintage_sq",
]
CPU_FEATURES["gbdt"] = CPU_FEATURES["plain"]

GPU_FEATURES = {
    "plain": [
        "log_cores",
        "log_mhz",
        "log_vram",
        "log_bw",
        "vintage",
        "has_rt",
        *[f"backend_{b}" for b in _BACKENDS],
    ],
}
GPU_FEATURES["structural"] = [*GPU_FEATURES["plain"], "log_cores_sq", "log_bw_sq", "cores_x_bw"]
GPU_FEATURES["gbdt"] = GPU_FEATURES["plain"]

FEATURE_LABELS = {
    "cpu": {"cores": "Cores", "boost_mhz": "Boost clock", "tdp_w": "TDP", "l3_mb": "L3"},
    "gpu": {"cores": "Shaders", "mhz": "Clock", "vram_gb": "VRAM", "bandwidth_gbps": "Bandwidth"},
}


# --- Product tables -----------------------------------------------------------------------------
def _gpu_spec_by_key() -> dict[str, tuple[str, tuple]]:
    out: dict[str, tuple[str, tuple]] = {}
    for name, spec in gwpg.GPU_SPECS.items():
        if spec[7]:  # laptop parts never reach the desktop model keys
            continue
        out.setdefault(catalogue.key_of("gpu", name), (name, spec))
    return out


def cpu_frame(bench: pd.DataFrame) -> pd.DataFrame:
    """CPU benchmark rows with specifications and model features; unmatched rows dropped."""
    b = bench[bench["kind"] == "cpu"]
    rows = []
    for r in b.itertuples():
        spec = cpu_specs.CPU_SPECS.get(r.model_key)
        if spec is None:
            continue
        p, e, threads, base, boost, l3, tdp, year = spec
        rows.append(
            {
                "model_key": r.model_key,
                "processor": r.device_name,
                "vendor": cpu_specs.vendor(r.model_key),
                "backend": r.device_type,
                "n_runs": int(r.n_runs),
                "score": float(r.median_score),
                "p25_score": float(r.p25_score),
                "p75_score": float(r.p75_score),
                "p_cores": p,
                "e_cores": e,
                "cores": p + e,
                "threads": threads,
                "base_mhz": base,
                "boost_mhz": boost,
                "l3_mb": float(l3),
                "tdp_w": tdp,
                "release_year": year,
                "spec_group": f"cpu|{spec}",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["log_cores"] = np.log(df["cores"])
    df["log_threads"] = np.log(df["threads"])
    df["e_share"] = df["e_cores"] / df["cores"]
    df["log_boost"] = np.log(df["boost_mhz"])
    df["log_base"] = np.log(df["base_mhz"])
    df["log1p_l3_per_core"] = np.log1p(df["l3_mb"] / df["cores"])
    df["log_tdp"] = np.log(df["tdp_w"])
    df["vintage"] = (df["release_year"] - 2017).astype(float)
    df["vendor_AMD"] = (df["vendor"] == "AMD").astype(float)
    df["log_cores_sq"] = df["log_cores"] ** 2
    df["cores_x_tdp"] = df["log_cores"] * df["log_tdp"]
    df["vintage_sq"] = df["vintage"] ** 2
    df["baseline_x"] = np.log(df["cores"] * df["boost_mhz"])
    df["family"] = [_cpu_family(k) for k in df["model_key"]]
    return df


def _cpu_family(key: str) -> str:
    if "threadripper" in key:
        return "Threadripper"
    if key.startswith("amd ryzen"):
        return "Ryzen"
    if "core ultra" in key:
        return "Core Ultra"
    return "Core"


def gpu_frame(bench: pd.DataFrame) -> pd.DataFrame:
    """GPU benchmark rows with specifications and model features; unmatched rows dropped."""
    specs = _gpu_spec_by_key()
    b = bench[bench["kind"] == "gpu"]
    rows = []
    for r in b.itertuples():
        hit = specs.get(r.model_key)
        if hit is None:
            continue
        name, (shaders, mhz, vram, bus, bw, arch, year, _mobile) = hit
        rows.append(
            {
                "model_key": r.model_key,
                "processor": name,
                "vendor": gwpg.gpu_vendor(name),
                "backend": str(r.device_type),
                "n_runs": int(r.n_runs),
                "score": float(r.median_score),
                "p25_score": float(r.p25_score),
                "p75_score": float(r.p75_score),
                "cores": shaders,
                "mhz": mhz,
                "vram_gb": vram,
                "bus_bits": bus,
                "bandwidth_gbps": bw,
                "family": arch,
                "release_year": year,
                "has_rt": float(arch not in _NO_RT_ARCH and "GTX" not in name),
                "spec_group": f"gpu|{shaders}|{mhz}|{vram}|{bw}|{arch}",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["log_cores"] = np.log(df["cores"])
    df["log_mhz"] = np.log(df["mhz"])
    df["log_vram"] = np.log(df["vram_gb"])
    df["log_bw"] = np.log(df["bandwidth_gbps"])
    df["vintage"] = (df["release_year"] - 2017).astype(float)
    for b_ in _BACKENDS:
        df[f"backend_{b_}"] = (df["backend"] == b_).astype(float)
    # Ray-tracing cores matter only where the backend uses them: OptiX does, Blender's HIP and
    # oneAPI paths do not by default.
    df["has_rt"] = df["has_rt"] * (df["backend"] == "OPTIX")
    df["log_cores_sq"] = df["log_cores"] ** 2
    df["log_bw_sq"] = df["log_bw"] ** 2
    df["cores_x_bw"] = df["log_cores"] * df["log_bw"]
    df["baseline_x"] = df["log_cores"] + df["log_mhz"]
    return df


# --- Held-out residuals -------------------------------------------------------------------------
def _fit_predict(tr: pd.DataFrame, te: pd.DataFrame, features: dict[str, list[str]]) -> dict:
    y = np.log(tr["score"])
    # A feature that is constant in the training rows (a backend no card here uses) is dropped.
    features = {k: [c for c in v if tr[c].nunique() > 1] for k, v in features.items()}
    out = {}
    X = sm.add_constant(tr[features["plain"]].astype(float), has_constant="add")
    Xt = sm.add_constant(te[features["plain"]].astype(float), has_constant="add")
    out["ols"] = sm.OLS(y, X).fit().predict(Xt).to_numpy()
    X = sm.add_constant(tr[features["structural"]].astype(float), has_constant="add")
    Xt = sm.add_constant(te[features["structural"]].astype(float), has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IterationLimitWarning)
            fit = sm.QuantReg(y, X).fit(q=0.5, max_iter=2000)
        out["structural"] = fit.predict(Xt).to_numpy()
    except (ValueError, np.linalg.LinAlgError):
        out["structural"] = sm.OLS(y, X).fit().predict(Xt).to_numpy()
    gbdt = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=3,
        max_iter=150,
        min_samples_leaf=5,
        l2_regularization=1.0,
        random_state=0,
    )
    gbdt.fit(tr[features["gbdt"]], y)
    out["nonlinear"] = gbdt.predict(te[features["gbdt"]])
    intercept = float((y - tr["baseline_x"]).mean())
    out["baseline"] = (intercept + te["baseline_x"]).to_numpy()
    return out


def out_of_fold(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Leave-one-product-out residuals (log) of the plain, structural, nonlinear and baseline
    models, holding out every product with identical specifications together."""
    features = CPU_FEATURES if kind == "cpu" else GPU_FEATURES
    out = df.reset_index(drop=True).copy()
    preds = {m: np.full(len(out), np.nan) for m in ("ols", "structural", "nonlinear", "baseline")}
    for tr_idx, te_idx in LeaveOneGroupOut().split(out, groups=out["spec_group"]):
        p = _fit_predict(out.iloc[tr_idx], out.iloc[te_idx], features)
        for m, v in p.items():
            preds[m][te_idx] = v
    y = np.log(out["score"].to_numpy())
    for m in ("ols", "structural", "nonlinear", "baseline"):
        out[f"{m}_resid_log"] = y - preds[m]
    return out


def _verdict(kind: str, r: pd.Series) -> str:
    pct = {m: 100 * np.expm1(r[f"{m}_resid_log"]) for m in ("ols", "structural", "nonlinear")}
    if kind == "cpu":
        return cpu_model.consensus_verdict(
            pct["ols"], pct["structural"], pct["nonlinear"], int(r["cores"])
        )
    verdict = gwpg._classify_gpu_consensus(
        pct["ols"], pct["structural"], pct["nonlinear"], float(r["cores"]), r["bandwidth_gbps"]
    )
    if verdict == gwpg.SENSITIVE_ENTRY and r["cores"] > 1024:
        # The SPEC label names entry-level cards; here the models simply disagree.
        return cpu_model.SENSITIVE_REGIME
    return verdict


def ranking(df: pd.DataFrame, kind: str, version: str | None) -> pd.DataFrame:
    """One row per product: measured score, held-out predictions, ranges, verdict."""
    if len(df) < MIN_PRODUCTS:
        return pd.DataFrame()
    out = out_of_fold(df, kind)
    out["residual_log"] = out["ols_resid_log"]
    out["residual_pct"] = 100 * np.expm1(out["ols_resid_log"])
    out["structural_residual_pct"] = 100 * np.expm1(out["structural_resid_log"])
    out["gbdt_residual_pct"] = 100 * np.expm1(out["nonlinear_resid_log"])
    out["measured_score"] = out["score"]
    out = rev.add_model_summary(out, cpu_model.RESID_COLS, "measured_score")
    out["consensus_verdict"] = [_verdict(kind, r) for _, r in out.iterrows()]
    out["kind"] = kind
    out["blender_version"] = version
    out["form_factor"] = "Desktop"
    out["generation"] = out["release_year"].astype(str)
    labels = FEATURE_LABELS[kind]
    flags = rev.extrapolation_flags(
        out.set_index("processor")[list(labels)], out[["processor", *labels]], labels
    )
    out["outside_training_range"] = out["processor"].map(flags).fillna("")
    return out.sort_values("residual_log", ascending=False).reset_index(drop=True)


VALIDATION_MODELS = {
    "Baseline (cores x clock)": "baseline_resid_log",
    "OLS": "ols_resid_log",
    "Structural L1": "structural_resid_log",
    "Nonlinear GBDT": "nonlinear_resid_log",
}


def validation(rank: pd.DataFrame, kind: str) -> pd.DataFrame:
    if rank.empty:
        return pd.DataFrame()
    models = dict(VALIDATION_MODELS)
    if kind == "gpu":
        models["Baseline (shaders x clock)"] = models.pop("Baseline (cores x clock)")
        models = {"Baseline (shaders x clock)": models.pop("Baseline (shaders x clock)"), **models}
    slices = {"Vendor": "vendor", "Family": "family", "Generation": "generation"}
    if kind == "gpu":
        slices["Backend"] = "backend"
    return rev.validation_table(rank, models, slices).assign(kind=kind)


# --- Version stability ----------------------------------------------------------------------------
def _bench_for_version(blender: pd.DataFrame, version: str) -> pd.DataFrame:
    """value.benchmark_by_model restricted to one Blender version."""
    bench, _ = value.benchmark_by_model(blender[blender["blender_version"] == version])
    return bench


def version_check(blender: pd.DataFrame, kind: str, primary: str) -> pd.DataFrame:
    """Repeat the fit on other Blender versions and compare each shared product's residual."""
    frame = cpu_frame if kind == "cpu" else gpu_frame
    base = ranking(frame(_bench_for_version(blender, primary)), kind, primary)
    if base.empty:
        return pd.DataFrame()
    base_r = base.set_index("model_key")["ols_resid_log"]
    sizes = {}
    for v in blender["blender_version"].dropna().unique():
        if v == primary:
            continue
        sizes[v] = len(frame(_bench_for_version(blender, v)))
    others = sorted(sizes, key=lambda v: (-sizes[v], v))[:N_CHECK_VERSIONS]
    rows = []
    for v in others:
        other = ranking(frame(_bench_for_version(blender, v)), kind, v)
        if other.empty:
            shared = pd.Series(dtype=float)
        else:
            o = other.set_index("model_key")["ols_resid_log"]
            shared = (base_r - o).dropna()
        n = int(len(shared))
        gap = float(np.median(np.abs(np.expm1(shared))) * 100) if n else None
        if n < gwpg.CALIBRATION_MIN_OVERLAP:
            status = "Not validated: too few shared products"
        elif gap is not None and gap <= gwpg.CALIBRATION_MAX_GAP_PCT:
            status = "Validated"
        else:
            status = "Not validated: residuals move with the Blender version"
        rows.append(
            {
                "kind": kind,
                "primary_version": primary,
                "check_version": v,
                "n_products": int(sizes[v]),
                "n_shared": n,
                "median_abs_gap_pct": gap,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def passes(val: pd.DataFrame, check: pd.DataFrame) -> tuple[bool, str]:
    """Whether the Blender model may assess products of this kind, and why."""
    if val.empty:
        return False, f"Fewer than {MIN_PRODUCTS} products with specifications"
    a = val[val["slice"] == "All"].set_index("model")["median_abs_pct_error"]
    base = float(a[[m for m in a.index if m.startswith("Baseline")][0]])
    ols = float(a["OLS"])
    if not ols < base:
        return False, f"OLS held-out error {ols:.1f}% is not below the baseline's {base:.1f}%"
    if check.empty or not (check["status"] == "Validated").all():
        bad = check[check["status"] != "Validated"] if not check.empty else check
        which = ", ".join(f"Blender {v}" for v in bad.get("check_version", []))
        return False, f"Version check not passed ({which or 'no other version'})"
    return True, (
        f"OLS held-out error {ols:.1f}% vs baseline {base:.1f}%; residuals agree within "
        f"{gwpg.CALIBRATION_MAX_GAP_PCT:.0f}% on Blender "
        + ", ".join(str(v) for v in check["check_version"])
    )


def build(blender: pd.DataFrame) -> dict:
    """Everything the export needs, for both kinds."""
    bench, version = value.benchmark_by_model(blender)
    out = {"version": version, "rank": {}, "validation": {}, "check": {}, "passed": {}}
    for kind, frame in (("cpu", cpu_frame), ("gpu", gpu_frame)):
        df = frame(bench) if not bench.empty else pd.DataFrame()
        rank = ranking(df, kind, version) if not df.empty else pd.DataFrame()
        val = validation(rank, kind)
        check = version_check(blender, kind, version) if not rank.empty else pd.DataFrame()
        ok, why = passes(val, check)
        out["rank"][kind] = rank
        out["validation"][kind] = val
        out["check"][kind] = check
        out["passed"][kind] = {
            "passed": ok,
            "reason": why,
            "n_benchmarked": int((bench["kind"] == kind).sum()) if not bench.empty else 0,
            "n_with_specs": int(len(df)),
        }
    return out
