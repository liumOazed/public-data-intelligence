"""Tech Value model step: baseline, plain regression, honest evaluation, residual ranking.

Target: log of the SPECrate 2017 Integer base score of one published result.

  baseline   log(score) = a + log(cores x MHz)           "cores times clock", one fitted constant
  plain      ordinary least squares on a handful of specification features

Processors are split by the first hardware-availability date they appear with. Models are fit
on the training period only, the validation period ranks them, and the holdout (the newest
generation) is scored once for whatever validation picked. Dates are fixed in config.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from pdi.tech_value import config

NUMERIC = [
    "log_cores",
    "log_mhz",
    "log_chips",
    "smt",
    "log1p_l2_mb_per_core",
    "log1p_l3_mb_per_core",
    "vintage_years",
]
VENDORS = ["AMD", "Other"]  # Intel is the reference level
PLAIN_FEATURES = NUMERIC + [f"vendor_{v}" for v in VENDORS]
STRUCTURAL_EXTRA = [
    "log_cores_per_chip",
    "l3_x_log_cpc",
    "smt_x_log_cpc",
    "vintage_x_log_cpc",
    "vintage_sq",
]
STRUCTURAL_FEATURES = PLAIN_FEATURES + STRUCTURAL_EXTRA
GBDT_FEATURES = PLAIN_FEATURES + ["log_cores_per_chip"]
# What AutoGluon is given: the same information, untransformed.
AUTOML_FEATURES = [
    "cores",
    "chips",
    "threads_per_core",
    "mhz",
    "l2_mb_per_core",
    "l3_mb_per_core",
    "vintage_years",
    "vendor",
    "family",
]
TARGET = "log_score"

BASELINE_NAME = "Naive baseline (cores x clock)"
PLAIN_NAME = "Plain regression (OLS)"


def classify_compiler(compiler: str | None) -> str:
    """Group SPEC compiler strings into readable compiler toolchains."""
    if not compiler or not isinstance(compiler, str):
        return "Other"
    s = compiler.lower()
    if "oneapi" in s or "icx" in s or "ifx" in s:
        return "Intel oneAPI (LLVM)"
    if "aocc" in s or "amd optimizing" in s:
        return "AMD AOCC"
    if (
        "intel(r) c" in s
        or "icc" in s
        or "c++ compiler for linux" in s
        or "version 19" in s
        or "version 18" in s
        or "version 17" in s
    ):
        return "Intel Classic ICC"
    if "gcc" in s or "gnu" in s:
        return "GNU GCC"
    if "oracle" in s or "developer studio" in s or "fujitsu" in s:
        return "Oracle / Vendor Studio"
    return "Other"


def consensus_verdict(
    ols_pct: float,
    structural_pct: float,
    gbdt_pct: float,
    cores_per_chip: int,
) -> str:
    """Classify a processor across linear (OLS), L1 structural spline, and L1 GBDT models."""
    if (
        ols_pct >= 10.0
        and structural_pct >= 8.0
        and gbdt_pct >= 8.0
        and (ols_pct - gbdt_pct) < 25.0
    ):
        return "True architectural standout"
    if ols_pct >= 12.0 and (gbdt_pct < 8.0 or (ols_pct - gbdt_pct) >= 18.0):
        if cores_per_chip <= 8:
            return "Small-core linearity artifact"
        return "Linearity / regime artifact"
    if ols_pct <= -10.0 and structural_pct <= -8.0 and gbdt_pct <= -8.0:
        return "Memory / scaling bottleneck"
    return "Near expected spec curve"


def prepare(spec: pd.DataFrame) -> pd.DataFrame:
    """Add model columns and the train / validation / holdout period of every row."""
    df = spec.copy()
    df["hw_avail"] = pd.to_datetime(df["hw_avail"])
    df[TARGET] = np.log(df[config.SPEC_METRIC])
    df["log_cores"] = np.log(df["cores"])
    df["log_mhz"] = np.log(df["mhz"])
    df["log_chips"] = np.log(df["chips"])
    df["log_cores_per_chip"] = np.log(df["cores"] / df["chips"])
    df["smt"] = (df["threads_per_core"] > 1).astype(float)
    # Caches missing from a disclosure are filled with the training-free constant 0 and flagged
    # by the median below; SPEC disclosures nearly always state them.
    for col in ("l2_mb_per_core", "l3_mb_per_core"):
        df[col] = df[col].astype(float)
        df[f"log1p_{col}"] = np.log1p(df[col].fillna(df[col].median()))
    df["vintage_years"] = (df["hw_avail"] - pd.Timestamp(config.VINTAGE_ORIGIN)).dt.days / 365.25
    df["l3_x_log_cpc"] = df["log1p_l3_mb_per_core"] * df["log_cores_per_chip"]
    df["smt_x_log_cpc"] = df["smt"] * df["log_cores_per_chip"]
    df["vintage_x_log_cpc"] = df["vintage_years"] * df["log_cores_per_chip"]
    df["vintage_sq"] = df["vintage_years"] ** 2
    for v in VENDORS:
        df[f"vendor_{v}"] = (df["vendor"] == v).astype(float)

    first_seen = df.groupby("processor")["hw_avail"].transform("min")
    df["first_avail"] = first_seen
    df["period"] = np.select(
        [
            first_seen < pd.Timestamp(config.VALID_START),
            first_seen < pd.Timestamp(config.HOLDOUT_START),
        ],
        ["train", "validation"],
        default="holdout",
    )
    return df.reset_index(drop=True)


# --- the two explainable models ----------------------------------------------------------------


@dataclass
class Baseline:
    intercept: float

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Baseline":
        return cls(float((df[TARGET] - df["log_cores"] - df["log_mhz"]).mean()))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.intercept + df["log_cores"] + df["log_mhz"]).to_numpy()


@dataclass
class Plain:
    result: sm.regression.linear_model.RegressionResultsWrapper
    features: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Plain":
        # A feature with no variation in the fitting data (say, no "Other" vendor yet) carries
        # no information and would make the fit rank-deficient, so it is left out.
        features = [f for f in PLAIN_FEATURES if df[f].nunique() > 1]
        x = sm.add_constant(df[features], has_constant="add")
        return cls(sm.OLS(df[TARGET], x).fit(), features)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = sm.add_constant(df[self.features], has_constant="add")
        return self.result.predict(x).to_numpy()

    def coefficients(self) -> pd.DataFrame:
        r = self.result
        return pd.DataFrame(
            {
                "term": r.params.index,
                "coefficient": r.params.to_numpy(),
                "std_error": r.bse.to_numpy(),
                "p_value": r.pvalues.to_numpy(),
            }
        )


def _fit_predict_structural(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """L1/Huber structural spline curve capturing diminishing core scaling and cache x density."""
    features = [f for f in STRUCTURAL_FEATURES if f in train.columns and train[f].nunique() > 1]
    spline_cols = [c for c in ("log_cores", "log_cores_per_chip") if c in features]
    rest_cols = [c for c in features if c not in spline_cols]
    if spline_cols:
        ct = ColumnTransformer(
            [
                ("sp", SplineTransformer(n_knots=6, degree=2, knots="uniform"), spline_cols),
                ("rest", "passthrough", rest_cols),
            ]
        )
        pipe = make_pipeline(ct, StandardScaler(), HuberRegressor(max_iter=500, alpha=0.05))
    else:
        pipe = make_pipeline(StandardScaler(), HuberRegressor(max_iter=500, alpha=0.05))
    pipe.fit(train[features], train[TARGET])
    return pipe.predict(test[features])


def _fit_predict_gbdt(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Nonlinear L1 Gradient Boosted Decision Trees over specification features."""
    features = [f for f in GBDT_FEATURES if f in train.columns and train[f].nunique() > 1]
    gbdt = HistGradientBoostingRegressor(loss="absolute_error", max_iter=150, random_state=0)
    gbdt.fit(train[features], train[TARGET])
    return gbdt.predict(test[features])


# --- errors --------------------------------------------------------------------------------------


def mae_log(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def median_abs_pct(y: np.ndarray, pred: np.ndarray) -> float:
    """Median absolute error as a percentage of the true score."""
    return float(np.median(np.abs(np.expm1(np.asarray(pred) - np.asarray(y)))) * 100)


def score_row(name: str, period: str, y, pred, **extra) -> dict:
    return {
        "model": name,
        "period": period,
        "n_results": int(len(y)),
        "mae_log": mae_log(y, pred),
        "median_abs_pct_error": median_abs_pct(y, pred),
        **extra,
    }


def time_fold_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window folds over the pre-holdout period.

    Processors are cut into N+1 blocks by first availability. Fold k trains on blocks before k
    and tests on block k. The spread of (baseline error - plain error) across folds is the
    noise level a holdout improvement has to clear.
    """
    pre = df[df["period"] != "holdout"]
    dates = pre.drop_duplicates("processor")["first_avail"].sort_values()
    edges = dates.quantile(np.linspace(0, 1, config.N_TIME_FOLDS + 2)).to_numpy()
    rows = []
    for k in range(1, config.N_TIME_FOLDS + 1):
        train = pre[pre["first_avail"] < edges[k]]
        upper = (
            pre["first_avail"] <= edges[k + 1]
            if k == config.N_TIME_FOLDS
            else (pre["first_avail"] < edges[k + 1])
        )
        test = pre[(pre["first_avail"] >= edges[k]) & upper]
        if len(train) < 30 or len(test) < 10:
            continue
        base = mae_log(test[TARGET], Baseline.fit(train).predict(test))
        plain = mae_log(test[TARGET], Plain.fit(train).predict(test))
        rows.append(
            {
                "fold": k,
                "train_until": pd.Timestamp(edges[k]).date().isoformat(),
                "n_train": len(train),
                "n_test": len(test),
                "baseline_mae_log": base,
                "plain_mae_log": plain,
                "improvement": base - plain,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class Evaluation:
    leaderboard: pd.DataFrame  # validation scores; the baseline is always a row
    holdout: pd.DataFrame  # one touch, after the pick
    folds: pd.DataFrame
    fold_spread: float
    verdict: dict
    counts: dict


def evaluate(df: pd.DataFrame, automl: dict | None = None) -> Evaluation:
    """Score baseline and plain model; fold in AutoGluon's saved results when they exist.

    `automl` is the summary written by automl.py: its validation leaderboard rows plus the
    holdout score of the model validation picked.
    """
    train = df[df["period"] == "train"]
    valid = df[df["period"] == "validation"]
    hold = df[df["period"] == "holdout"]
    counts = {
        p: {"results": int(len(part)), "processors": int(part["processor"].nunique())}
        for p, part in (("train", train), ("validation", valid), ("holdout", hold))
    }
    if min(len(train), len(valid), len(hold)) < 20:
        raise ValueError(f"not enough results in every period to evaluate: {counts}")

    base, plain = Baseline.fit(train), Plain.fit(train)
    board = [
        score_row(BASELINE_NAME, "validation", valid[TARGET], base.predict(valid), kind="baseline"),
        score_row(PLAIN_NAME, "validation", valid[TARGET], plain.predict(valid), kind="plain"),
    ]
    held = [
        score_row(BASELINE_NAME, "holdout", hold[TARGET], base.predict(hold), kind="baseline"),
        score_row(PLAIN_NAME, "holdout", hold[TARGET], plain.predict(hold), kind="plain"),
    ]
    if automl:
        board += automl["leaderboard"]
        held.append(automl["holdout"])

    leaderboard = pd.DataFrame(board).sort_values("mae_log").reset_index(drop=True)
    leaderboard.insert(0, "rank", leaderboard.index + 1)
    holdout = pd.DataFrame(held)

    folds = time_fold_spread(df)
    spread = float(folds["improvement"].std(ddof=1)) if len(folds) > 1 else float("nan")

    # The pick is made on validation only. The holdout row of the pick decides the verdict.
    picked = leaderboard.iloc[0]["model"]
    picked_kind = leaderboard.iloc[0]["kind"]
    hold_by_kind = holdout.set_index("kind")["mae_log"]
    base_mae = float(hold_by_kind["baseline"])
    pick_mae = float(hold_by_kind.get(picked_kind, base_mae))
    improvement = base_mae - pick_mae
    # NaN spread (too few folds) never passes: no measured noise level, no claim.
    beats = bool(picked_kind != "baseline" and improvement > spread)
    verdict = {
        "picked_on_validation": picked,
        "holdout_mae_log_baseline": base_mae,
        "holdout_mae_log_picked": pick_mae,
        "holdout_improvement": improvement,
        "fold_spread": spread,
        "beats_baseline": beats,
        "published_model": picked if beats else BASELINE_NAME,
        "statement": (
            f"{picked} beats the cores-times-clock baseline on the newest generation by more "
            "than the spread between folds."
            if beats
            else "No signal found: the model picked on validation does not beat the "
            "cores-times-clock baseline on the newest generation by more than the spread "
            "between folds."
        ),
    }
    holdout["published"] = holdout["model"] == verdict["published_model"]
    return Evaluation(leaderboard, holdout, folds, spread, verdict, counts)


# --- residual ranking ------------------------------------------------------------------------


def _predict_held_out(out: pd.DataFrame, train_idx, test_idx) -> tuple:
    tr, te = out.iloc[train_idx], out.iloc[test_idx]
    return (
        test_idx,
        Plain.fit(tr).predict(te),
        _fit_predict_structural(tr, te),
        _fit_predict_gbdt(tr, te),
    )


def out_of_fold_residuals(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """Leave-one-processor-out predictions: every result is scored by models that never saw
    any result for its processor.

    Computes three held-out reference curves, refitting all three once per processor:
      1. Plain OLS (`predicted_log` / `residual_log`) — constant-elasticity linear specification
      2. L1 Structural Spline (`structural_pred_log` / `structural_residual_log`) — diminishing
         returns on core density plus L3 cache and SMT interactions
      3. Nonlinear L1 GBDT (`gbdt_pred_log` / `gbdt_residual_log`) — tree-based median curve
    """
    out = df.copy()
    if "log_cores_per_chip" not in out.columns:
        out["log_cores_per_chip"] = np.log(out["cores"] / out["chips"])
        out["l3_x_log_cpc"] = out["log1p_l3_mb_per_core"] * out["log_cores_per_chip"]
        out["smt_x_log_cpc"] = out["smt"] * out["log_cores_per_chip"]
        out["vintage_x_log_cpc"] = out["vintage_years"] * out["log_cores_per_chip"]
        out["vintage_sq"] = out["vintage_years"] ** 2
    cols = ["predicted_log", "structural_pred_log", "gbdt_pred_log"]
    for c in cols:
        out[c] = np.nan
    splits = LeaveOneGroupOut().split(out, groups=out["processor"])
    held_out = Parallel(n_jobs=n_jobs)(delayed(_predict_held_out)(out, tr, te) for tr, te in splits)
    for test_idx, *preds in held_out:
        for c, pred in zip(cols, preds, strict=True):
            out.iloc[test_idx, out.columns.get_loc(c)] = pred
    out["residual_log"] = out[TARGET] - out["predicted_log"]
    out["structural_residual_log"] = out[TARGET] - out["structural_pred_log"]
    out["gbdt_residual_log"] = out[TARGET] - out["gbdt_pred_log"]
    return out


def residual_ranking(oof: pd.DataFrame) -> pd.DataFrame:
    """One row per processor: multi-model residual consensus, tuning spread, and cache scaling.

    A positive residual is a question, not an answer. Comparing the plain OLS residual against
    the L1 Structural Spline and Nonlinear L1 GBDT residuals separates true architectural
    standouts from 4-core OLS linearity artifacts, while within-processor tuning spread and
    compiler family separate silicon performance from OEM/compiler tuning.
    """
    has_struct = "structural_residual_log" in oof.columns
    has_gbdt = "gbdt_residual_log" in oof.columns
    rows = []
    for processor, part in oof.groupby("processor"):
        if len(part) < config.MIN_RESULTS_FOR_RANK:
            continue
        med = part["residual_log"].median()
        typical = part.loc[(part["residual_log"] - med).abs().idxmin()]
        ols_pct = float(np.expm1(med) * 100)
        res_min = float(np.expm1(part["residual_log"].min()) * 100)
        res_max = float(np.expm1(part["residual_log"].max()) * 100)
        struct_pct = (
            float(np.expm1(part["structural_residual_log"].median()) * 100)
            if has_struct
            else ols_pct
        )
        gbdt_pct = (
            float(np.expm1(part["gbdt_residual_log"].median()) * 100) if has_gbdt else ols_pct
        )
        cores_per_chip = int(round(typical["cores"] / typical["chips"]))
        verdict = consensus_verdict(ols_pct, struct_pct, gbdt_pct, cores_per_chip)

        # Peak vs Base compiler flag headroom on disclosures that ran Peak (> Base).
        peak_gain_pct = None
        if "peak_score" in part.columns:
            valid_peak = part[
                (part["peak_score"] > part[config.SPEC_METRIC]) & (part[config.SPEC_METRIC] > 0)
            ]
            if not valid_peak.empty:
                gains = (valid_peak["peak_score"] / valid_peak[config.SPEC_METRIC] - 1.0) * 100.0
                peak_gain_pct = float(gains.median())

        n_oems = (
            int(part["hardware_vendor"].dropna().nunique())
            if "hardware_vendor" in part.columns
            else 1
        )
        l3_per_core = (
            float(typical["l3_mb_per_core"])
            if "l3_mb_per_core" in typical and pd.notna(typical["l3_mb_per_core"])
            else None
        )
        l2_per_core = (
            float(typical["l2_mb_per_core"])
            if "l2_mb_per_core" in typical and pd.notna(typical["l2_mb_per_core"])
            else None
        )
        score_per_core = float((part[config.SPEC_METRIC] / part["cores"]).median())
        ghz = float(typical["mhz"]) / 1000.0 if float(typical["mhz"]) > 0 else 1.0
        score_per_core_ghz = float(score_per_core / ghz)

        rows.append(
            {
                "processor": processor,
                "vendor": typical["vendor"],
                "family": typical["family"],
                "first_avail": typical["first_avail"].date().isoformat(),
                "n_results": int(len(part)),
                "n_oems": n_oems,
                "cores_per_chip": cores_per_chip,
                "mhz": float(typical["mhz"]),
                "l2_mb_per_core": l2_per_core,
                "l3_mb_per_core": l3_per_core,
                "median_score": float(part[config.SPEC_METRIC].median()),
                "score_per_core": score_per_core,
                "score_per_core_ghz": score_per_core_ghz,
                "residual_pct": ols_pct,
                "structural_residual_pct": struct_pct,
                "gbdt_residual_pct": gbdt_pct,
                "consensus_verdict": verdict,
                "residual_pct_min": res_min,
                "residual_pct_max": res_max,
                "tuning_spread_pct": float(res_max - res_min),
                "peak_gain_pct": peak_gain_pct,
                "compiler_family": classify_compiler(typical.get("compiler")),
                "config_system": typical.get("system"),
                "config_chips": int(typical["chips"]),
                "config_memory": typical.get("memory_text"),
                "config_compiler": typical.get("compiler"),
                "config_os": typical.get("os"),
                "config_test_date": (
                    typical["test_date"].date().isoformat()
                    if pd.notna(typical.get("test_date"))
                    else None
                ),
                "disclosure_url": typical.get("disclosure_url"),
            }
        )
    cols = ["residual_pct"]
    return pd.DataFrame(rows).sort_values(cols, ascending=False).reset_index(drop=True)
