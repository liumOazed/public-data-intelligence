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
from sklearn.model_selection import GroupKFold

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


def prepare(spec: pd.DataFrame) -> pd.DataFrame:
    """Add model columns and the train / validation / holdout period of every row."""
    df = spec.copy()
    df["hw_avail"] = pd.to_datetime(df["hw_avail"])
    df[TARGET] = np.log(df[config.SPEC_METRIC])
    df["log_cores"] = np.log(df["cores"])
    df["log_mhz"] = np.log(df["mhz"])
    df["log_chips"] = np.log(df["chips"])
    df["smt"] = (df["threads_per_core"] > 1).astype(float)
    # Caches missing from a disclosure are filled with the training-free constant 0 and flagged
    # by the median below; SPEC disclosures nearly always state them.
    for col in ("l2_mb_per_core", "l3_mb_per_core"):
        df[col] = df[col].astype(float)
        df[f"log1p_{col}"] = np.log1p(df[col].fillna(df[col].median()))
    df["vintage_years"] = (df["hw_avail"] - pd.Timestamp(config.VINTAGE_ORIGIN)).dt.days / 365.25
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


def out_of_fold_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Plain-model predictions for every result from a model that never saw its processor."""
    out = df.copy()
    out["predicted_log"] = np.nan
    n_groups = out["processor"].nunique()
    folds = GroupKFold(n_splits=min(config.N_RESIDUAL_FOLDS, n_groups))
    for train_idx, test_idx in folds.split(out, groups=out["processor"]):
        model = Plain.fit(out.iloc[train_idx])
        out.iloc[test_idx, out.columns.get_loc("predicted_log")] = model.predict(out.iloc[test_idx])
    out["residual_log"] = out[TARGET] - out["predicted_log"]
    return out


def residual_ranking(oof: pd.DataFrame) -> pd.DataFrame:
    """One row per processor: median out-of-fold residual and the configuration behind it.

    A positive residual is a question, not an answer. The configuration shown is the one
    of the result closest to the processor's median, so the reader can check memory,
    compiler and operating system before reading anything into the number.
    """
    rows = []
    for processor, part in oof.groupby("processor"):
        if len(part) < config.MIN_RESULTS_FOR_RANK:
            continue
        med = part["residual_log"].median()
        typical = part.loc[(part["residual_log"] - med).abs().idxmin()]
        rows.append(
            {
                "processor": processor,
                "vendor": typical["vendor"],
                "family": typical["family"],
                "first_avail": typical["first_avail"].date().isoformat(),
                "n_results": int(len(part)),
                "cores_per_chip": int(round(typical["cores"] / typical["chips"])),
                "mhz": float(typical["mhz"]),
                "median_score": float(part[config.SPEC_METRIC].median()),
                "residual_pct": float(np.expm1(med) * 100),
                "residual_pct_min": float(np.expm1(part["residual_log"].min()) * 100),
                "residual_pct_max": float(np.expm1(part["residual_log"].max()) * 100),
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
