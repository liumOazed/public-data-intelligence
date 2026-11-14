"""Fair price: what a part would cost if price followed benchmark performance alone.

For each kind of part (processors and graphics cards separately, because their Blender scores
are on different scales) the logarithm of price is modelled from the logarithm of the score.
A part priced below its prediction is cheap for what it does; one priced above is paying for
something the benchmark does not measure, or is simply overpriced.

The sample is a few dozen parts, so the rules are strict and simple:

* A naive baseline is always on the leaderboard: every part costs the same per point of score.
* Every model is scored by leave-one-out prediction, so a part's fair price always comes from
  a fit that never saw that part.
* Only parts with at least two listings are used for fitting. A single listing is often one
  marketplace seller's asking price for a discontinued part, and would bend the line.
* A model replaces the baseline only when its leave-one-out error is lower by more than
  twice the standard error of the gain. Otherwise the baseline sets the fair price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pdi.tech_value import config

BASELINE = "Naive baseline (same price per point)"
Fit = Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]]


def prepare(table: pd.DataFrame) -> pd.DataFrame:
    """Priced, scored parts with the columns the models use."""
    if table.empty:
        return pd.DataFrame()
    df = table[table["score"].notna() & table["price_usd"].notna()].copy()
    df["log_price"] = np.log(df["price_usd"].astype(float))
    df["log_score"] = np.log(df["score"].astype(float))
    df["vendor"] = df["model_key"].str.split().str[0]
    df["trusted"] = df["n_listings"] >= config.FAIR_PRICE_MIN_LISTINGS
    return df.reset_index(drop=True)


def _design(df: pd.DataFrame, vendors: list[str]) -> np.ndarray:
    cols = [np.ones(len(df)), df["log_score"].to_numpy()]
    cols += [(df["vendor"] == v).to_numpy(float) for v in vendors]
    return np.column_stack(cols)


def fit_baseline(train: pd.DataFrame):
    a = float(np.median(train["log_price"] - train["log_score"]))
    return lambda df: a + df["log_score"].to_numpy()


def _fit_line(train: pd.DataFrame, with_vendor: bool):
    # The first vendor is the reference level; a vendor seen once cannot have its own level.
    counts = train["vendor"].value_counts()
    vendors = sorted(counts[counts >= 2].index)[1:] if with_vendor else []
    beta, *_ = np.linalg.lstsq(_design(train, vendors), train["log_price"].to_numpy(), rcond=None)
    return lambda df: _design(df, vendors) @ beta


def fit_line(train: pd.DataFrame):
    return _fit_line(train, with_vendor=False)


def fit_line_vendor(train: pd.DataFrame):
    return _fit_line(train, with_vendor=True)


def fit_forest(train: pd.DataFrame):
    vendors = sorted(train["vendor"].unique())

    def features(df: pd.DataFrame) -> np.ndarray:
        return _design(df, vendors)[:, 1:]

    forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0)
    forest.fit(features(train), train["log_price"].to_numpy())
    return lambda df: forest.predict(features(df))


MODELS: dict[str, Fit] = {
    BASELINE: fit_baseline,
    "Straight line (log price on log score)": fit_line,
    "Straight line with brand": fit_line_vendor,
    "Random forest": fit_forest,
}


def leave_one_out(train: pd.DataFrame, fit: Fit) -> np.ndarray:
    out = np.empty(len(train))
    for i in range(len(train)):
        rest = train.drop(train.index[i])
        out[i] = fit(rest)(train.iloc[[i]])[0]
    return out


@dataclass
class Result:
    leaderboard: pd.DataFrame
    parts: pd.DataFrame
    verdicts: dict


def evaluate(table: pd.DataFrame) -> Result:
    df = prepare(table)
    boards, parts, verdicts = [], [], {}
    for kind in ("gpu", "cpu"):
        part = df[df["kind"] == kind] if len(df) else df
        train = part[part["trusted"]] if len(part) else part
        if len(train) < config.FAIR_PRICE_MIN_PARTS:
            verdicts[kind] = {
                "statement": f"Only {len(train)} parts have two or more listings, fewer than "
                f"the {config.FAIR_PRICE_MIN_PARTS} needed to fit a fair price.",
                "n_parts": int(len(train)),
            }
            continue
        y = train["log_price"].to_numpy()
        oof = {name: leave_one_out(train, fit) for name, fit in MODELS.items()}
        err = {name: np.abs(y - pred) for name, pred in oof.items()}
        best = min(err, key=lambda name: err[name].mean())
        gain = err[BASELINE] - err[best]
        se = float(gain.std(ddof=1) / np.sqrt(len(gain))) if best != BASELINE else 0.0
        beats = best != BASELINE and float(gain.mean()) > 2 * se
        # A model that does not clearly beat the baseline is not used: the simpler rule wins.
        picked = best if beats else BASELINE
        for name in MODELS:
            boards.append(
                {
                    "kind": kind,
                    "model": name,
                    "n_parts": int(len(train)),
                    "mae_log": float(err[name].mean()),
                    "median_abs_pct_error": float(np.median(np.expm1(err[name])) * 100),
                    "used": name == picked,
                    "is_baseline": name == BASELINE,
                }
            )
        typical = float(np.median(np.expm1(err[picked])) * 100)
        verdicts[kind] = {
            "used": picked,
            "best_challenger": best if best != BASELINE else None,
            "n_parts": int(len(train)),
            "mae_log_baseline": float(err[BASELINE].mean()),
            "mae_log_best": float(err[best].mean()),
            "gain": float(gain.mean()),
            "gain_standard_error": se,
            "beats_baseline": bool(beats),
            "typical_error_pct": typical,
            "statement": (
                f"{picked} predicts a part's price within {typical:.0f}% for a typical part "
                f"and beats the same-price-per-point baseline by more than twice the "
                f"standard error of the gain."
                if beats
                else f"No model clearly beats the same-price-per-point baseline on "
                f"{len(train)} parts, so the baseline sets the fair price. It is typically "
                f"off by {typical:.0f}%, so read the gaps as rough."
            ),
        }
        # Trusted parts get their out-of-fold prediction; the rest are predicted by the fit
        # on all trusted parts.
        full = MODELS[picked](train)
        fair = pd.Series(np.exp(full(part)), index=part.index)
        fair.loc[train.index] = np.exp(oof[picked])
        out = part.assign(fair_price_usd=fair)
        out["gap_pct"] = (out["price_usd"] / out["fair_price_usd"] - 1) * 100
        out["confidence"] = np.where(out["trusted"], "two or more listings", "single listing")
        out["reading"] = np.select(
            [
                out["gap_pct"] <= -config.FAIR_PRICE_GAP_PCT,
                out["gap_pct"] >= config.FAIR_PRICE_GAP_PCT,
            ],
            ["below fair price", "above fair price"],
            default="near fair price",
        )
        parts.append(out)
    cols = [
        "kind",
        "model_key",
        "price_usd",
        "fair_price_usd",
        "gap_pct",
        "reading",
        "confidence",
        "score",
        "n_listings",
        "n_retailers",
        "retailer",
        "listing_url",
    ]
    all_parts = (
        pd.concat(parts)[cols].sort_values(["kind", "gap_pct"]).reset_index(drop=True)
        if parts
        else pd.DataFrame(columns=cols)
    )
    return Result(pd.DataFrame(boards), all_parts, verdicts)
