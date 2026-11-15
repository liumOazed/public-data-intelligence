"""The lead-time test: do US-reported imports say something about an economy's own exports
before its own figure is out? The rules are in pdi/trade/config.py and were written first."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdi.trade import config


def yoy(values: pd.Series) -> pd.Series:
    """100 x log change on twelve months earlier, on a complete monthly index."""
    s = values.sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
    return 100 * np.log(s.where(s > 0)).diff(12)


def test_one(us_imports: pd.Series, own_exports: pd.Series) -> dict:
    """Run the pre-registered test for one economy. Both inputs are monthly levels."""
    frame = pd.DataFrame({"us": yoy(us_imports), "own": yoy(own_exports)})
    frame["own_lag"] = frame["own"].shift(1)
    corr = {
        str(k): _corr(frame["own"], frame["us"].shift(k)) for k in config.LEAD_CORRELATION_SHIFTS
    }
    data = frame.dropna()
    split = pd.Timestamp(config.LEAD_TRAIN_END + "-01")
    train, test = data[data.index <= split], data[data.index > split]
    out = {"train_months": int(len(train)), "test_months": int(len(test)), "correlation": corr}
    if len(train) < config.LEAD_MIN_TRAIN_MONTHS or len(test) < config.LEAD_MIN_TEST_MONTHS:
        return out | {"verdict": "not enough data", "passes": False}
    x = np.column_stack([np.ones(len(train)), train["own_lag"], train["us"]])
    coef, *_ = np.linalg.lstsq(x, train["own"].to_numpy(), rcond=None)
    pred = coef[0] + coef[1] * test["own_lag"] + coef[2] * test["us"]
    err_model = (test["own"] - pred).abs()
    err_base = (test["own"] - test["own_lag"]).abs()
    gain = err_base - err_model
    se = float(gain.std(ddof=1) / np.sqrt(len(gain)))
    passes = bool(gain.mean() > config.LEAD_SE_MULTIPLE * se)
    return out | {
        "test_start": test.index.min().strftime("%Y-%m"),
        "test_end": test.index.max().strftime("%Y-%m"),
        "mae_baseline": float(err_base.mean()),
        "mae_model": float(err_model.mean()),
        "gain": float(gain.mean()),
        "gain_se": se,
        "coef_us_imports": float(coef[2]),
        "coef_own_lag": float(coef[1]),
        "passes": passes,
        "verdict": "US imports add information" if passes else "no signal found",
    }


def _corr(a: pd.Series, b: pd.Series) -> float | None:
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) < 24:
        return None
    return float(both.iloc[:, 0].corr(both.iloc[:, 1]))


def run(totals: pd.DataFrame, macro_obs: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for country in config.SEA:
        us = totals[totals["partner"] == country].set_index("period")["value"]
        own = pd.Series(dtype=float)
        if macro_obs is not None:
            own = macro_obs[macro_obs["series_id"] == f"{country}.exports_usd"].set_index("period")[
                "value"
            ]
        if us.empty or own.empty:
            rows.append({"country": country, "verdict": "not run: exports not collected yet",
                         "passes": False})  # fmt: skip
            continue
        rows.append({"country": country, **test_one(us, own)})
    return pd.DataFrame(rows)
