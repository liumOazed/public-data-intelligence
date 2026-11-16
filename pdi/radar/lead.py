"""The Demand Radar test, as fixed in pdi/radar/config.py and on the Decisions page.

For each pair: does last month's change in pageviews (on a year earlier) improve a forecast
of this month's change in retail sales (on a year earlier) over last month's change in sales?
Fitted on months to TRAIN_END, scored once on the months after.

Run as `python -m pdi.radar.lead`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pdi.common import findings
from pdi.common.export import write_table
from pdi.common.fetch import utc_now
from pdi.radar import config


def yoy(values: pd.Series) -> pd.Series:
    """100 x log change on twelve months earlier, on a complete monthly index."""
    s = values.sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
    return 100 * np.log(s.where(s > 0)).diff(12)


def test_one(views: pd.Series, sales: pd.Series) -> dict:
    frame = pd.DataFrame({"sales": yoy(sales), "views": yoy(views)})
    frame["sales_lag"] = frame["sales"].shift(1)
    frame["views_lag"] = frame["views"].shift(1)
    data = frame.dropna()
    split = pd.Timestamp(config.TRAIN_END + "-01")
    train, test = data[data.index <= split], data[data.index > split]
    out = {"train_months": int(len(train)), "test_months": int(len(test))}
    if len(train) < config.MIN_TRAIN_MONTHS or len(test) < config.MIN_TEST_MONTHS:
        return out | {"verdict": "not enough data", "passes": False}
    x = np.column_stack([np.ones(len(train)), train["sales_lag"], train["views_lag"]])
    coef, *_ = np.linalg.lstsq(x, train["sales"].to_numpy(), rcond=None)
    pred = coef[0] + coef[1] * test["sales_lag"] + coef[2] * test["views_lag"]
    err_model = (test["sales"] - pred).abs()
    err_base = (test["sales"] - test["sales_lag"]).abs()
    gain = err_base - err_model
    se = float(gain.std(ddof=1) / np.sqrt(len(gain)))
    passes = bool(gain.mean() > config.SE_MULTIPLE * se)
    same_month = data["sales"].corr(data["views"])
    return out | {
        "train_start": train.index.min().strftime("%Y-%m"),
        "test_start": test.index.min().strftime("%Y-%m"),
        "test_end": test.index.max().strftime("%Y-%m"),
        "mae_baseline": float(err_base.mean()),
        "mae_model": float(err_model.mean()),
        "gain": float(gain.mean()),
        "gain_se": se,
        "coef_views_lag": float(coef[2]),
        "corr_same_month": float(same_month),
        "passes": passes,
        "verdict": "pageviews add information" if passes else "no signal found",
    }


def run(views: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, (name, articles) in config.PAIRS.items():
        v = views[views["article"].isin(articles)].groupby("period")["views"].sum()
        s = sales[sales["category"] == code].set_index("period")["sales_musd"]
        rows.append({"category": code, "name": name, "articles": len(articles),
                     **test_one(v, s)})  # fmt: skip
    return pd.DataFrame(rows)


def finding(result: pd.DataFrame) -> dict:
    passed = result.loc[result["passes"], "name"].tolist()
    if passed:
        return findings.from_test(
            True,
            passed=f"For {', '.join(passed)}, adding pageviews beat the persistence baseline on "
            "months the model never saw, by more than twice the standard error of the gain. One "
            "of four tests at this bar can pass by chance.",
            failed="",
            otherwise=findings.OBSERVATION,
        )
    return findings.observation(
        "No signal found: in none of the four pairs did pageviews beat the persistence baseline "
        "by more than twice the standard error of the gain."
    )


def main() -> None:
    views = pd.read_parquet(config.PAGEVIEWS)
    sales = pd.read_parquet(config.SALES)
    result = run(views, sales)
    retrieved = json.loads((config.STORE_DIR / "retrieved.json").read_text())
    config.RESULT.write_text(
        json.dumps({"run_at": utc_now().isoformat(), "rows": result.to_dict("records")},
                   indent=1, default=str) + "\n"
    )  # fmt: skip
    write_table(
        config.MODULE, "lead_test", result, sources=["wikimedia_pageviews", "census_marts"],
        retrieved_at=retrieved,
        extra={"finding": finding(result), "one_off": True, "train_end": config.TRAIN_END,
               "se_multiple": config.SE_MULTIPLE},
        description="Demand Radar experiment: does last month's change in Wikipedia pageviews "
        "improve a forecast of this month's change in US retail sales, by kind of business?",
    )  # fmt: skip
    for _, r in result.iterrows():
        print(f"{r['category']} {r['name']}: {r['verdict']}")


if __name__ == "__main__":
    main()
