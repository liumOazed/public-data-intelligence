"""Forecast test with AutoGluon TimeSeries: does any model forecast next month's move better
than a naive rule, on months it never saw?

Runs weekly, not on every pipeline run, on CPU with a fixed time limit. The procedure is fixed
in pdi/macro/config.py:

1. Fit every model in AUTOML_MODELS on data before VALID_START.
2. For each validation month, forecast that month one step ahead from the data before it,
   with the fitted models (no refitting on later data). Rank models on that.
3. The baseline is whichever of the four rule-of-thumb models did best on validation.
4. Score the pick and the naive models once on the holdout months, the same way.
5. The pick passes only if it beats the baseline on the holdout by more than the spread of
   its improvement across the quarterly validation folds.

Errors are scaled by each series' mean absolute month-to-month change in the training period,
so a series in percent and one in percentage points can be averaged.
"""

from __future__ import annotations

import json
import shutil
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd

from pdi.common.fetch import utc_now
from pdi.common.paths import MODELS
from pdi.macro import config, flags, store

SERIES_START = "2000-01-01"


def target_frame(obs: pd.DataFrame) -> pd.DataFrame:
    """Long table (item_id, timestamp, target): each series' monthly move."""
    cat = config.catalogue()
    frames = []
    for sid in sorted(obs["series_id"].unique()):
        x = flags.moves(flags.monthly_values(obs, sid), cat[sid].transform)
        x = x[x.index >= SERIES_START].dropna()
        if len(x) < 48:
            continue
        frames.append(pd.DataFrame({"item_id": sid, "timestamp": x.index, "target": x.values}))
    return pd.concat(frames, ignore_index=True)


def scales(long: pd.DataFrame, before: str) -> pd.Series:
    train = long[long["timestamp"] < before].sort_values(["item_id", "timestamp"])
    s = train.groupby("item_id")["target"].apply(lambda t: t.diff().abs().mean())
    return s[s > 0]


def months_between(start: str, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="MS"))


def rolling_errors(predictor, long, months, models, scale) -> pd.DataFrame:
    """One-step forecasts for each month in `months`, from data strictly before it."""
    from autogluon.timeseries import TimeSeriesDataFrame

    rows = []
    for m in months:
        ctx = long[long["timestamp"] < m]
        last = ctx.groupby("item_id")["timestamp"].max()
        # Only series whose latest value is the month before m: their next value is m.
        items = last[last == m - pd.DateOffset(months=1)].index.intersection(scale.index)
        actual = long[(long["timestamp"] == m) & long["item_id"].isin(items)].set_index("item_id")
        items = actual.index
        if len(items) == 0:
            continue
        data = TimeSeriesDataFrame.from_data_frame(ctx[ctx["item_id"].isin(items)])
        for name in models:
            pred = predictor.predict(data, model=name)["mean"].droplevel("timestamp")
            err = (pred.reindex(items) - actual["target"]).abs() / scale.reindex(items)
            for item, e in err.items():
                rows.append({"model": name, "month": m, "item_id": item, "scaled_error": e})
    return pd.DataFrame(rows).dropna()


def board(errors: pd.DataFrame, period: str) -> list[dict]:
    g = errors.groupby("model")["scaled_error"]
    out = pd.DataFrame({"scaled_mae": g.mean(), "n_forecasts": g.size()}).reset_index()
    out["period"] = period
    out["kind"] = np.where(out["model"].isin(config.NAIVE_MODELS), "naive baseline", "automl")
    return out.sort_values("scaled_mae").to_dict(orient="records")


def fold_spread(errors: pd.DataFrame, picked: str, baseline: str) -> tuple[float, list[dict]]:
    months = sorted(errors["month"].unique())
    folds = np.array_split(np.array(months), config.N_VALID_FOLDS)
    rows = []
    for i, f in enumerate(folds, 1):
        e = errors[errors["month"].isin(f)]
        p = e.loc[e["model"] == picked].set_index(["month", "item_id"])["scaled_error"]
        b = e.loc[e["model"] == baseline].set_index(["month", "item_id"])["scaled_error"]
        both = p.index.intersection(b.index)
        rows.append(
            {
                "fold": i,
                "months": f"{pd.Timestamp(f[0]):%b %Y} to {pd.Timestamp(f[-1]):%b %Y}",
                "n_forecasts": int(len(both)),
                "baseline_scaled_mae": float(b[both].mean()),
                "picked_scaled_mae": float(p[both].mean()),
            }
        )
        rows[-1]["improvement"] = rows[-1]["baseline_scaled_mae"] - rows[-1]["picked_scaled_mae"]
    spread = float(np.std([r["improvement"] for r in rows], ddof=1))
    return spread, rows


def run(obs: pd.DataFrame, time_limit: int = config.AUTOML_TIME_LIMIT_S) -> dict:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    long = target_frame(obs)
    end = long["timestamp"].max()
    scale = scales(long, config.VALID_START)
    train = TimeSeriesDataFrame.from_data_frame(long[long["timestamp"] < config.VALID_START])

    model_dir = MODELS / "macro"
    shutil.rmtree(model_dir, ignore_errors=True)
    predictor = TimeSeriesPredictor(
        prediction_length=1,
        freq="MS",
        target="target",
        eval_metric="MASE",
        path=str(model_dir),
        verbosity=1,
    ).fit(
        train,
        hyperparameters=config.AUTOML_MODELS,
        time_limit=time_limit,
        num_val_windows=3,
        enable_ensemble=True,
    )
    models = predictor.model_names()
    print(f"fitted: {models}")

    valid_months = months_between(
        config.VALID_START, pd.Timestamp(config.HOLDOUT_START) - pd.DateOffset(months=1)
    )
    valid = rolling_errors(predictor, long, valid_months, models, scale)
    leaderboard = board(valid, "validation")
    picked = leaderboard[0]["model"]
    naive_scores = {
        r["model"]: r["scaled_mae"] for r in leaderboard if r["model"] in config.NAIVE_MODELS
    }
    baseline = min(naive_scores, key=naive_scores.get)
    spread, folds = fold_spread(valid, picked, baseline)

    # The single holdout touch, after the pick.
    hold_months = months_between(config.HOLDOUT_START, end)
    hold_models = list(dict.fromkeys([picked, *config.NAIVE_MODELS]))
    hold = rolling_errors(predictor, long, hold_months, hold_models, scale)
    p = hold.loc[hold["model"] == picked].set_index(["month", "item_id"])["scaled_error"]
    b = hold.loc[hold["model"] == baseline].set_index(["month", "item_id"])["scaled_error"]
    both = p.index.intersection(b.index)
    improvement = float(b[both].mean() - p[both].mean())
    passed = bool(picked != baseline and improvement > 0 and improvement > spread)

    # Next month's forecast for every series, from all data, with the published model.
    published = picked if passed else baseline
    data = TimeSeriesDataFrame.from_data_frame(long)
    fc = predictor.predict(data, model=published)["mean"].reset_index()
    fc.columns = ["item_id", "target_month", "forecast"]
    base_fc = predictor.predict(data, model=baseline)["mean"].reset_index()
    fc["baseline_forecast"] = base_fc["mean"].to_numpy()

    per_series = (
        hold[hold["model"].isin([picked, baseline])]
        .pivot_table(index="item_id", columns="model", values="scaled_error", aggfunc="mean")
        .reset_index()
    )

    return {
        "summary": {
            "trained_at": utc_now().isoformat(),
            "autogluon_version": version("autogluon.timeseries"),
            "python_version": sys.version.split()[0],
            "time_limit_s": time_limit,
            "models_tried": list(config.AUTOML_MODELS),
            "n_series": int(long["item_id"].nunique()),
            "valid_start": config.VALID_START,
            "holdout_start": config.HOLDOUT_START,
            "data_end": f"{end:%Y-%m}",
            "picked": picked,
            "baseline": baseline,
            "published": published,
            "holdout_scaled_mae_picked": float(p[both].mean()),
            "holdout_scaled_mae_baseline": float(b[both].mean()),
            "holdout_n_forecasts": int(len(both)),
            "holdout_improvement": improvement,
            "fold_spread": spread,
            "passed": passed,
            "folds": folds,
        },
        "leaderboard": pd.DataFrame(leaderboard),
        "holdout": pd.DataFrame(board(hold, "holdout")),
        "per_series": per_series,
        "forecasts": fc,
    }


def main() -> None:
    obs, _ = store.load()
    out = run(obs)
    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    config.AUTOML_SUMMARY.write_text(json.dumps(out["summary"], indent=1) + "\n")
    out["leaderboard"].to_parquet(config.AUTOML_LEADERBOARD, index=False)
    out["holdout"].to_parquet(config.AUTOML_HOLDOUT, index=False)
    out["forecasts"].to_parquet(config.AUTOML_FORECASTS, index=False)
    out["per_series"].to_parquet(config.STORE_DIR / "automl_per_series.parquet", index=False)
    s = out["summary"]
    print(
        f"picked on validation: {s['picked']}; baseline: {s['baseline']}; holdout improvement "
        f"{s['holdout_improvement']:.3f} against fold spread {s['fold_spread']:.3f}; "
        f"passed: {s['passed']}"
    )


if __name__ == "__main__":
    main()
