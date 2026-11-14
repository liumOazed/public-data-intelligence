"""Scheduled model selection with AutoGluon Tabular.

Runs weekly, not on every pipeline run, with a fixed time limit on CPU. AutoGluon trains on
the training period and ranks its models on the validation period; nothing here tunes on
the holdout. The one model validation picks is then scored once on the holdout, and the
result is saved for the export step, which puts it on the same leaderboard as the naive
baseline and the plain regression.

The AutoML score is not what the ranking page uses. It shows how much signal the plain
model leaves behind.
"""

from __future__ import annotations

import json
import shutil
import sys
from importlib.metadata import version

import pandas as pd

from pdi.common.fetch import utc_now
from pdi.common.paths import MODELS
from pdi.tech_value import config
from pdi.tech_value.model import AUTOML_FEATURES, TARGET, prepare, score_row


def run(spec: pd.DataFrame, time_limit: int = config.AUTOML_TIME_LIMIT_S) -> dict:
    from autogluon.tabular import TabularPredictor

    df = prepare(spec)
    cols = AUTOML_FEATURES + [TARGET]
    train = df.loc[df["period"] == "train", cols]
    valid = df.loc[df["period"] == "validation", cols]
    hold = df.loc[df["period"] == "holdout", cols]

    model_dir = MODELS / "tech_value"
    shutil.rmtree(model_dir, ignore_errors=True)
    predictor = TabularPredictor(
        label=TARGET,
        problem_type="regression",
        eval_metric="mean_absolute_error",
        path=str(model_dir),
        verbosity=1,
    ).fit(
        train,
        # Explicit time-ordered validation data: no random folds, so no future leaks in.
        tuning_data=valid,
        time_limit=time_limit,
        presets=config.AUTOML_PRESET,
        # The training job is CPU-only and installs no deep-learning stack, so the neural
        # nets are left out by name instead of failing one by one.
        excluded_model_types=["NN_TORCH", "FASTAI"],
        ag_args_fit={"num_gpus": 0},
    )

    fit_times = predictor.leaderboard(silent=True).set_index("model")["fit_time"].to_dict()
    rows = []
    for name in predictor.model_names():
        pred = predictor.predict(valid, model=name).to_numpy()
        rows.append(
            score_row(
                f"AutoGluon: {name}",
                "validation",
                valid[TARGET].to_numpy(),
                pred,
                kind="automl",
                fit_time_s=round(float(fit_times.get(name, float("nan"))), 1),
            )
        )
    rows.sort(key=lambda r: r["mae_log"])
    best = rows[0]["model"].removeprefix("AutoGluon: ")

    # The single holdout touch, after the pick.
    holdout = score_row(
        f"AutoGluon: {best}",
        "holdout",
        hold[TARGET].to_numpy(),
        predictor.predict(hold, model=best).to_numpy(),
        kind="automl",
    )
    return {
        "trained_at": utc_now().isoformat(),
        "autogluon_version": version("autogluon.tabular"),
        "python_version": sys.version.split()[0],
        "preset": config.AUTOML_PRESET,
        "time_limit_s": time_limit,
        "n_spec_results": int(len(df)),
        "valid_start": config.VALID_START,
        "holdout_start": config.HOLDOUT_START,
        "leaderboard": rows,
        "holdout": holdout,
    }


def main() -> None:
    summary = run(pd.read_parquet(config.SPEC_TABLE))
    config.AUTOML_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    config.AUTOML_SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    pd.DataFrame(summary["leaderboard"]).to_parquet(config.AUTOML_LEADERBOARD, index=False)
    print(f"picked on validation: {summary['leaderboard'][0]['model']}")
    print(f"holdout: {summary['holdout']}")


if __name__ == "__main__":
    main()
