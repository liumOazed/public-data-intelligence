import numpy as np
import pandas as pd

from pdi.tech_value import model


def test_periods_follow_time_and_never_split_a_processor(spec_signal):
    df = model.prepare(spec_signal)
    assert set(df["period"]) == {"train", "validation", "holdout"}
    assert (df.groupby("processor")["period"].nunique() == 1).all()
    last = df.groupby("period")["first_avail"].max()
    first = df.groupby("period")["first_avail"].min()
    assert last["train"] < first["validation"] <= last["validation"] < first["holdout"]


def test_baseline_is_always_on_the_leaderboard(spec_signal):
    ev = model.evaluate(model.prepare(spec_signal))
    assert model.BASELINE_NAME in set(ev.leaderboard["model"])
    assert model.BASELINE_NAME in set(ev.holdout["model"])
    assert list(ev.leaderboard["rank"]) == [1, 2]


def test_signal_is_found_when_it_exists(spec_signal):
    ev = model.evaluate(model.prepare(spec_signal))
    assert ev.verdict["beats_baseline"]
    assert ev.verdict["published_model"] == model.PLAIN_NAME
    assert ev.verdict["holdout_improvement"] > ev.fold_spread > 0


def test_no_signal_publishes_the_baseline(spec_noise):
    ev = model.evaluate(model.prepare(spec_noise))
    assert not ev.verdict["beats_baseline"]
    assert ev.verdict["published_model"] == model.BASELINE_NAME
    assert ev.verdict["statement"].startswith("No signal found")
    assert ev.holdout.set_index("model").loc[model.BASELINE_NAME, "published"]


def test_automl_rows_join_the_same_leaderboard(spec_signal):
    df = model.prepare(spec_signal)
    automl = {
        "leaderboard": [
            {
                "model": "AutoGluon: Fake",
                "period": "validation",
                "n_results": 1,
                "mae_log": 0.0001,
                "median_abs_pct_error": 0.01,
                "kind": "automl",
            }
        ],
        "holdout": {
            "model": "AutoGluon: Fake",
            "period": "holdout",
            "n_results": 1,
            "mae_log": 0.0001,
            "median_abs_pct_error": 0.01,
            "kind": "automl",
        },
    }
    ev = model.evaluate(df, automl)
    assert ev.leaderboard.iloc[0]["model"] == "AutoGluon: Fake"
    assert model.BASELINE_NAME in set(ev.leaderboard["model"])
    assert ev.verdict["picked_on_validation"] == "AutoGluon: Fake"


def test_residuals_are_out_of_fold_and_ranked(spec_signal):
    df = model.prepare(spec_signal)
    oof = model.out_of_fold_residuals(df)
    assert oof["predicted_log"].notna().all()
    ranking = model.residual_ranking(oof)
    assert ranking["residual_pct"].is_monotonic_decreasing
    assert (ranking["n_results"] >= 3).all()
    assert {"config_memory", "config_compiler", "config_os"} <= set(ranking.columns)
    # Residuals of a well-specified model centre on zero.
    assert abs(np.median(ranking["residual_pct"])) < 2


def test_coefficients_recover_the_generating_process(spec_signal):
    coef = model.Plain.fit(model.prepare(spec_signal)).coefficients().set_index("term")
    assert abs(coef.loc["log_cores", "coefficient"] - 0.9) < 0.05
    assert abs(coef.loc["vintage_years", "coefficient"] - 0.06) < 0.01
    assert isinstance(coef, pd.DataFrame)
