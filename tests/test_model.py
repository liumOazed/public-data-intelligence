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
    assert oof["structural_pred_log"].notna().all()
    assert oof["gbdt_pred_log"].notna().all()
    ranking = model.residual_ranking(oof)
    assert ranking["residual_pct"].is_monotonic_decreasing
    assert (ranking["n_results"] >= 3).all()
    expected_cols = {
        "config_memory",
        "config_compiler",
        "config_os",
        "structural_residual_pct",
        "gbdt_residual_pct",
        "consensus_verdict",
        "tuning_spread_pct",
        "peak_gain_pct",
        "compiler_family",
        "l3_mb_per_core",
        "score_per_core",
        "score_per_core_ghz",
    }
    assert expected_cols <= set(ranking.columns)
    assert (ranking["tuning_spread_pct"] >= 0).all()
    # Residuals of a well-specified model centre on zero.
    assert abs(np.median(ranking["residual_pct"])) < 2


def test_consensus_verdict_and_compiler_classification():
    assert (
        model.consensus_verdict(46.5, 50.9, 73.7, cores_per_chip=32)
        == "True architectural standout"
    )
    assert (
        model.consensus_verdict(40.8, 23.1, 4.0, cores_per_chip=4)
        == "Small-core linearity artifact"
    )
    assert (
        model.consensus_verdict(48.8, 20.1, -6.2, cores_per_chip=12)
        == "Linearity / regime artifact"
    )
    assert (
        model.consensus_verdict(-21.4, -21.2, -8.1, cores_per_chip=72)
        == "Memory / scaling bottleneck"
    )
    assert model.classify_compiler("C/C++/Fortran: Version 2024.1 of Intel oneAPI") == (
        "Intel oneAPI (LLVM)"
    )
    assert model.classify_compiler("Version 19.1.1.217 of Intel C/C++") == "Intel Classic ICC"
    assert model.classify_compiler("AMD Optimizing C/C++ Compiler 4.2 (AOCC)") == "AMD AOCC"
    assert model.classify_compiler("GNU GCC 12.2") == "GNU GCC"


def test_coefficients_recover_the_generating_process(spec_signal):
    coef = model.Plain.fit(model.prepare(spec_signal)).coefficients().set_index("term")
    assert abs(coef.loc["log_cores", "coefficient"] - 0.9) < 0.05
    assert abs(coef.loc["vintage_years", "coefficient"] - 0.06) < 0.01
    assert isinstance(coef, pd.DataFrame)
