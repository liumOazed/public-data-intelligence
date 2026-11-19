import numpy as np
import pandas as pd

from pdi.tech_value import gwpg, model


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


def test_gwpg_gpu_canonicalization_and_ranking():
    assert gwpg.canon_gpu("1x NVIDIA RTX PRO 6000 Blackwell Workstation Edition") == (
        "NVIDIA RTX PRO 6000 Blackwell"
    )
    assert gwpg.canon_gpu("1x NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU") == (
        "NVIDIA RTX PRO 5000 Blackwell Laptop"
    )
    assert gwpg.canon_gpu("NVIDIA RTX 4000 SFF Ada Generation") == "NVIDIA RTX 4000 Ada"
    assert gwpg.canon_gpu("AMD FirePro WX5100") == "AMD Radeon Pro WX 5100"

    sample_html = """
    <table>
      <tr><th>Submitter</th><th>System</th><th>GPU</th><th>Resolution</th><th>maya-06</th><th>snx-04</th></tr>
      <tr><td>Dell</td><td>Precision 1</td><td>NVIDIA RTX 6000 Ada</td><td>1920x1080</td><td>180.0</td><td>220.0</td></tr>
      <tr><td>HP</td><td>Z6 G5</td><td>NVIDIA RTX 6000 Ada</td><td>1920x1080</td><td>175.0</td><td>210.0</td></tr>
      <tr><td>Dell</td><td>Precision 2</td><td>NVIDIA RTX 5000 Ada</td><td>1920x1080</td><td>140.0</td><td>170.0</td></tr>
      <tr><td>Lenovo</td><td>P3 Tower</td><td>NVIDIA RTX 5000 Ada</td><td>1920x1080</td><td>145.0</td><td>175.0</td></tr>
      <tr><td>HP</td><td>Z2 Tower</td><td>NVIDIA RTX 4000 Ada</td><td>1920x1080</td><td>90.0</td><td>110.0</td></tr>
      <tr><td>Dell</td><td>Precision 3</td><td>NVIDIA RTX 4000 Ada</td><td>1920x1080</td><td>88.0</td><td>105.0</td></tr>
      <tr><td>MIFCOM</td><td>Workstation 1</td><td>AMD Radeon AI PRO R9700</td><td>1920x1080</td><td>135.0</td><td>160.0</td></tr>
      <tr><td>MIFCOM</td><td>Workstation 2</td><td>AMD Radeon AI PRO R9700</td><td>1920x1080</td><td>138.0</td><td>164.0</td></tr>
      <tr><td>HP</td><td>Z4 G4</td><td>NVIDIA RTX A6000</td><td>1920x1080</td><td>130.0</td><td>155.0</td></tr>
      <tr><td>Dell</td><td>Precision 4</td><td>NVIDIA RTX A6000</td><td>1920x1080</td><td>132.0</td><td>158.0</td></tr>
      <tr><td>HP</td><td>Z2 G9</td><td>NVIDIA RTX A4000</td><td>1920x1080</td><td>85.0</td><td>98.0</td></tr>
      <tr><td>Lenovo</td><td>P520</td><td>NVIDIA RTX A4000</td><td>1920x1080</td><td>83.0</td><td>96.0</td></tr>
    </table>
    """
    df = gwpg.parse(
        {
            "SPECviewperf 2020 v3": {
                "url": "https://www.spec.org/gwpg/gpc.data/viewperf2020v3/summary.html",
                "html": sample_html,
                "retrieved_at": "2026-04-19T10:00:00Z",
            }
        }
    )
    assert len(df) == 12
    oof = gwpg.out_of_fold_gpu_residuals(df)
    ranking = gwpg.gpu_residual_ranking(oof, min_results=2)
    assert len(ranking) == 6
    assert {"NVIDIA", "AMD"} == set(ranking["vendor"])
    assert {
        "residual_pct",
        "structural_residual_pct",
        "gbdt_residual_pct",
        "consensus_verdict",
        "tuning_spread_pct",
        "peak_gain_pct",
        "bandwidth_gbps",
        "vram_gb",
    } <= set(ranking.columns)
