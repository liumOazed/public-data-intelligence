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
    assert {"pred_ols", "pred_structural", "pred_nonlinear", "n_configs", "n_systems"} <= set(
        ranking.columns
    )
    assert (ranking["pred_ols_lo"].dropna() <= ranking["pred_ols_hi"].dropna()).all()
    val = model.residual_validation(ranking)
    overall = val[(val["slice"] == "All")].set_index("model")
    # On data generated from the plain model, the plain model beats the cores-x-clock baseline.
    assert (
        overall.loc["OLS", "mean_abs_log_error"]
        < overall.loc["Baseline (cores x clock)", "mean_abs_log_error"]
    )
    # Residuals of a well-specified model centre on zero.
    assert abs(np.median(ranking["residual_pct"])) < 2


def test_consensus_verdict_and_compiler_classification():
    assert model.consensus_verdict(46.5, 50.9, 73.7, cores_per_chip=32) == model.ABOVE
    assert model.consensus_verdict(40.8, 23.1, 4.0, cores_per_chip=4) == model.SENSITIVE_SMALL_CORE
    assert model.consensus_verdict(48.8, 20.1, -6.2, cores_per_chip=12) == model.SENSITIVE_REGIME
    assert model.consensus_verdict(-21.4, -21.2, -8.1, cores_per_chip=72) == model.BELOW
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
        "workload_specialization_pct",
        "bandwidth_gbps",
        "vram_gb",
        "pred_ols",
        "pred_nonlinear_lo",
        "model_spread_pts",
        "outside_training_range",
    } <= set(ranking.columns)
    # Predicted x exp(residual) gives back the measured score.
    back = ranking["pred_ols"] * np.exp(ranking["ols_resid_log"])
    assert np.allclose(back, ranking["measured_score"])
    val = gwpg.gpu_residual_validation(ranking)
    assert {"Baseline (shaders x clock)", "OLS"} <= set(val["model"])

    # A second suite on a different scale: the same cards, scores doubled. Calibration puts
    # them back on one scale, so the held-out check passes.
    doubled = sample_html
    for v in (
        "175.0",
        "140.0",
        "145.0",
        "90.0",
        "88.0",
        "135.0",
        "138.0",
        "130.0",
        "132.0",
        "85.0",
        "83.0",
    ):
        doubled = doubled.replace(f"<td>{v}</td>", f"<td>{float(v) * 2:.1f}</td>", 1)
    both = gwpg.parse(
        {
            "SPECviewperf 2020 v3": {"url": "u", "html": sample_html},
            "SPECviewperf 15": {"url": "u", "html": doubled},
        }
    )
    check = gwpg.calibration_validation(both).set_index("suite")
    assert check.loc["SPECviewperf 2020 v3", "status"] == "Reference scale"
    assert check.loc["SPECviewperf 15", "n_shared_cards"] == 6
    assert check.loc["SPECviewperf 15", "median_abs_gap_pct"] < 15
    assert check.loc["SPECviewperf 15", "status"] == "Validated"
    assert gwpg.in_cohorts(both, gwpg.validated_cohorts(check.reset_index())).shape[0] == 24
    work = gwpg.workload_table(both)
    assert set(work["suite"]) == {"SPECviewperf 2020 v3", "SPECviewperf 15"}
    assert (work.groupby("suite").size() == 6).all()


def test_residual_evidence_helpers():
    from pdi.tech_value import residual_evidence as rev

    assert rev.related_group("AMD EPYC 9454P") == "AMD EPYC 9454"
    assert rev.related_group("Intel Xeon Gold 6430") == "Intel Xeon Gold 6430"
    lo, hi = rev.loo_interval(np.linspace(-0.2, 0.2, 21))
    assert (lo < hi).all()
    chips = pd.DataFrame({"cores": [4, 64, 512]}, index=["a", "b", "c"])
    results = pd.DataFrame({"processor": ["a", "b", "c"], "cores": [4, 64, 512]})
    flags = rev.extrapolation_flags(chips, results, {"cores": "Cores"})
    assert flags["c"] == "Cores above training range"
    assert flags["a"] == "Cores below training range"
    assert flags["b"] == ""


def test_catalogue_status_and_reasons():
    from pdi.tech_value import catalogue

    value = pd.DataFrame(
        [
            {
                "kind": "gpu",
                "model_key": "nvidia rtx 4060",
                "price_usd": 300.0,
                "score_per_dollar": 10.0,
            },
            {
                "kind": "cpu",
                "model_key": "amd ryzen 9 7950x",
                "price_usd": 500.0,
                "score_per_dollar": 1.1,
            },
        ]
    )
    fair = value[["kind", "model_key"]].assign(fair_price_usd=[350.0, 480.0], gap_pct=[-14.3, 4.2])
    bench = pd.DataFrame([{"kind": "gpu", "model_key": "nvidia rtx 4060", "median_score": 3000.0}])
    spec = pd.DataFrame(
        {"processor": ["AMD Ryzen 9 7950X"] * 2, "hardware_vendor": ["ASUS", "ASUS"]}
    )
    gw = pd.DataFrame(
        [
            {"processor": "NVIDIA RTX A2000", "oem": "HP", "spec_status": "ok"},
            {"processor": "NVIDIA Quadro X9", "oem": "HP", "spec_status": "missing"},
        ]
    )
    rank = pd.DataFrame(
        [{"processor": "NVIDIA RTX A2000", "residual_pct": 3.0, "consensus_verdict": "x"}]
    )
    cat = catalogue.build(
        value=value,
        fair=fair,
        benchmarks=bench,
        cpu_rank=pd.DataFrame(columns=rank.columns),
        gpu_rank=rank,
        spec=spec,
        gwpg=gw,
        dates={},
    ).set_index("model_key")
    assert cat.loc["nvidia rtx 4060", "coverage_status"] == catalogue.NOT_FOUND
    assert cat.loc["nvidia rtx 4060", "reason"].startswith("Unsupported source")
    assert cat.loc["amd ryzen 9 7950x", "coverage_status"] == catalogue.LIMITED
    assert cat.loc["nvidia rtx a2000", "coverage_status"] == catalogue.LIMITED  # one OEM
    assert cat.loc["nvidia quadro x9", "coverage_status"] == catalogue.MISSING_SPECS
    assert bool(cat.loc["nvidia rtx 4060", "on_value_ranking"])
    # The cross-page panel and comparison read these from the same row.
    assert cat.loc["nvidia rtx 4060", "value_rank"] == 1
    assert cat.loc["nvidia rtx 4060", "fair_price_usd"] == 350.0
    assert cat.loc["nvidia rtx 4060", "score_per_dollar"] == 10.0


def test_gwpg_recognises_consumer_cards():
    assert gwpg.canon_gpu("NVIDIA GeForce RTX 4060") in gwpg.GPU_SPECS
    assert gwpg.canon_gpu("GeForce RTX 4070 Ti Super") == "NVIDIA GeForce RTX 4070 Ti SUPER"
    assert gwpg.canon_gpu("Radeon RX 7900 XTX") == "AMD Radeon RX 7900 XTX"
    assert gwpg.gpu_vendor(gwpg.canon_gpu("Arc A770")) == "Intel"


def test_catalogue_tracks_retail_products_and_shows_sub_threshold_results():
    from pdi.tech_value import catalogue

    value = pd.DataFrame(
        [
            {"kind": "cpu", "model_key": "amd ryzen 9 7950x", "price_usd": 500.0},
            {"kind": "cpu", "model_key": "intel core i9-14900k", "price_usd": 450.0},
            {"kind": "gpu", "model_key": "nvidia rtx a400", "price_usd": 150.0},
            {"kind": "gpu", "model_key": "nvidia rtx 4500 ada", "price_usd": 2500.0},
        ]
    )
    spec = pd.DataFrame(
        {
            "processor": ["AMD Ryzen 9 7950X"] * 2 + ["AMD EPYC 9654"] * 4,
            "hardware_vendor": ["ASUS", "Supermicro", "Dell", "HPE", "Lenovo", "Dell"],
            "base_score": [175.0, 177.0, 1500.0, 1520.0, 1490.0, 1510.0],
            "chips": [1, 1, 2, 2, 2, 2],
        }
    )
    gw = pd.DataFrame(
        [
            # Four results, but only one in a suite whose calibration passed.
            {
                "processor": "NVIDIA RTX 4500 Ada",
                "oem": o,
                "spec_status": "ok",
                "suite": s,
                "os_family": "Windows",
                "res_tier": "4K",
                "composite_score": v,
                "in_ranked_suites": r,
            }
            for o, s, v, r in [
                ("HP", "SPECviewperf 2020 v3", 100.0, True),
                ("Dell", "SPECviewperf 13", 80.0, False),
                ("Dell", "SPECviewperf 13", 90.0, False),
                ("HP", "SPECviewperf 13", 70.0, False),
            ]
        ]
    )
    cpu_rank = pd.DataFrame(
        [
            {
                "processor": "AMD EPYC 9654",
                "residual_pct": 1.0,
                "consensus_verdict": "Near prediction",
                "measured_per_socket": 750.0,
            }
        ]
    )
    cat = catalogue.build(
        value=value,
        fair=value[["kind", "model_key"]].iloc[0:0],
        benchmarks=pd.DataFrame(columns=["kind", "model_key", "median_score"]),
        cpu_rank=cpu_rank,
        gpu_rank=pd.DataFrame(columns=["processor", "residual_pct", "consensus_verdict"]),
        spec=spec,
        gwpg=gw,
        dates={},
    ).set_index("model_key")

    # Every priced product is tracked; SPEC-only hardware stays in the catalogue untracked.
    assert cat["tracked"].sum() == 4
    assert not cat.loc["amd epyc 9654", "tracked"]
    # One disclosure below the threshold is still a visible measurement, kept apart from
    # whether the models assess the chip.
    r = cat.loc["amd ryzen 9 7950x"]
    assert r["measured_value"] == 176.0 and r["measured_n"] == 2
    assert r["assessment"] == catalogue.MEASURED_ONLY
    assert r["reason"] == "Insufficient disclosures (2 of 3 needed)"
    assert cat.loc["amd epyc 9654", "assessment"] == catalogue.ASSESSED
    assert cat.loc["intel core i9-14900k", "assessment"] == catalogue.NO_MEASUREMENT
    assert "SPEC CPU 2017" in cat.loc["intel core i9-14900k", "reason"]
    assert cat.loc["intel core i9-14900k", "next_source"].startswith("OpenBenchmarking")
    # The RTX A400 is a workstation card, not a consumer one.
    assert not cat.loc["nvidia rtx a400", "reason"].startswith("Unsupported source")
    # The usable count, not the total, is compared with the minimum.
    g = cat.loc["nvidia rtx 4500 ada"]
    assert "(1 of 2 needed; 3 more in other suites)" in g["reason"]
    assert g["measured_value"] == 80.0 and "SPECviewperf 13" in g["measured_metric"]
    assert g["measured_n"] == 4 and g["measured_groups"] == 2

    summary = catalogue.tracked_summary(cat.reset_index())
    assert summary["cpu"] == {
        "tracked": 2,
        "assessed": 0,
        "measured_only": 1,
        "no_measurement": 1,
        "catalogue": 3,
    }


def test_coverage_history_keeps_one_count_per_day_and_first_seen():
    from pdi.tech_value import catalogue

    cat = pd.DataFrame(
        {
            "kind": ["gpu", "gpu"],
            "model_key": ["a", "b"],
            "tracked": [True, False],
            "assessment": [catalogue.ASSESSED, catalogue.ASSESSED],
        }
    )
    hist, seen = catalogue.update_history(cat, [], {}, "2026-11-20")
    assert seen == {"gpu|a": "2026-11-20"}
    cat.loc[1, "tracked"] = True
    hist, seen = catalogue.update_history(cat, hist, seen, "2026-11-20")
    hist, seen = catalogue.update_history(cat, hist, seen, "2026-11-27")
    gpu = [h for h in hist if h["kind"] == "gpu"]
    assert [(h["date"], h["tracked"], h["assessed"]) for h in gpu] == [
        ("2026-11-20", 2, 2),
        ("2026-11-27", 2, 2),
    ]
    # A product keeps the day it was first tracked.
    assert seen == {"gpu|a": "2026-11-20", "gpu|b": "2026-11-20"}
