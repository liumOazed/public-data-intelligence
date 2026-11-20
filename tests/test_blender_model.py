import numpy as np
import pandas as pd

from pdi.tech_value import blender_model, catalogue, cpu_specs, gwpg, names


def _bench(keys, scale=1.0, noise=0.0, seed=0):
    """Synthetic Blender CPU results: score follows cores, SMT, clock and power, plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in keys:
        p, e, threads, base, boost, l3, tdp, year = cpu_specs.CPU_SPECS[k]
        score = scale * 1e-4 * (p + 0.5 * e) * boost * (1 + 0.25 * (threads > p + e))
        score *= (tdp / 65) ** 0.3 * (1 + 0.04 * (year - 2017))
        score *= float(np.exp(rng.normal(0, noise)))
        rows.append(
            {
                "kind": "cpu",
                "model_key": k,
                "device_name": k,
                "device_type": "CPU",
                "blender_version": "4.5",
                "n_runs": 10,
                "median_score": score,
                "p25_score": score * 0.95,
                "p75_score": score * 1.05,
            }
        )
    return pd.DataFrame(rows)


def test_every_cpu_spec_is_consistent():
    for key, (p, e, threads, base, boost, l3, tdp, year) in cpu_specs.CPU_SPECS.items():
        assert names.model_key(key) == ("cpu", key), key
        assert p >= 1 and e >= 0, key
        assert p + e <= threads <= 2 * (p + e), key
        assert base <= boost, key
        assert 2 <= l3 <= 384 and 35 <= tdp <= 350 and 2010 <= year <= 2026, key


def test_gpu_specs_cover_every_card_key_once():
    keys = blender_model._gpu_spec_by_key()
    assert keys["nvidia gtx 1080 ti"][0] == "NVIDIA GeForce GTX 1080 Ti"
    assert keys["nvidia rtx 4090"][1][0] == 16384
    # Laptop parts never become desktop model keys.
    assert all(spec[7] == 0 for _, spec in keys.values())


def test_identical_specs_are_held_out_together():
    df = blender_model.cpu_frame(_bench(["intel core i5-12400", "intel core i5-12400f"]))
    assert df["spec_group"].nunique() == 1


def test_ranking_flags_an_outlier_and_beats_the_baseline():
    keys = sorted(cpu_specs.CPU_SPECS)[:80]
    bench = _bench(keys, noise=0.03)
    target = "amd ryzen 9 5950x"
    bench.loc[bench["model_key"] == target, "median_score"] *= 1.5
    rank = blender_model.ranking(blender_model.cpu_frame(bench), "cpu", "4.5")
    r = rank.set_index("model_key").loc[target]
    assert r["residual_pct"] > 25
    assert r["consensus_verdict"] == "Above prediction across models"
    val = blender_model.validation(rank, "cpu")
    a = val[val["slice"] == "All"].set_index("model")["median_abs_pct_error"]
    assert a["OLS"] < a["Baseline (cores x clock)"]


def test_too_few_products_are_not_fitted():
    bench = _bench(sorted(cpu_specs.CPU_SPECS)[: blender_model.MIN_PRODUCTS - 1])
    assert blender_model.ranking(blender_model.cpu_frame(bench), "cpu", "4.5").empty
    ok, why = blender_model.passes(pd.DataFrame(), pd.DataFrame())
    assert not ok and "Fewer than" in why


def test_version_check_uses_the_gwpg_thresholds():
    keys = sorted(cpu_specs.CPU_SPECS)[:60]
    same = pd.concat(
        [
            _bench(keys, noise=0.05, seed=1),
            _bench(keys, scale=0.8, noise=0.05, seed=1).assign(blender_version="4.4"),
        ]
    )
    check = blender_model.version_check(same, "cpu", "4.5")
    assert list(check["status"]) == ["Validated"]
    assert check["median_abs_gap_pct"].iloc[0] <= gwpg.CALIBRATION_MAX_GAP_PCT
    # Residuals that change with the version fail the check.
    other = pd.concat(
        [
            _bench(keys, noise=0.05, seed=1),
            _bench(keys, noise=0.4, seed=2).assign(blender_version="4.4"),
        ]
    )
    check = blender_model.version_check(other, "cpu", "4.5")
    assert check["status"].iloc[0].startswith("Not validated")


def test_catalogue_counts_blender_assessments_separately():
    value = pd.DataFrame(
        [
            {"kind": "cpu", "model_key": "amd ryzen 7 9700x", "price_usd": 300.0},
            {"kind": "cpu", "model_key": "intel core i5-12400", "price_usd": 150.0},
        ]
    )
    benchmarks = pd.DataFrame(
        [
            {"kind": "cpu", "model_key": "amd ryzen 7 9700x", "median_score": 300.0},
            {"kind": "cpu", "model_key": "intel core i5-12400", "median_score": 150.0},
        ]
    )
    blender_rank = pd.DataFrame(
        [
            {
                "kind": "cpu",
                "model_key": "amd ryzen 7 9700x",
                "residual_pct": 12.0,
                "pred_ols": 268.0,
                "consensus_verdict": "Near prediction",
            }
        ]
    )
    empty = pd.DataFrame(columns=["processor", "residual_pct", "consensus_verdict"])
    cat = catalogue.build(
        value=value,
        fair=value[["kind", "model_key"]].iloc[0:0],
        benchmarks=benchmarks,
        cpu_rank=empty,
        gpu_rank=empty,
        spec=pd.DataFrame(),
        gwpg=pd.DataFrame(),
        dates={},
        blender_rank=blender_rank,
        blender_spec_keys={"cpu": {"amd ryzen 7 9700x"}},
    ).set_index("model_key")
    r = cat.loc["amd ryzen 7 9700x"]
    assert r["assessment"] == catalogue.ASSESSED
    assert r["assessed_by"] == "Blender Open Data"
    assert r["next_source"] == ""
    q = cat.loc["intel core i5-12400"]
    # A Blender score without specifications is a measurement, not an assessment.
    assert q["assessment"] == catalogue.MEASURED_ONLY
    assert "no specifications" in q["blender_reason"]
    s = catalogue.tracked_summary(cat.reset_index())["cpu"]
    assert s["assessed"] == 1 and s["assessed_blender"] == 1 and s["assessed_spec"] == 0
