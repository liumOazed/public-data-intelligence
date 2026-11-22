import numpy as np
import pandas as pd
import pytest

from pdi.tech_value import cpu_specs, efficiency_ml, efficiency_signals, gwpg, process_nodes


def test_every_cpu_and_desktop_gpu_has_a_process_node():
    assert all(process_nodes.cpu_node(k) for k in cpu_specs.CPU_SPECS)
    for name, spec in gwpg.GPU_SPECS.items():
        if not spec[7]:
            assert process_nodes.gpu_node(name, spec[5]), name


def test_process_node_examples():
    assert process_nodes.cpu_node("amd ryzen 5 3600") == ("Zen 2", "TSMC N7", 7)
    assert process_nodes.cpu_node("amd ryzen 5 2400g")[0] == "Zen"  # Raven Ridge, not Zen+
    assert process_nodes.cpu_node("amd ryzen 7 5800x3d")[0] == "Zen 3"
    assert process_nodes.cpu_node("intel core i9-10920x")[0] == "Cascade Lake-X"
    assert process_nodes.cpu_node("intel core ultra 7 265k")[1] == "TSMC N3B"
    assert process_nodes.gpu_node("AMD Radeon RX 7600", "RDNA 3") == ("TSMC N6", 6)
    assert process_nodes.gpu_node("AMD Radeon RX 7900 XTX", "RDNA 3") == ("TSMC N5", 5)


def _chips(n=40, seed=0):
    rng = np.random.default_rng(seed)
    cores = rng.integers(4, 64, n).astype(float)
    year = rng.integers(2015, 2026, n)
    df = pd.DataFrame({
        "model_key": [f"chip {i}" for i in range(n)], "spec_group": [f"g{i}" for i in range(n)],
        "arch": rng.choice(["A", "B", "C"], n), "log_cores": np.log(cores),
        "vintage": (year - 2017).astype(float), "year": year,
    })  # fmt: skip
    df["log_ppw"] = 0.1 * df["vintage"] + 0.3 * df["log_cores"] + rng.normal(0, 0.02, n)
    return df


def test_held_out_predictions_never_see_the_chip():
    df = _chips()
    fit = efficiency_ml.held_out(df, "log_ppw", ["log_cores", "vintage"])
    # The linear model recovers the generating line, so held-out errors are near the noise.
    assert fit["resid_ols"].abs().median() < 0.05
    assert fit["resid_base"].abs().median() > fit["resid_ols"].abs().median()
    # Change one chip's own value: its held-out prediction must not move.
    moved = df.copy()
    moved.loc[0, "log_ppw"] += 5
    refit = efficiency_ml.held_out(moved, "log_ppw", ["log_cores", "vintage"])
    assert refit.loc[0, "pred_ols"] == pytest.approx(fit.loc[0, "pred_ols"])
    val = pd.DataFrame(efficiency_ml.validation(fit, "cpu", "x"))
    assert efficiency_ml.beats_baseline(val, "cpu", "x")


def test_tiers_find_two_bands_and_one_for_a_continuum():
    rng = np.random.default_rng(1)
    two = np.concatenate([rng.normal(np.log(30), 0.05, 30), rng.normal(np.log(8), 0.05, 30)])
    one = np.log(rng.uniform(2, 8, 60))

    def frame(v):
        return pd.DataFrame({"kind": "gpu", "model_key": [f"c{i}" for i in range(len(v))],
                             "name": "x", "vendor": "nvidia", "year": 2025, "watts": 100.0,
                             "price_usd": None, "log_ppw": v, "score_per_watt": np.exp(v)})  # fmt: skip

    chips, summ = efficiency_ml.tiers(frame(two))
    assert summ["n_tiers"].iloc[0] == 2
    top = chips[chips["tier"] == 1]
    assert top["score_per_watt"].min() > 20  # tier 1 is the efficient band
    _, summ = efficiency_ml.tiers(frame(one))
    assert summ["n_tiers"].iloc[0] == 1


def test_turning_point_finds_a_bend_and_not_a_line():
    x = np.arange(2010, 2026)
    bent = np.where(x <= 2018, 0.3 * (x - 2010), 0.3 * 8 + 0.05 * (x - 2018))
    tp = efficiency_signals.turning_point(
        x, bent + np.random.default_rng(0).normal(0, 0.01, len(x))
    )
    assert tp["strength"] == "strong"
    assert tp["year"] in (2017, 2018, 2019)
    assert tp["before"] > tp["after"]
    line = efficiency_signals.turning_point(
        x, 0.1 * x + np.random.default_rng(0).normal(0, 0.05, len(x))
    )
    assert line["strength"] == "none"


def test_pace_compares_recent_with_before():
    years = np.arange(2014, 2026)
    best = np.exp(np.where(years <= 2021, 0.3 * (years - 2014), 2.1 + 0.05 * (years - 2021)))
    p = efficiency_signals.pace(years, best, 4)
    assert p["prior"] == pytest.approx(np.expm1(0.3) * 100, rel=0.01)
    assert p["recent"] == pytest.approx(np.expm1(0.05) * 100, rel=0.01)
    assert efficiency_signals._pace_word(p["recent"], p["prior"]) == "slowing"


def test_product_classes_compare_like_with_like():
    pc = efficiency_signals.product_class
    assert pc("cpu", "amd ryzen 7 7700x") == "Ryzen 7 X"
    assert pc("cpu", "amd ryzen 7 5800x3d") == "Ryzen 7 X3D"
    assert pc("cpu", "intel core i7-13700k") == pc("cpu", "intel core ultra 7 265k") == "Core 7 K"
    assert pc("cpu", "intel core i5-12400f") == "Core 5"
    assert pc("gpu", "nvidia rtx 4070 ti") == "GeForce xx70 Ti"
    assert pc("gpu", "nvidia gtx 980") == pc("gpu", "nvidia rtx 3080") == "GeForce xx80"
    assert pc("gpu", "amd rx 7800 xt") == "Radeon x800 XT"


def test_signals_build_on_a_small_frame():
    rows = []
    for year in range(2014, 2026):
        for i, (vendor, key) in enumerate([("nvidia", f"nvidia rtx {year}60"), ("nvidia", f"nvidia rtx {year}70"),
                                            ("amd", f"amd rx {year % 10}600"), ("amd", f"amd rx {year % 10}700")]):  # fmt: skip
            ppw = np.exp(0.2 * (year - 2014)) * (1.5 if vendor == "nvidia" else 1.0) * (1 + 0.1 * i)
            rows.append({"kind": "gpu", "model_key": key, "vendor": vendor, "year": year,
                         "watts": 150.0 * 1.03 ** (year - 2014), "score_per_watt": ppw,
                         "log_ppw": np.log(ppw), "price_usd": 300.0 + year, "arch": f"A{year}",
                         "node": "N"})  # fmt: skip
    sig, series = efficiency_signals.build(pd.DataFrame(rows))
    by = sig.set_index("signal")
    assert by.loc["frontier_pace", "direction"] == "steady"
    assert by.loc["vendor_gap", "direction"] == "no clear change"
    assert set(sig["label"]) <= {"explanation", "observation"}
    assert not series.empty
