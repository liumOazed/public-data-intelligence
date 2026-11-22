import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdi.tech_value import efficiency as ef
from pdi.tech_value import electricity
from pdi.tech_value.gpu_specs import GPU_SPECS, NOT_ENTERED

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "site" / "tech-value" / "efficiency.js"


def _bench(rows):
    df = pd.DataFrame(rows, columns=["kind", "model_key", "median_score"])
    return df.assign(p25_score=df["median_score"], p75_score=df["median_score"], n_runs=50,
                     device_type="OPTIX")  # fmt: skip


def test_metrics_match_the_worked_example():
    # 6,000 samples per minute: a 10,000-sample job takes 100 s; at 200 W that is 20 kJ.
    assert ef.seconds_per_job(6000) == pytest.approx(100.0)
    assert ef.energy_per_job(6000, 200) == pytest.approx(20_000.0)
    assert ef.edp(6000, 200, 1.0) == pytest.approx(2_000_000.0)
    assert ef.edp(6000, 200, 2.0) == pytest.approx(200_000_000.0)


def test_edp_stops_a_slow_low_power_chip_winning_on_energy_alone():
    fast = (6000, 300)  # 20 points per watt
    slow = (1000, 40)  # 25 points per watt: wins on energy per job
    assert ef.energy_per_job(*slow) < ef.energy_per_job(*fast)
    assert ef.edp(*fast, 1.0) < ef.edp(*slow, 1.0)


def test_tco_charges_only_the_hours_under_load():
    total, power = ef.tco(500, 200, 0.17, hours_per_day=4, load_share=0.3, years=3)
    kwh = 200 * 4 * 0.3 * 365 * 3 / 1000  # 262.8 kWh
    assert power == pytest.approx(kwh * 0.17)
    assert total == pytest.approx(500 + kwh * 0.17)


def test_frontier_marks_only_dominated_models():
    df = pd.DataFrame(
        {"kind": "gpu", "score": [1000, 900, 2000, 1500], "watts": [100, 120, 200, 250]}
    )
    assert ef.frontier(df).tolist() == [True, False, True, False]


def test_build_joins_rated_power_and_explains_missing_models():
    bench = _bench([
        ("gpu", "nvidia rtx 4060", 3200.0),
        ("gpu", "nvidia rtx 2050", 900.0),
        ("gpu", "nvidia made up 9999", 100.0),
        ("cpu", "amd ryzen 5 3600", 150.0),
    ])  # fmt: skip
    value = pd.DataFrame({"kind": ["gpu"], "model_key": ["nvidia rtx 4060"], "price_usd": [299.0]})
    ranked, missing = ef.build(bench, value)
    assert set(ranked["model_key"]) == {"nvidia rtx 4060", "amd ryzen 5 3600"}
    assert (ranked["power_kind"] == "rated").all()
    gpu = ranked[ranked["kind"] == "gpu"].iloc[0]
    assert gpu["watts"] == GPU_SPECS["nvidia rtx 4060"][0]
    assert gpu["score_per_watt"] == pytest.approx(3200 / 115)
    assert gpu["price_usd"] == 299.0
    reasons = dict(zip(missing["model_key"], missing["reason"], strict=True))
    assert reasons["nvidia rtx 2050"].startswith("Laptop only")
    assert reasons["nvidia made up 9999"] == "No rated power figure entered yet"


def test_summary_falls_back_to_a_labelled_rate_without_the_series():
    s = ef.summary(pd.DataFrame(), None)
    assert s["electricity"]["fallback"] is True
    assert s["tco_defaults"] == {"hours_per_day": 4.0, "load_share": 0.30, "years": 3}


def test_gpu_table_is_plausible_and_disjoint_from_the_exclusions():
    for key, (watts, year) in GPU_SPECS.items():
        assert 30 <= watts <= 600, key
        assert 2013 <= year <= 2026, key
        assert key == key.lower()
    assert not set(GPU_SPECS) & set(NOT_ENTERED)


def test_electricity_csv_parser_skips_missing_months():
    text = "observation_date,APU000072610\n2026-08-01,0.181\n2026-09-01,.\n2026-10-01,0.179\n"
    df = electricity.parse_csv(text)
    assert df["usd_per_kwh"].tolist() == [0.181, 0.179]
    df["retrieved_at"] = "2026-11-22T00:00:00+00:00"
    assert electricity.latest_rate(df)["period"] == "2026-10"


def test_homepage_and_navbar_link_every_tech_value_page():
    home = (ROOT / "site" / "index.qmd").read_text()
    nav = (ROOT / "site" / "_quarto.yml").read_text()
    for page in ("index", "fair-price", "residuals", "efficiency"):
        assert f'href="tech-value/{page}.html"' in home
        assert f"tech-value/{page}.qmd" in nav
    for page in ("index", "fair-price", "residuals", "efficiency"):
        text = (ROOT / "site" / "tech-value" / f"{page}.qmd").read_text()
        assert 'href="efficiency.html"' in text, page


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_browser_calculations_match_python():
    rows = [
        {"score": 3200.0, "watts": 115.0},
        {"score": 9100.0, "watts": 360.0},
        {"score": 3000.0, "watts": 160.0},
        {"score": 14900.0, "watts": 575.0},
    ]
    script = f"""
const c = require({json.dumps(str(JS))});
const rows = {json.dumps(rows)};
console.log(JSON.stringify({{
  edp1: rows.map((r) => c.edp(r.score, r.watts, 10000, 1)),
  edp2: rows.map((r) => c.edp(r.score, r.watts, 10000, 1.6)),
  frontier: c.frontier(rows),
  tco: c.tco(500, 200, 0.17, 4, 0.3, 3),
}}));
"""
    out = json.loads(
        subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    )
    df = pd.DataFrame(rows).assign(kind="gpu")
    np.testing.assert_allclose(out["edp1"], ef.edp(df["score"], df["watts"], 1.0))
    np.testing.assert_allclose(out["edp2"], ef.edp(df["score"], df["watts"], 1.6))
    assert out["frontier"] == ef.frontier(df).tolist()
    total, power = ef.tco(500, 200, 0.17, 4, 0.3, 3)
    assert out["tco"]["total"] == pytest.approx(float(total))
    assert out["tco"]["power"] == pytest.approx(float(power))
