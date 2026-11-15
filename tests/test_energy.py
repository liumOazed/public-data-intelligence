"""Energy Demand: the parser and the arithmetic, on a made-up file in the source's layout.

The live slice (tests/fixtures/live/ember_monthly.csv) is tested in test_live_samples.py.
"""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from pdi.common.fetch import CollectError, utc_now
from pdi.energy import collect, config, export

HEADER = "Area,ISO 3 code,Date,Area type,Electricity source,Is aggregated source,Generation (TWh)"


def ember_csv(countries=("USA", "MYS", "PHL", "SGP", "THA", "VNM"), start="2015-01", months=130,
              scale=1.0, demand_factor=1.0, extra_rows=()) -> str:  # fmt: skip
    rng = np.random.default_rng(0)
    lines = [HEADER]
    for k, iso in enumerate(countries):
        level = 400.0 if iso == "USA" else 5.0 + 4 * k
        for i, period in enumerate(pd.date_range(start, periods=months, freq="MS")):
            season = 1 + 0.1 * np.sin(2 * np.pi * period.month / 12)
            gen = scale * level * season * 1.003**i * period.days_in_month / 30
            gen *= 1 + rng.normal(0, 0.005)
            imports = 0.01 * gen
            rows = {
                "Total generation": gen,
                "Net imports": imports,
                "Demand": demand_factor * (gen + imports),
                "Coal": 0.4 * gen,
            }
            for name, value in rows.items():
                lines.append(f"{iso} name,{iso},{period:%Y-%m-%d},Country,{name},False,{value:.3f}")
    lines += list(extra_rows)
    # An area that is not one of ours, to be ignored.
    lines.append("World,,2020-01-01,Region,Demand,False,2000.0")
    return "\n".join(lines) + "\n"


def test_parse_keeps_only_demand_for_our_countries():
    demand, check = collect.parse_ember(ember_csv())
    assert list(demand.columns) == collect.TIDY
    assert set(demand["country"]) == {"US", "MY", "PH", "SG", "TH", "VN"}
    assert len(demand) == 6 * 130 and not demand.duplicated(["country", "period"]).any()
    assert collect.validate(demand, check) is demand


def test_a_missing_country_stops_the_run():
    demand, check = collect.parse_ember(ember_csv(countries=("USA", "MYS", "PHL", "SGP", "THA")))
    with pytest.raises(CollectError, match="VN"):
        collect.validate(demand, check)


def test_a_unit_change_stops_the_run():
    demand, check = collect.parse_ember(ember_csv(scale=1000.0))  # GWh read as TWh
    with pytest.raises(CollectError, match="unit change"):
        collect.validate(demand, check)


def test_demand_must_still_mean_generation_plus_net_imports():
    demand, check = collect.parse_ember(ember_csv(demand_factor=0.8))
    with pytest.raises(CollectError, match="generation plus net imports"):
        collect.validate(demand, check)


def test_duplicates_and_a_changed_layout_stop_the_run():
    dup = "USA name,USA,2015-01-01,Country,Demand,False,1.0"
    with pytest.raises(CollectError, match="two values"):
        collect.parse_ember(ember_csv(extra_rows=(dup,)))
    with pytest.raises(CollectError, match="layout"):
        collect.parse_ember("Area,Code,Date,Value\nUS,USA,2020-01-01,1\n")


def test_indonesia_is_picked_up_if_the_source_adds_it():
    demand, _ = collect.parse_ember(ember_csv(countries=("USA", "IDN")))
    assert "ID" in set(demand["country"])


def test_vintages_keep_first_seen_values(tmp_path):
    demand, _ = collect.parse_ember(ember_csv(countries=("USA",), months=24))
    path = tmp_path / "vintages.parquet"
    assert collect.append_vintages(demand, "2026-11-15T00:00:00+00:00", path) == 24
    assert collect.append_vintages(demand, "2026-11-22T00:00:00+00:00", path) == 0
    revised = demand.copy()
    revised.loc[revised.index[-1], "demand_twh"] *= 1.02
    assert collect.append_vintages(revised, "2026-11-29T00:00:00+00:00", path) == 1
    assert len(pd.read_parquet(path)) == 25


def test_a_recent_fetch_is_not_repeated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVED", tmp_path / "retrieved.json")
    monkeypatch.setattr(config, "DEMAND", tmp_path / "demand.parquet")
    assert not collect.is_fresh()
    config.DEMAND.write_bytes(b"x")
    now = utc_now()
    config.RETRIEVED.write_text(json.dumps({config.SOURCE: (now - timedelta(days=2)).isoformat()}))
    assert collect.is_fresh(now)
    config.RETRIEVED.write_text(json.dumps({config.SOURCE: (now - timedelta(days=7)).isoformat()}))
    assert not collect.is_fresh(now)


def test_growth_is_measured_per_day_and_the_index_is_100_in_the_base_year():
    # Flat demand per day: February's lower total must not read as a fall.
    periods = pd.date_range("2018-01", "2021-12", freq="MS")
    flat = pd.DataFrame(
        {"country": "SG", "period": periods, "demand_twh": 0.15 * periods.days_in_month}
    )
    latest, history = export.build(flat)
    assert np.allclose(history["yoy_pct"].dropna(), 0)
    assert np.allclose(history["index"], 100)
    assert np.allclose(history["year_yoy_pct"].dropna(), 0, atol=0.3)  # 2020 has a leap day
    assert latest.loc[0, "months"] == 48 and not latest.loc[0, "flagged"]


def test_an_unusual_latest_month_is_flagged_and_a_gap_stays_a_gap():
    demand, _ = collect.parse_ember(ember_csv())
    us = demand["country"] == "US"
    last = demand.loc[us, "period"].max()
    demand.loc[us & (demand["period"] == last), "demand_twh"] *= 1.15
    # Drop one month in the middle for Thailand.
    hole = pd.Timestamp("2022-06-01")
    demand = demand[~((demand["country"] == "TH") & (demand["period"] == hole))]
    latest, history = export.build(demand)
    row = latest.set_index("country").loc["US"]
    assert row["flagged"] and row["z"] > config.FLAG_Z
    assert not latest.set_index("country").loc["MY", "flagged"]
    th = history[(history["country"] == "TH") & (history["period"] == hole)]
    assert len(th) == 1 and th["demand_twh"].isna().all()
    assert "spreads" in export.sentence(row.rename_axis(None).to_frame().T.assign(name="x").iloc[0])


def test_export_writes_the_contract_tables(tmp_path, monkeypatch):
    demand, _ = collect.parse_ember(ember_csv())
    monkeypatch.setattr(config, "DEMAND", tmp_path / "demand.parquet")
    monkeypatch.setattr(config, "RETRIEVED", tmp_path / "retrieved.json")
    monkeypatch.setattr("pdi.common.export.SITE_DATA", tmp_path / "site")
    export.main()  # nothing collected yet: empty tables with a reason, not a crash
    empty = json.loads((tmp_path / "site/energy/demand_latest.json").read_text())
    assert empty["data"] == [] and "not been collected" in empty["meta"]["status"]
    demand.to_parquet(config.DEMAND, index=False)
    config.RETRIEVED.write_text(json.dumps({config.SOURCE: "2026-11-15T00:00:00+00:00"}))
    export.main()
    out = json.loads((tmp_path / "site/energy/demand_latest.json").read_text())
    assert out["meta"]["finding"]["label"] == "observation"
    assert out["meta"]["not_covered"] == ["Indonesia"]
    assert out["meta"]["sources"][0]["retrieved_at"] == "2026-11-15T00:00:00+00:00"
    assert len(out["data"]) == 6 and all(r["sentence"].endswith(".") for r in out["data"])
