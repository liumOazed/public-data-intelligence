"""Trade and Supply Chain: parser, totals check, share arithmetic and the lead-time test."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pdi.common.fetch import CollectError
from pdi.trade import collect, config, export, lead, shares


def answer(rows, header=("I_COMMODITY", "I_COMMODITY_SDESC", "GEN_VAL_MO", "time", "COMM_LVL",
                         "CTY_CODE")):  # fmt: skip
    return [list(header), *[list(map(str, r)) for r in rows]]


def synthetic(months=120, start="2015-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """China loses ten points of chapter 85 after 2018 and Vietnam gains seven of them."""
    periods = pd.date_range(start, periods=months, freq="MS")
    rows = []
    for i, p in enumerate(periods):
        after = max(0, (p.year - 2018) * 12 + p.month) / 12
        cn85 = max(0.2, 0.40 - 0.02 * after)
        vn85 = min(0.30, 0.05 + 0.014 * after)
        world = {"85": 30e9, "61": 8e9, "98": 1e9, "10": 0.1e9}
        share = {
            "CN": {"85": cn85, "61": 0.30, "98": 0.1, "10": 0.0},
            "VN": {"85": vn85, "61": 0.15, "98": 0.0, "10": 0.1},
        }
        for other in ("ID", "MY", "PH", "SG", "TH"):
            share[other] = {"85": 0.02, "61": 0.03, "98": 0.0, "10": 0.0}
        for ch, w in world.items():
            w = w * (1 + 0.002 * i)
            rows.append(("WLD", ch, p, w))
            for partner, sh in share.items():
                if sh[ch] > 0:
                    rows.append((partner, ch, p, w * sh[ch]))
    imports = pd.DataFrame(rows, columns=["partner", "chapter", "period", "value"])
    totals = imports.groupby(["partner", "period"], as_index=False)["value"].sum()
    return imports, totals


def test_parse_reads_strings_as_numbers_and_months():
    payload = answer([("85", "ELECTRIC MACHINERY", 123456789, "2026-08", "HS2", "5520"),
                      ("61", "KNIT APPAREL", 1000, "2026-08", "HS2", "5520"),
                      ("8517", "PHONES", 5, "2026-08", "HS4", "5520")])  # fmt: skip
    df, names = collect.parse_chapters(payload, "VN")
    assert list(df.columns) == ["partner", "chapter", "period", "value"]
    assert set(df["chapter"]) == {"85", "61"} and names["85"] == "ELECTRIC MACHINERY"
    assert df["period"].iloc[0] == pd.Timestamp("2026-08-01") and df["value"].sum() == 123457789
    with pytest.raises(CollectError, match="no rows"):
        collect.parse([["GEN_VAL_MO", "time"]])
    with pytest.raises(CollectError, match="layout"):
        collect.parse([["VALUE", "time"], ["1", "2026-08"]])
    with pytest.raises(CollectError, match="two values"):
        collect.parse_chapters(answer([("85", "x", 1, "2026-08", "HS2", "5520")] * 2), "VN")


def test_chapters_must_add_up_to_the_published_total():
    imports, totals = synthetic(months=24)
    check = collect.validate(imports, totals)
    assert check["gap_share"].max() < 1e-12
    off = totals.copy()
    off.loc[(off["partner"] == "VN") & (off["period"] == off["period"].max()), "value"] *= 1.05
    with pytest.raises(CollectError, match="do not add up"):
        collect.validate(imports, off)
    with pytest.raises(CollectError, match="missing"):
        collect.validate(imports, totals[totals["partner"] != "CN"])
    with pytest.raises(CollectError, match="no chapter data"):
        collect.validate(imports[imports["partner"] != "TH"], totals)
    late = imports[~((imports["partner"] == "SG") & (imports["period"] == imports["period"].max()))]
    late_totals = late.groupby(["partner", "period"], as_index=False)["value"].sum()
    with pytest.raises(CollectError, match="different months"):
        collect.validate(late, late_totals)


def test_share_shift_recovers_the_built_in_move():
    imports, _ = synthetic()
    rolled = shares.with_shares(shares.twelve_month(imports))
    shift = shares.shifts(rolled)
    row = shift.set_index(["partner", "chapter"])
    assert row.loc[("CN", "85"), "share_base"] == pytest.approx(40.0, abs=0.01)
    assert row.loc[("CN", "85"), "shift_since_base"] < -9
    assert row.loc[("VN", "85"), "shift_since_base"] > 6
    assert row.loc[("ID", "85"), "shift_since_base"] == pytest.approx(0, abs=1e-6)
    # A chapter a partner never ships is a zero share, not a missing one.
    assert ("CN", "10") in row.index and row.loc[("CN", "10"), "share_pct"] == 0
    # "ALL" is every chapter together, special chapters included.
    assert row.loc[("CN", "ALL"), "share_pct"] == pytest.approx(
        100 * row.loc[("CN", "ALL"), "usd_12m"] / row.loc[("CN", "ALL"), "world_12m"]
    )
    china = shares.china_and_sea(shift, {"85": "Electrical machinery"})
    # Small and special chapters are left out; the biggest fall for China comes first.
    assert list(china["chapter"]) == ["85", "61"]
    assert china.loc[0, "largest_gainer"] == "VN" and china.loc[0, "name"] == "Electrical machinery"


def _levels(growth: np.ndarray, start="2011-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(growth), freq="MS")
    return pd.Series(100 * np.exp(np.cumsum(growth) / 100), index=idx)


def test_lead_test_passes_only_when_us_imports_carry_information():
    rng = np.random.default_rng(1)
    n = 190
    common = rng.normal(0, 3, n)
    us = _levels(common + rng.normal(0, 0.3, n))
    linked = _levels(common + rng.normal(0, 0.3, n))
    unrelated = _levels(rng.normal(0, 3, n))
    good = lead.test_one(us, linked)
    assert good["passes"] and good["mae_model"] < good["mae_baseline"]
    assert good["test_start"] == "2023-01" and good["correlation"]["0"] > 0.8
    bad = lead.test_one(us, unrelated)
    assert not bad["passes"] and bad["verdict"] == "no signal found"
    short = lead.test_one(us.iloc[-40:], linked.iloc[-40:])
    assert short["verdict"] == "not enough data" and not short["passes"]


def test_export_writes_every_table_and_never_forecasts_without_a_pass(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir()
    for name in ("IMPORTS", "TOTALS", "CHAPTERS", "RETRIEVED", "LEAD_TEST"):
        monkeypatch.setattr(config, name, store / getattr(config, name).name)
    monkeypatch.setattr("pdi.common.export.SITE_DATA", tmp_path / "site")
    monkeypatch.setattr(export.macro, "OBSERVATIONS", store / "none.parquet")
    monkeypatch.setattr(export.macro, "STORE_DIR", store)
    export.main()
    empty = json.loads((tmp_path / "site/trade/partners.json").read_text())
    assert empty["data"] == [] and "not been collected" in empty["meta"]["status"]

    imports, totals = synthetic()
    imports.to_parquet(config.IMPORTS, index=False)
    totals.to_parquet(config.TOTALS, index=False)
    config.CHAPTERS.write_text(json.dumps({"85": "Electrical machinery", "61": "Knit apparel"}))
    config.RETRIEVED.write_text(json.dumps({config.SOURCE: "2026-11-15T00:00:00+00:00"}))
    export.main()
    out = {t: json.loads((tmp_path / f"site/trade/{t}.json").read_text()) for t in export.TABLES}
    assert out["partners"]["meta"]["finding"]["label"] == "observation"
    assert len(out["partners"]["data"]) == 7
    assert out["totals_check"]["meta"]["criterion_met"] is True
    assert out["china_shift"]["data"][0]["chapter"] == "85"
    # No export figures yet: the test is reported as not run, with no finding attached.
    assert out["lead_test"]["meta"]["test_ran"] is False
    assert out["lead_test"]["meta"]["finding"] is None

    rng = np.random.default_rng(2)
    obs = []
    for c in config.SEA:
        us = totals[totals["partner"] == c].set_index("period")["value"]
        noise = np.exp(rng.normal(0, 0.05, len(us)))
        obs.append(pd.DataFrame({"series_id": f"{c}.exports_usd", "period": us.index,
                                 "value": us.to_numpy() * 10 * noise}))  # fmt: skip
    pd.concat(obs).to_parquet(store / "obs.parquet", index=False)
    monkeypatch.setattr(export.macro, "OBSERVATIONS", store / "obs.parquet")
    export.main()
    lead_out = json.loads((tmp_path / "site/trade/lead_test.json").read_text())
    assert lead_out["meta"]["test_ran"] is True
    assert lead_out["meta"]["finding"]["label"] in ("observation", "prediction")
    assert len(lead_out["data"]) == 6 and "corr_shift_0" in lead_out["meta"]["columns"]
