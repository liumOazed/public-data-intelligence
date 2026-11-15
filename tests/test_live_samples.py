"""Every parser against a slice of what its live source actually sent.

The slices in tests/fixtures/live/ are refreshed by `python -m pdi.samples` (the "Refresh test
samples" workflow). The checks here are about format and plausibility, not exact numbers, so
a refresh only breaks a test when the source's format or units really changed.
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import pytest

from pdi.macro import collect, config, store
from pdi.tech_value import blender, retail, spec

CAT = config.catalogue()


def _meta(live, name):
    return json.loads((live / f"{name}.meta.json").read_text())


def test_every_slice_records_where_it_came_from(live):
    files = [p for p in live.iterdir() if not p.name.endswith(".meta.json")]
    assert len(files) >= 15
    for p in files:
        meta = _meta(live, p.name)
        assert meta["url"].startswith("https://")
        assert "api_key=REDACTED" in meta["url"] or "api_key" not in meta["url"]
        assert meta["full_response_bytes"] >= p.stat().st_size or "slice" in meta


# --- Tech Value -------------------------------------------------------------------------------


def test_spec_dump(live):
    df = spec.parse((live / "spec_rint2017.csv").read_bytes())
    assert len(df) >= 8
    assert {"Intel", "AMD"} <= set(df["vendor"])
    assert df["mhz"].between(500, 6000).all(), "MHz must arrive as MHz"
    assert (df["cores"] % df["chips"] == 0).all()
    assert df["disclosure_url"].str.startswith("https://www.spec.org/cpu2017/results/").all()
    assert not df["processor"].str.contains("  ").any()


def test_blender_launches(live):
    lines = (live / "blender_launches.jsonl").read_text().splitlines()
    results = [blender.parse_launch(json.loads(line)) for line in lines]
    ok = [r for r in results if not isinstance(r, str)]
    reasons = Counter(r for r in results if isinstance(r, str))
    # The slice was cut to hold several backends that parse and several launches that don't.
    assert {key[1] for key, _ in ok} >= {"CPU", "CUDA", "OPTIX"}
    assert all(score > 0 for _, score in ok)
    assert all(version.split(".")[0] >= "3" for (_, _, version), _ in ok)
    # Blender 2.x launches report render time, not samples per minute: a different measure,
    # filed as old_schema. On the first slice these were what the parser called unreadable.
    assert reasons["unreadable"] == 0
    assert reasons["old_schema"] >= 1


@pytest.mark.parametrize(
    "kind, key", [("cpu", "amd ryzen 7 9700x"), ("gpu", "nvidia rtx 5070")]
)  # fmt: skip
def test_central_computers_search_page(live, kind, key):
    url = "https://www.centralcomputer.com/catalogsearch/result/?q=x"
    items = retail.parse_magento_search((live / f"centralcomputer_{kind}.html").read_text(), url)
    assert len(items) >= 3
    assert all(x.price >= config_price(kind) for x in items)
    assert all(x.url and x.url.startswith("https://www.centralcomputer.com/") for x in items)
    kept = [x for x in items if retail.keep(x, kind, key)]
    assert kept, "at least one listing on the page is the part searched for"
    # The search also returns neighbours (9700F, 5070 Ti); the filter must drop them.
    assert len(kept) < len(items)


def config_price(kind: str) -> float:
    from pdi.tech_value import config as tv

    return tv.MIN_PART_PRICE[kind]


# --- Macro Monitor --------------------------------------------------------------------------


def _months_are_unique_and_monthly(df: pd.DataFrame) -> None:
    assert not df.duplicated(["series_id", "period"]).any()
    assert (df["period"].dt.day == 1).all()


def test_fred_csv(live):
    df = collect.parse_fred_csv((live / "fred_CPIAUCSL.csv").read_text(), CAT["US.cpi"])
    _months_are_unique_and_monthly(df)
    assert df["value"].between(250, 500).all()
    assert len(df) >= 20


def test_fred_api_and_notes(live):
    info = json.loads((live / "fred_CPIAUCSL_series.json").read_text())
    collect.check_fred_notes(info, CAT["US.cpi"])  # public domain: must not stop
    un = collect.parse_fred_api(
        json.loads((live / "fred_UNRATE_observations.json").read_text()), CAT["US.unemployment"]
    )
    _months_are_unique_and_monthly(un)
    assert un["value"].between(2, 15).all()
    usd = collect._monthly_mean(
        collect.parse_fred_api(
            json.loads((live / "fred_DTWEXBGS_observations.json").read_text()),
            CAT["US.dollar_broad"],
        )
    )
    _months_are_unique_and_monthly(usd)
    assert usd["value"].between(80, 160).all()


def test_bis_exchange_rates_and_policy_rates(live):
    fx = collect.parse_bis_csv((live / "bis_WS_XRU.csv").read_text(), "fx_usd")
    _months_are_unique_and_monthly(fx)
    assert set(fx["series_id"]) == {f"{c}.fx_usd" for c in config.SEA}
    last = fx.sort_values("period").groupby("series_id")["value"].last()
    assert 20000 < last["VN.fx_usd"] < 35000 and 2 < last["MY.fx_usd"] < 6
    rates = collect.parse_bis_csv((live / "bis_WS_CBPOL.csv").read_text(), "policy_rate")
    assert set(rates["series_id"]) == {"ID.policy_rate", "MY.policy_rate", "PH.policy_rate",
                                       "TH.policy_rate"}  # fmt: skip
    assert rates["value"].between(0, 20).all()
    # Every BIS series the live answer carries is in the catalogue, and none it lacks is.
    assert set(fx["series_id"]) | set(rates["series_id"]) <= set(CAT)
    assert not {"VN.policy_rate", "SG.policy_rate"} & set(CAT)


def test_opendosm(live):
    rows = json.loads((live / "opendosm_ipi.json").read_text())
    df = collect.parse_dosm(rows, config.DOSM_SERIES[0])
    assert len(df) == 3 and df["value"].between(50, 300).all()


def test_indeed(live):
    df = collect.parse_indeed((live / "indeed_US.csv").read_text())
    _months_are_unique_and_monthly(df)
    assert not df.empty and df["value"].between(30, 300).all()


@pytest.mark.parametrize(
    "name, indicator, countries",
    [
        ("imf_ITG_exports.csv", "exports_usd", {"PH", "TH", "VN"}),
        ("imf_IL_reserves.csv", "reserves", {"PH", "TH", "VN"}),
        ("imf_CPI.csv", "cpi", {"TH", "VN"}),
    ],
)
def test_imf(live, name, indicator, countries):
    df = collect.parse_imf_csv((live / name).read_text(), indicator)
    _months_are_unique_and_monthly(df)
    assert {s.split(".")[0] for s in df["series_id"]} == countries
    if indicator in ("exports_usd", "reserves"):
        # OBS_VALUE is in dollars, not millions: Vietnam ships tens of billions a month.
        assert df["value"].between(1e9, 1e12).all()
    else:
        assert df["value"].between(50, 200).all()
    store.check_units(df[df["series_id"].isin(CAT)])
