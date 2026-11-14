import json

import pytest

from pdi.tech_value import brightdata


def test_parse_keeps_new_us_parts_that_were_asked_for(fixtures):
    records = json.loads((fixtures / "brightdata_amazon_sample.json").read_text())
    wanted = {("cpu", "amd ryzen 7 9800x3d"), ("gpu", "nvidia rtx 5070")}
    rows = brightdata.parse(records, wanted, "2026-11-14")
    # Not the renewed chip, the whole PC, the 5070 Ti, the Canadian listing or the error row.
    assert [r["sku"] for r in rows] == ["B0AAA", "B0DDD", "B0EEE"]
    by_sku = {r["sku"]: r for r in rows}
    assert by_sku["B0DDD"]["sale_price"] == 579.99
    assert not by_sku["B0DDD"]["online_availability"]
    assert by_sku["B0AAA"]["source"] == "amazon_brightdata"


def test_no_key_means_no_request(monkeypatch):
    monkeypatch.delenv("BRIGHTDATA_API_KEY", raising=False)
    with pytest.raises(brightdata.MissingKey):
        brightdata.collect()
