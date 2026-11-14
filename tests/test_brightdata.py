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


def test_used_and_renewed_listings_are_dropped_even_with_a_clean_title():
    base = {"title": "EVGA GeForce RTX 3070 XC3 Ultra Gaming 8GB", "currency": "USD"}
    records = [
        # Renewed listing whose title does not say so: only the category gives it away.
        {
            **base,
            "asin": "R1",
            "final_price": 384.97,
            "seller_name": "Some Store",
            "root_bs_category": "Amazon Renewed",
        },
        # Amazon's own used-goods store.
        {
            **base,
            "asin": "R2",
            "final_price": 392.14,
            "seller_name": "Amazon Resale",
            "root_bs_category": "Computer Graphics Cards",
        },
        {**base, "asin": "R3", "final_price": 349.99, "buybox_seller": "Renewed Technology Group"},
        {
            **base,
            "asin": "N1",
            "final_price": 585.0,
            "seller_name": "Amazon.com",
            "root_bs_category": "Computer Graphics Cards",
        },
    ]
    rows = brightdata.parse(records, {("gpu", "nvidia rtx 3070")}, "2026-11-14")
    assert [r["sku"] for r in rows] == ["N1"]


def test_only_amazon_or_the_brand_itself_counts_as_new():
    base = {"title": "AMD Ryzen 7 5700X 8-Core Desktop Processor", "currency": "USD"}
    records = [
        {**base, "asin": "M1", "final_price": 120.0, "seller_name": "Hcir Equipment"},
        {**base, "asin": "M2", "final_price": 130.0},  # no seller named at all
        {**base, "asin": "A1", "final_price": 189.0, "buybox_seller": "Amazon.com"},
        {**base, "asin": "B1", "final_price": 185.0, "seller_name": "AMD Official", "brand": "AMD"},
    ]
    rows = brightdata.parse(records, {("cpu", "amd ryzen 7 5700x")}, "2026-11-14")
    assert [r["sku"] for r in rows] == ["A1", "B1"]


def test_bestbuy_keeps_new_parts_sold_by_best_buy_itself(fixtures):
    records = json.loads((fixtures / "brightdata_bestbuy_sample.json").read_text())
    wanted = {("cpu", "amd ryzen 7 9800x3d"), ("gpu", "nvidia rtx 5070")}
    rows = brightdata.parse_bestbuy(records, wanted, "2026-11-14")
    # Not the whole PC, the 5060 Ti nobody asked for, the marketplace seller, the refurbished
    # card or the collector's timeout row.
    assert [r["sku"] for r in rows] == ["1001", "1002"]
    by_sku = {r["sku"]: r for r in rows}
    assert by_sku["1001"]["sale_price"] == 879.99
    assert by_sku["1001"]["source"] == "bestbuy_brightdata"
    assert by_sku["1001"]["online_availability"]
    assert not by_sku["1002"]["online_availability"]


def test_both_stores_are_started_before_either_is_read(monkeypatch):
    calls = []
    wanted = [("gpu", "nvidia rtx 5070")]
    monkeypatch.setenv("BRIGHTDATA_API_KEY", "k")
    monkeypatch.setattr(brightdata.retail, "targets", lambda: wanted)
    monkeypatch.setattr(brightdata, "start", lambda s, w: calls.append(("start", s)) or s)

    def finish(store, snapshot, w, day):
        calls.append(("finish", store))
        raise brightdata.CollectError("not ready in time")

    monkeypatch.setattr(brightdata, "finish", finish)
    brightdata.main()  # a store that fails is a warning, not a crash
    assert calls == [
        ("start", "bestbuy"),
        ("start", "amazon"),
        ("finish", "amazon"),
        ("finish", "bestbuy"),
    ]
