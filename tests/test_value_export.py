import json

import pandas as pd

from pdi.common import registry
from pdi.common.export import write_table
from pdi.common.fetch import redact
from pdi.tech_value import value


def _prices():
    base = dict(online_availability=True, regular_price=500.0, url="u", retrieved_at="t")
    return pd.DataFrame(
        [
            dict(
                date="2026-11-13",
                sku="1",
                name="A",
                category="cpu",
                sale_price=520.0,
                price_update_date="2026-11-01",
                model_key="amd ryzen 9 7950x",
                **base,
            ),
            dict(
                date="2026-11-14",
                sku="1",
                name="A",
                category="cpu",
                sale_price=500.0,
                price_update_date="2026-11-14",
                model_key="amd ryzen 9 7950x",
                **base,
            ),
            dict(
                date="2026-11-14",
                sku="2",
                name="B",
                category="cpu",
                sale_price=450.0,
                price_update_date="2026-11-10",
                model_key="amd ryzen 9 7950x",
                **base,
            ),
            dict(
                date="2026-11-14",
                sku="3",
                name="C",
                category="cpu",
                sale_price=300.0,
                price_update_date="2026-11-10",
                model_key="intel core i5-14600k",
                **base,
            ),
            dict(
                date="2026-11-14",
                sku="4",
                name="paste",
                category="cpu",
                sale_price=10.0,
                price_update_date="2026-11-10",
                model_key=None,
                **base,
            ),
        ]
    )


def _blender():
    row = dict(n_runs=40, p25_score=500.0, p75_score=560.0, retrieved_at="t")
    return pd.DataFrame(
        [
            dict(
                device_name="AMD Ryzen 9 7950X 16-Core Processor",
                device_type="CPU",
                blender_version="4.5",
                median_score=540.0,
                **row,
            ),
            dict(
                device_name="AMD Ryzen 9 7950X 16-Core Processor",
                device_type="CPU",
                blender_version="4.2",
                median_score=900.0,
                **row,
            ),
            dict(
                device_name="NVIDIA GeForce RTX 4070",
                device_type="OPTIX",
                blender_version="4.5",
                median_score=5000.0,
                **row,
            ),
        ]
    )


def test_value_table_uses_cheapest_latest_price_and_one_blender_version():
    bench, version = value.benchmark_by_model(_blender())
    assert version == "4.5"
    table = value.value_table(_prices(), bench)
    top = table.iloc[0]
    assert top["model_key"] == "amd ryzen 9 7950x"
    assert top["price_usd"] == 450.0 and top["price_date"] == "2026-11-10"
    assert top["n_listings"] == 2
    assert top["score"] == 540.0  # not the 4.2 score
    assert abs(top["score_per_dollar"] - 1.2) < 1e-9
    # Priced but unbenchmarked models stay visible with an empty score.
    other = table[table["model_key"] == "intel core i5-14600k"].iloc[0]
    assert pd.isna(other["score"]) and pd.isna(other["score_per_dollar"])


def test_coverage_and_history():
    bench, _ = value.benchmark_by_model(_blender())
    table = value.value_table(_prices(), bench)
    cov = value.coverage(_prices(), table).iloc[0]
    assert (cov["listings"], cov["listings_recognised"]) == (4, 3)
    assert (cov["models_priced"], cov["models_with_score"]) == (2, 1)
    hist = value.price_history(_prices())
    amd = hist[hist["model_key"] == "amd ryzen 9 7950x"]
    assert list(amd["price_usd"]) == [520.0, 450.0]


def test_empty_inputs_give_empty_tables():
    assert value.value_table(pd.DataFrame(), pd.DataFrame()).empty
    assert value.benchmark_by_model(pd.DataFrame())[0].empty
    assert value.value_table(_prices(), pd.DataFrame())["score"].isna().all()


def test_export_envelope_carries_the_label(tmp_path):
    df = pd.DataFrame({"a": [1.0, float("nan")], "d": [pd.Timestamp("2026-01-01"), pd.NaT]})
    path = write_table(
        "tech-value",
        "t",
        df,
        sources=["spec_cpu2017"],
        description="d",
        retrieved_at={"spec_cpu2017": "2026-11-14T00:00:00+00:00"},
        out_dir=tmp_path,
    )
    payload = json.loads(path.read_text())
    assert payload["data"][1] == {"a": None, "d": None}
    src = payload["meta"]["sources"][0]
    for field in ("name", "url", "licence", "frequency", "typical_lag", "retrieved_at"):
        assert src[field]
    assert payload["meta"]["rows"] == 2


def test_registry_is_complete():
    for key, src in registry.load().items():
        assert src["status"] in registry.STATUSES, key


def test_api_keys_never_reach_disk():
    assert "SECRET" not in redact("https://api.bestbuy.com/v1/products?apiKey=SECRET&page=1")


def test_a_price_far_below_the_median_cannot_set_the_headline():
    prices = _prices().assign(source="store_a")
    latest = prices[prices["date"] == prices["date"].max()]
    row = latest[latest["model_key"] == "amd ryzen 9 7950x"].iloc[[0]]
    extra = pd.concat(
        [row.assign(sale_price=460.0, source="store_b"), row.assign(sale_price=90.0, sku="bad")]
    )
    bench, _ = value.benchmark_by_model(_blender())
    top = value.value_table(pd.concat([prices, extra]), bench).iloc[0]
    assert top["price_usd"] == 450.0 and top["n_ignored_low"] == 1
    assert top["n_retailers"] == 2
    assert top["median_price_usd"] >= top["price_usd"]


def test_a_weekly_source_still_counts_between_its_runs():
    prices = _prices().assign(source="store_a")
    latest = prices[prices["date"] == prices["date"].max()]
    row = latest[latest["model_key"] == "amd ryzen 9 7950x"].iloc[[0]]
    earlier = (pd.Timestamp(prices["date"].max()) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    stale = (pd.Timestamp(prices["date"].max()) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    extra = pd.concat(
        [
            row.assign(sale_price=440.0, source="store_weekly", date=earlier),
            row.assign(sale_price=300.0, source="store_gone", date=stale),
        ]
    )
    bench, _ = value.benchmark_by_model(_blender())
    top = value.value_table(pd.concat([prices, extra]), bench).iloc[0]
    assert top["price_usd"] == 440.0 and top["retailer"] == "store_weekly"
    assert top["n_retailers"] == 2


def test_single_listing_is_marked_low_confidence():
    prices = pd.DataFrame(
        {
            "date": ["2026-11-14"] * 3,
            "source": ["store_a", "store_b", "store_a"],
            "category": ["cpu"] * 3,
            "model_key": ["ryzen 7 7700x", "ryzen 7 7700x", "ryzen 5 5600"],
            "name": ["a", "b", "c"],
            "url": ["u1", "u2", "u3"],
            "sale_price": [300.0, 310.0, 120.0],
            "regular_price": [300.0, 310.0, 120.0],
            "online_availability": [True] * 3,
            "price_update_date": ["2026-11-14"] * 3,
        }
    )
    table = value.value_table(prices, pd.DataFrame()).set_index("model_key")
    assert table.loc["ryzen 7 7700x", "confidence"] == "normal"
    assert table.loc["ryzen 5 5600", "confidence"] == "low"
