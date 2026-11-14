import json

import pytest

from pdi.tech_value import bestbuy, blender, config


def test_parse_launches(fixtures):
    lines = (fixtures / "blender_sample.jsonl").read_text().splitlines()
    parsed = [blender.parse_launch(json.loads(line)) for line in lines]
    assert parsed[0] == (("AMD Ryzen 9 7950X 16-Core Processor", "CPU", "4.5"), 540.0)
    # Patch version 4.5.1 is reported as 4.5.
    assert parsed[1] == (("NVIDIA GeForce RTX 4070 Ti SUPER", "OPTIX", "4.5"), 7100.0)
    assert parsed[2] == "multi_device"
    assert parsed[3] == "incomplete_scenes"
    assert parsed[4] == "old_schema"


def test_aggregate_needs_minimum_runs(fixtures, monkeypatch):
    lines = (fixtures / "blender_sample.jsonl").read_text().splitlines()
    df, skipped = blender.aggregate(lines)
    assert df.empty and skipped["used"] == 2

    monkeypatch.setattr(config, "BLENDER_MIN_RUNS", 3)
    df, _ = blender.aggregate([lines[0]] * 3 + ["not json"])
    assert len(df) == 1 and df.iloc[0]["n_runs"] == 3
    assert df.iloc[0]["median_score"] == 540.0


def test_bestbuy_page(fixtures):
    body = json.loads((fixtures / "bestbuy_cpu_page.json").read_text())
    rows = bestbuy.parse_page(body, "cpu", "2026-11-14")
    # The refurbished listing is a different measurement and is left out.
    assert [r["sku"] for r in rows] == ["1000001", "1000002", "1000004"]
    assert rows[0]["model_key"] == "amd ryzen 9 7950x"
    assert rows[0]["sale_price"] == 499.99 and rows[0]["price_update_date"] == "2026-11-10"
    assert rows[2]["model_key"] is None


def test_collect_without_key_is_explicit(monkeypatch):
    monkeypatch.delenv(config.BESTBUY_KEY_ENV, raising=False)
    with pytest.raises(bestbuy.MissingKey):
        bestbuy.collect()
