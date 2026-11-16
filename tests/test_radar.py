import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pdi import health
from pdi.radar import collect, config, lead


def _months(n=130):
    return pd.date_range("2015-07-01", periods=n, freq="MS")


def test_parse_pageviews_and_sales():
    views = collect.parse_pageviews(
        {
            "items": [
                {"timestamp": "2015070100", "views": 10},
                {"timestamp": "2015080100", "views": 12},
            ]
        },
        "Laptop",
    )
    assert list(views["views"]) == [10, 12] and views["period"].iloc[1] == pd.Timestamp(
        "2015-08-01"
    )
    payload = [
        ["cell_value", "category_code", "data_type_code", "seasonally_adj", "time"],
        ["100", "443", "SM", "no", "2015-01"],
        ["90", "443", "SM", "yes", "2015-01"],
        ["5", "443", "E_SM", "no", "2015-01"],
        ["110", "443", "SM", "no", "2015-02"],
    ]
    sales = collect.parse_sales(payload, "443")
    assert list(sales["sales_musd"]) == [100.0, 110.0]


def _series(lead_strength, seed=0):
    rng = np.random.default_rng(seed)
    idx = _months()
    g_views = rng.normal(0, 0.05, len(idx))
    noise = rng.normal(0, 0.01, len(idx))
    g_sales = lead_strength * np.roll(g_views, 1) + noise
    views = pd.Series(1e6 * np.exp(np.cumsum(g_views)), index=idx)
    sales = pd.Series(5e3 * np.exp(np.cumsum(g_sales)), index=idx)
    return views, sales


def test_a_real_lead_passes_and_noise_does_not():
    views, sales = _series(0.8)
    assert lead.test_one(views, sales)["passes"]
    views, _ = _series(0.0, seed=1)
    _, sales = _series(0.0, seed=2)
    assert not lead.test_one(views, sales)["passes"]


def test_finding_is_a_prediction_only_on_a_pass():
    passed = pd.DataFrame([{"name": "A", "passes": True}, {"name": "B", "passes": False}])
    failed = pd.DataFrame([{"name": "A", "passes": False}])
    assert lead.finding(passed)["label"] == "prediction"
    assert lead.finding(failed)["label"] == "observation"


def test_pairs_are_four_with_unique_articles():
    articles = [a for _n, arts in config.PAIRS.values() for a in arts]
    assert len(config.PAIRS) == 4 and len(articles) == len(set(articles))


def test_health_skips_a_one_off_table(tmp_path):
    d = tmp_path / "radar"
    d.mkdir()
    payload = {"meta": {"generated_at": "2020-01-01T00:00:00+00:00", "one_off": True,
                        "sources": []}, "data": []}  # fmt: skip
    (d / "lead_test.json").write_text(json.dumps(payload))
    problems = health.check(tmp_path, now=datetime(2026, 11, 16, tzinfo=timezone.utc))
    assert not any(p.startswith("radar/") for p in problems)
