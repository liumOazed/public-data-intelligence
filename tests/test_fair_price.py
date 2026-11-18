import numpy as np
import pandas as pd

from pdi.tech_value import config, fair_price


def _table(n=16, slope=1.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    score = np.exp(rng.uniform(4, 7, n))
    price = np.exp(0.5 + slope * np.log(score) + rng.normal(0, noise, n))
    return pd.DataFrame(
        {
            "kind": "cpu",
            "model_key": [f"amd ryzen 7 {1000 + i}x" for i in range(n)],
            "price_usd": price,
            "score": score,
            "n_listings": 3,
            "n_retailers": 2,
            "retailer": "store_a",
            "listing_url": None,
        }
    )


def test_baseline_is_kept_when_price_is_proportional_to_score():
    result = fair_price.evaluate(_table(slope=1.0))
    verdict = result.verdicts["cpu"]
    assert verdict["used"] == fair_price.BASELINE and not verdict["beats_baseline"]
    assert verdict["finding"]["label"] == "observation"
    board = result.leaderboard
    assert board["is_baseline"].sum() == 1 and board["used"].sum() == 1
    assert set(board["model"]) == set(fair_price.MODELS) | {fair_price.ENSEMBLE_NAME}


def test_a_line_replaces_the_baseline_when_price_rises_slower_than_score():
    result = fair_price.evaluate(_table(n=30, slope=0.5, noise=0.03))
    verdict = result.verdicts["cpu"]
    assert verdict["beats_baseline"] and verdict["used"] != fair_price.BASELINE
    assert verdict["finding"]["label"] == "prediction"


def test_cheap_and_dear_parts_are_flagged_and_single_listings_do_not_fit():
    table = _table(slope=1.0, noise=0.01)
    table.loc[0, "price_usd"] *= 0.5
    table.loc[1, "price_usd"] *= 2.0
    table.loc[2, ["price_usd", "n_listings"]] = [table.loc[2, "price_usd"] * 5, 1]
    parts = fair_price.evaluate(table).parts.set_index("model_key")
    assert parts.loc["amd ryzen 7 1000x", "reading"] == "below fair price"
    assert parts.loc["amd ryzen 7 1001x", "reading"] == "above fair price"
    odd = parts.loc["amd ryzen 7 1002x"]
    assert odd["confidence"] == "single listing" and odd["gap_pct"] > 300
    # The wild single listing did not move anyone else's fair price.
    assert parts.loc["amd ryzen 7 1005x", "reading"] == "near fair price"


def test_too_few_parts_gives_no_fair_price():
    result = fair_price.evaluate(_table(n=config.FAIR_PRICE_MIN_PARTS - 1))
    assert result.parts.empty and result.leaderboard.empty
    assert "fewer than" in result.verdicts["cpu"]["statement"]
    assert fair_price.evaluate(pd.DataFrame()).parts.empty


def test_structural_regime_model_separates_workstation_premium():
    rng = np.random.default_rng(7)
    n = 24
    score = np.exp(rng.uniform(6, 9, n))
    is_ws = np.array([i % 3 == 0 for i in range(n)])
    models = [
        f"nvidia rtx a{4000 + i * 100}" if ws else f"nvidia rtx 40{60 + (i % 4) * 10}"
        for i, ws in enumerate(is_ws)
    ]
    price = np.exp(0.3 + 0.9 * np.log(score) + 0.95 * is_ws + rng.normal(0, 0.04, n))
    table = pd.DataFrame(
        {
            "kind": "gpu",
            "model_key": models,
            "price_usd": price,
            "score": score,
            "n_listings": 3,
            "n_retailers": 2,
            "retailer": "store_a",
            "listing_url": None,
        }
    )
    result = fair_price.evaluate(table)
    verdict = result.verdicts["gpu"]
    assert verdict["beats_baseline"]
    assert verdict["n_models_beating_baseline"] >= 3
    assert set(result.parts["regime"]) == {"Consumer / Mainstream", "Workstation / Pro"}
    assert (result.parts["floor_p20_usd"] <= result.parts["fair_price_usd"]).all()
    assert (result.parts["ceiling_p80_usd"] >= result.parts["fair_price_usd"]).all()
    assert "ensemble_price_usd" in result.parts.columns
    assert "model_disagreement_pct" in result.parts.columns
    assert 0.0 <= verdict["p20_p80_loo_coverage_pct"] <= 100.0


def test_market_signals_frontier_distance_and_spread():
    table = _table(n=16, slope=1.0, noise=0.01)
    # Part 0: score=100, price=400; Part 1: score=110, price=430; Part 2: score=105, price=450
    table.loc[0, ["model_key", "score", "price_usd"]] = ["amd ryzen 5 9600x", 100.0, 400.0]
    table.loc[1, ["model_key", "score", "price_usd"]] = ["amd ryzen 7 9700x", 110.0, 430.0]
    table.loc[2, ["model_key", "score", "price_usd"]] = ["amd ryzen 7 7700x", 105.0, 450.0]
    table["median_price_usd"] = table["price_usd"] * 1.25
    table["source_median_price_usd"] = table["price_usd"] * 1.10
    table["retail_floor_usd"] = table["price_usd"]
    table["marketplace_floor_usd"] = table["price_usd"] * 1.15

    history = pd.DataFrame(
        {
            "date": ["2026-11-11", "2026-11-18"],
            "kind": ["cpu", "cpu"],
            "model_key": ["amd ryzen 7 7700x", "amd ryzen 7 7700x"],
            "price_usd": [500.0, 450.0],
        }
    )
    res = fair_price.evaluate(table, history=history)
    parts = res.parts.set_index("model_key")
    c = parts.loc["amd ryzen 7 7700x"]
    # Dominated by amd ryzen 7 9700x (score 110 >= 105, price 430 < 450) -> $20 drop (4.444%)
    assert np.isclose(c["drop_to_frontier_usd"], 20.0)
    assert np.isclose(c["drop_to_frontier_pct"], 20.0 / 450.0 * 100.0)
    assert c["blocking_rival"] == "amd ryzen 7 9700x"
    assert np.isclose(c["price_change_7d_pct"], -10.0)
    assert c["gen_stage"] == "Previous gen (-1)"
    assert c["channel_signal"] == "Retail undercuts marketplace"
    assert np.isclose(c["listing_spread_pct"], 20.0)
