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
    assert set(board["model"]) == set(fair_price.MODELS)


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
