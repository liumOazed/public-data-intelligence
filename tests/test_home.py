import pandas as pd

from pdi import home

COLS = ["subject", "period", "rule", "measure", "new", "page"]


def _items(rows):
    return pd.DataFrame(rows, columns=COLS)


PRICES = pd.DataFrame(
    [
        ("2026-11-20", "cpu", "amd ryzen 5 5500", 90.0, 3),
        ("2026-11-20", "cpu", "amd ryzen 5 7600x", 145.99, 2),
        ("2026-11-20", "gpu", "nvidia rtx 5060 ti", 499.0, 4),
    ],
    columns=["date", "kind", "model_key", "price_usd", "n_listings"],
)
PAGE = "tech-value/index.qmd"


def test_largest_new_price_move_wins_and_is_named_for_display():
    items = _items([
        ("amd ryzen 5 5500", "2026-11-20", "price move", -0.10, True, PAGE),
        ("amd ryzen 5 7600x", "2026-11-20", "price move", -0.513, True, PAGE),
        ("nvidia rtx 5060 ti", "2026-11-20", "price move", 0.9, False, PAGE),
        ("US CPI", "2026-10-01", "unusual move", 9.0, True, "macro/index.qmd"),
    ])  # fmt: skip
    move = home.top_price_move(items, PRICES)
    assert move["name"] == "AMD Ryzen 5 7600X"
    assert move["pct"] == "−51.3%"
    assert move["direction"] == "fell"
    assert move["price"] == "$145.99"


def test_standing_moves_are_used_when_nothing_is_new_and_whole_dollars_drop_cents():
    items = _items([
        ("amd ryzen 5 5500", "2026-11-20", "price move", -0.10, False, PAGE),
        ("nvidia rtx 5060 ti", "2026-11-20", "price move", 0.06, False, PAGE),
    ])  # fmt: skip
    move = home.top_price_move(items, PRICES)
    assert move["name"] == "AMD Ryzen 5 5500"
    assert move["price"] == "$90"


def test_no_price_move_gives_none_and_the_card_says_so():
    items = _items([("US CPI", "2026-10-01", "unusual move", 9.0, True, "macro/index.qmd")])
    assert home.top_price_move(items, PRICES) is None
    card = home.this_week_card(items, {"built_at": "2026-11-20T01:35:51+00:00"}, None)
    assert "No price moved enough" in card and "20 Nov 2026" in card
    empty = home.this_week_card(_items([]), {}, None)
    assert "Nothing is flagged this week" in empty


def test_latest_brief_reads_the_newest_file(tmp_path):
    assert home.latest_brief(tmp_path) is None
    (tmp_path / "2026-11.md").write_text('---\ntitle: "November 2026"\n---\n')
    (tmp_path / "2026-12.md").write_text('---\ntitle: "December 2026"\n---\n')
    assert home.latest_brief(tmp_path) == {"title": "December 2026", "href": "briefs/2026-12.html"}
