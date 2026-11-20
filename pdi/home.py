"""What the homepage shows from the exported tables. The page itself only lays these out."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd

from pdi.tech_value.dashboard import display_name


def _money(x: float) -> str:
    return f"${x:,.0f}" if round(x, 2) == round(x) else f"${x:,.2f}"


def top_price_move(items: pd.DataFrame, prices: pd.DataFrame) -> dict | None:
    """The largest flagged price move, preferring moves first seen in the last seven days.

    `items` is unusual/week and `prices` is tech-value/price_history. Returns None when no
    price move is flagged.
    """
    if items.empty or "rule" not in items:
        return None
    moves = items[items["rule"] == "price move"].copy()
    if moves.empty:
        return None
    if "new" in moves and moves["new"].astype(bool).any():
        moves = moves[moves["new"].astype(bool)]
    moves["size"] = moves["measure"].astype(float).abs()
    row = moves.sort_values(["size", "subject"], ascending=[False, True]).iloc[0]

    price = None
    if not prices.empty:
        day = str(row["period"])[:10]
        hit = prices[(prices["model_key"] == row["subject"]) & (prices["date"].astype(str) == day)]
        if not hit.empty:
            price = float(hit["price_usd"].iloc[0])
    change = float(row["measure"])
    return {
        "name": display_name(str(row["subject"])),
        "change": change,
        "pct": f"{'+' if change > 0 else '−'}{abs(100 * change):.1f}%",
        "direction": "rose" if change > 0 else "fell",
        "price": _money(price) if price is not None else None,
        "period": str(row["period"])[:10],
        "page": str(row["page"]),
    }


def this_week_card(items: pd.DataFrame, meta: dict, move: dict | None) -> str:
    """The "This week" card as HTML. Links are relative to the site root."""
    built = meta.get("built_at") or meta.get("generated_at")
    stamp = datetime.fromisoformat(built).strftime("%-d %b %Y") if built else ""
    if move:
        detail = f"Lowest US listing {move['direction']}"
        detail += f" to {move['price']}." if move["price"] else "."
        sign = "up" if move["change"] > 0 else "down"
        body = (
            f'<div class="home-move">'
            f'<div class="home-move-pct home-move-{sign}">{html.escape(move["pct"])}</div>'
            f'<div class="home-move-what"><strong>{html.escape(move["name"])}</strong>'
            f"<span>{html.escape(detail)}</span></div></div>"
        )
    elif items.empty:
        body = '<p class="home-move-none">Nothing is flagged this week.</p>'
    else:
        body = (
            f'<p class="home-move-none">No price moved enough to be flagged. '
            f"{len(items)} other items are flagged across the modules.</p>"
        )
    return (
        '<section class="home-card home-week" aria-labelledby="home-week-title">'
        f'<div class="home-card-top"><span class="home-kicker">This week</span>'
        f'<span class="home-date">{stamp}</span></div>'
        '<h2 id="home-week-title">What moved this week?</h2>'
        f"{body}"
        '<a class="home-btn home-btn-outline" href="unusual.html">See all flagged moves '
        '<span aria-hidden="true">→</span></a></section>'
    )


def latest_brief(briefs_dir: Path) -> dict | None:
    """The newest written brief, by file name (briefs are named by month, 2026-11.md etc.)."""
    files = sorted(briefs_dir.glob("20*.md"))
    if not files:
        return None
    f = files[-1]
    title = f.stem
    for line in f.read_text().splitlines():
        if line.startswith("title:"):
            title = line.split(":", 1)[1].strip().strip("\"'")
            break
    return {"title": title, "href": f"briefs/{f.stem}.html"}
