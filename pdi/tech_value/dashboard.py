"""Value ranking dashboard: the calculations behind the charts on the Value ranking page.

Everything here reads the tables the export step already writes (value, benchmarks,
price_history). Nothing is collected or re-priced: the price is always the lowest new
listing the ranking uses, and the benchmark is the Blender median the ranking uses.

The page's charts recompute some of this in the browser when filters change
(site/tech-value/value-dashboard.js). The two implementations are checked against each
other in tests/test_value_dashboard.py.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from pdi.tech_value import config, names

KINDS = {"gpu": "Graphics cards", "cpu": "Desktop CPUs"}
KIND_SHORT = {"gpu": "GPU", "cpu": "CPU"}
MAKERS = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel"}

# Sample-size gates, fixed before looking at what they let through.
MIN_FOR_PERCENTILES = 10  # fewer ranked models than this: no percentiles
MIN_FOR_MARKET_INSIGHT = 8  # fewer than this: no "compared with the median" sentences
MIN_HISTORY_DAYS = 14  # a price trend needs at least this span between first and last day
STABLE_PRICE_SHARE = 0.02  # a change smaller than this over the span is called stable
DEFAULT_BRACKETS = {"gpu": [300, 500, 750, 1000, 1500], "cpu": [150, 250, 400, 600]}


def maker(key: str) -> str:
    """Manufacturer from the normalised model key, whose first word is always the maker."""
    first = str(key).split(" ", 1)[0]
    return MAKERS.get(first, first.title())


def ranked(value: pd.DataFrame) -> pd.DataFrame:
    """Rows that carry a valid ranking: a benchmark score and a positive price."""
    if value.empty:
        cols = ["kind", "model_key", "price_usd", "score", "score_per_dollar", "maker"]
        return pd.DataFrame(columns=cols)
    ok = value["score"].notna() & value["price_usd"].notna() & (value["price_usd"] > 0)
    out = value[ok].copy()
    out["score_per_dollar"] = out["score"] / out["price_usd"]
    out["maker"] = out["model_key"].map(maker)
    return out


def price_checked(bench: pd.DataFrame) -> pd.DataFrame:
    """The models the price collectors look up: the most-benchmarked of each kind."""
    if bench.empty:
        return bench
    parts = []
    for kind in KINDS:
        part = bench[bench["kind"] == kind]
        active = part[part["model_key"].map(lambda k, kd=kind: names.retail_active(kd, str(k)))]
        pool = active if not active.empty else part
        parts.append(
            pool.sort_values("n_runs", ascending=False).head(config.CRAWLER_MODELS_PER_KIND)
        )
    return pd.concat(parts, ignore_index=True)


def quantile(values, q: float) -> float | None:
    """Linear-interpolation quantile (numpy's default), the same rule the page uses."""
    arr = np.sort(np.asarray([v for v in values if v is not None], dtype=float))
    if len(arr) == 0:
        return None
    return float(np.quantile(arr, q))


def share_below(values, x: float) -> float | None:
    """Share of the other models with a strictly lower value: 'faster than N% of ...'."""
    others = [v for v in values if v is not None]
    if len(others) < 2:
        return None
    below = sum(1 for v in others if v < x)
    return below / (len(others) - 1)


def dominates(a: dict, b: dict) -> bool:
    """a dominates b: no more expensive, no slower, and strictly better on one of the two."""
    return (
        a["price_usd"] <= b["price_usd"]
        and a["score"] >= b["score"]
        and (a["price_usd"] < b["price_usd"] or a["score"] > b["score"])
    )


def frontier(df: pd.DataFrame) -> pd.Series:
    """True for models no other model in df dominates (the efficiency frontier)."""
    rows = df[["price_usd", "score"]].to_dict("records")
    return pd.Series(
        [not any(dominates(o, r) for o in rows if o is not r) for r in rows], index=df.index
    )


def best_dominator(df: pd.DataFrame, key: str) -> dict | None:
    """The alternative that dominates `key` with the highest score per dollar, with the gaps."""
    me = df[df["model_key"] == key]
    if me.empty:
        return None
    r = me.iloc[0].to_dict()
    rivals = [o for o in df.to_dict("records") if o["model_key"] != key and dominates(o, r)]
    if not rivals:
        return None
    best = max(rivals, key=lambda o: o["score"] / o["price_usd"])
    return {
        "model_key": best["model_key"],
        "more_score_pct": 100 * (best["score"] / r["score"] - 1),
        "lower_price_pct": 100 * (1 - best["price_usd"] / r["price_usd"]),
    }


def upgrade(df: pd.DataFrame, base: str, other: str) -> dict:
    """Price and score difference of moving from `base` to `other`, same kind only."""
    a = df[df["model_key"] == base].iloc[0]
    b = df[df["model_key"] == other].iloc[0]
    if a["kind"] != b["kind"]:
        raise ValueError("compare models of the same kind only")
    return {
        "price_diff_usd": float(b["price_usd"] - a["price_usd"]),
        "price_pct": 100 * float(b["price_usd"] / a["price_usd"] - 1),
        "score_diff": float(b["score"] - a["score"]),
        "score_pct": 100 * float(b["score"] / a["score"] - 1),
        "value_pct": 100 * float(b["score_per_dollar"] / a["score_per_dollar"] - 1),
    }


def brackets(df: pd.DataFrame, edges: list[float]) -> pd.DataFrame:
    """Per price bracket: model count, best score per dollar, top score, medians.

    Brackets are [lo, hi): a $500 part sits in "$500 to $750", not "$250 to $500".
    """
    edges = sorted(e for e in edges if e > 0)
    bounds = [0.0, *edges, math.inf]
    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=False):
        part = df[(df["price_usd"] >= lo) & (df["price_usd"] < hi)]
        row = {"low": lo, "high": hi, "n": int(len(part))}
        if len(part):
            best = part.loc[part["score_per_dollar"].idxmax()]
            top = part.loc[part["score"].idxmax()]
            row |= {
                "best_value": best["model_key"],
                "best_value_ratio": float(best["score_per_dollar"]),
                "top_score_model": top["model_key"],
                "top_score": float(top["score"]),
                "median_score": float(part["score"].median()),
                "median_price": float(part["price_usd"].median()),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def funnel(bench: pd.DataFrame, value: pd.DataFrame) -> pd.DataFrame:
    """Benchmark universe -> price-checked -> priced -> ranked, per kind."""
    checked = price_checked(bench)
    checked_pairs = (
        set(zip(checked["kind"], checked["model_key"], strict=False)) if len(checked) else set()
    )
    priced_pairs = (
        set(zip(value["kind"], value["model_key"], strict=False)) if len(value) else set()
    )
    looked_up = checked_pairs | priced_pairs
    rk = ranked(value)
    rows = []
    for kind in KINDS:
        rows.append(
            {
                "kind": kind,
                "benchmarked": int((bench["kind"] == kind).sum()) if len(bench) else 0,
                "price_checked": sum(1 for kd, _ in looked_up if kd == kind),
                "priced": int((value["kind"] == kind).sum()) if len(value) else 0,
                "ranked": int((rk["kind"] == kind).sum()) if len(rk) else 0,
            }
        )
    return pd.DataFrame(rows)


def freshness(bench: pd.DataFrame, value: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """For each price-checked model: fresh, stale or never priced.

    Fresh means the ranking has a price for it (the ranking already drops prices older than
    PRICE_MAX_AGE_DAYS). Stale means a price was seen before but none recently enough.
    """
    checked = price_checked(bench)
    if checked.empty and value.empty:
        return pd.DataFrame(columns=["kind", "model_key", "status", "last_seen"])
    in_value = set(zip(value["kind"], value["model_key"], strict=False)) if len(value) else set()
    checked_list = (
        list(zip(checked["kind"], checked["model_key"], strict=False)) if len(checked) else []
    )
    seen_set = set(checked_list)
    for pair in sorted(in_value):
        if pair not in seen_set:
            checked_list.append(pair)
            seen_set.add(pair)
    last_seen = (
        history.groupby(["kind", "model_key"])["date"].max().to_dict() if len(history) else {}
    )
    rows = []
    for kind, key in checked_list:
        seen = last_seen.get((kind, key))
        if (kind, key) in in_value:
            status = "fresh"
        elif seen:
            status = "stale"
        else:
            status = "never priced"
        rows.append({"kind": kind, "model_key": key, "status": status, "last_seen": seen})
    return pd.DataFrame(rows)


def history_summary(history: pd.DataFrame) -> dict:
    """Collection days, gaps, and per-model change, gated on the span of history.

    Gaps are calendar days between the first and last collection day with no collection at
    all. Nothing is interpolated: a change is first observed price to last observed price.
    """
    if history.empty:
        return {"days": [], "gaps": [], "span_days": 0, "trend_ok": False, "models": []}
    days = sorted({str(d)[:10] for d in history["date"]})
    first, last = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
    span = (last - first).days
    have = set(days)
    gaps = [
        str(first + timedelta(days=i))
        for i in range(span + 1)
        if str(first + timedelta(days=i)) not in have
    ]
    trend_ok = span >= MIN_HISTORY_DAYS
    models = []
    for (kind, key), g in history.groupby(["kind", "model_key"]):
        g = g.sort_values("date")
        p0, p1 = float(g["price_usd"].iloc[0]), float(g["price_usd"].iloc[-1])
        own_span = (date.fromisoformat(str(g["date"].iloc[-1])[:10])
                    - date.fromisoformat(str(g["date"].iloc[0])[:10])).days  # fmt: skip
        models.append(
            {
                "kind": kind,
                "model_key": key,
                "n_obs": int(len(g)),
                "first_date": str(g["date"].iloc[0])[:10],
                "last_date": str(g["date"].iloc[-1])[:10],
                "first_price": p0,
                "last_price": p1,
                "low": float(g["price_usd"].min()),
                "median": float(g["price_usd"].median()),
                "high": float(g["price_usd"].max()),
                "change_usd": p1 - p0,
                "change_pct": 100 * (p1 / p0 - 1) if p0 else None,
                "trend_ok": trend_ok and own_span >= MIN_HISTORY_DAYS,
            }
        )
    return {"days": days, "gaps": gaps, "span_days": span, "trend_ok": trend_ok, "models": models}


def _usd(x: float) -> str:
    return f"${x:,.0f}"


UPPER_WORDS = {"rtx", "rx", "gtx", "xt", "xtx", "gre", "ai"}


def display_name(key: str) -> str:
    """'nvidia rtx 5060 ti' -> 'NVIDIA RTX 5060 Ti'; 'intel core i7-14700kf' -> '... i7-14700KF'."""
    words = []
    for w in str(key).split():
        if w in MAKERS:
            words.append(MAKERS[w])
        elif w in UPPER_WORDS:
            words.append(w.upper())
        elif w == "ti":
            words.append("Ti")
        elif any(ch.isdigit() for ch in w):
            up = w.upper()
            words.append("i" + up[1:] if len(up) > 1 and up[0] == "I" and up[1].isdigit() else up)
        else:
            words.append(w.title())
    return " ".join(words)


_name = display_name


def insights(bench: pd.DataFrame, value: pd.DataFrame, history: pd.DataFrame) -> list[dict]:
    """Short sentences computed from the tables, each with the numbers behind it.

    An insight that needs a minimum sample is left out, not softened, when the sample is short.
    """
    out: list[dict] = []
    rk = ranked(value)
    fun = funnel(bench, value).set_index("kind")
    for kind, label in KIND_SHORT.items():
        f = fun.loc[kind]
        if f["price_checked"]:
            prefix = "Only " if f["ranked"] * 2 < f["price_checked"] else ""
            out.append({
                "kind": kind, "topic": "Coverage",
                "text": (
                    f"{prefix}{f['ranked']} of the {f['price_checked']} {label}s the "
                    f"collectors look up have a matching price, out of {f['benchmarked']} "
                    f"{label}s with a benchmark score."
                ),
                "detail": f"Benchmarked: {f['benchmarked']}. Price-checked: "
                          f"{f['price_checked']}. Priced: "
                          f"{f['priced']}. Ranked: {f['ranked']}.",
            })  # fmt: skip
        part = rk[rk["kind"] == kind]
        n = len(part)
        if n < MIN_FOR_MARKET_INSIGHT:
            continue
        med = float(part["score_per_dollar"].median())
        best = part.loc[part["score_per_dollar"].idxmax()]
        lead = 100 * (best["score_per_dollar"] / med - 1)
        out.append({
            "kind": kind, "topic": "Value",
            "text": f"{_name(best['model_key'])} gives {lead:.0f}% "
                    f"more benchmark score per dollar than the median ranked {label} (n = {n})."
                    + (" Its price rests on a single listing." if best.get("confidence") == "low"
                       else ""),
            "detail": f"{_name(best['model_key'])}: {best['score']:,.0f} points at "
                      f"{_usd(best['price_usd'])} = {best['score_per_dollar']:.2f} points per $. "
                      f"Median of {n} ranked {label}s: {med:.2f} points per $.",
        })  # fmt: skip
        top = part.loc[part["score"].idxmax()]
        if top["model_key"] != best["model_key"]:
            rank = int((part["score_per_dollar"] > top["score_per_dollar"]).sum()) + 1
            out.append({
                "kind": kind, "topic": "Value",
                "text": f"The fastest {label}, {_name(top['model_key'])}, ranks {rank} of {n} "
                        "on score per dollar.",
                "detail": f"{_name(top['model_key'])}: {top['score']:,.0f} points at "
                          f"{_usd(top['price_usd'])} = {top['score_per_dollar']:.2f} points per $.",
            })  # fmt: skip
        on = frontier(part)
        dominated = part[~on]
        if len(dominated):
            names = ", ".join(_name(k) for k in dominated["model_key"])
            out.append({
                "kind": kind, "topic": "Dominated",
                "text": f"{len(dominated)} of {n} ranked {label}s are dominated: another {label} "
                        "costs no more and scores at least as high.",
                "detail": f"Dominated: {names}.",
            })  # fmt: skip
        lo, hi = quantile(part["price_usd"], 0.25), quantile(part["price_usd"], 0.75)
        out.append({
            "kind": kind, "topic": "Market",
            "text": f"The middle half of ranked {label}s cost between {_usd(lo)} and {_usd(hi)}.",
            "detail": f"25th and 75th percentile of the lowest listing across {n} ranked {label}s.",
        })  # fmt: skip
        eff = part[on].sort_values("price_usd")
        if best["model_key"] in set(eff["model_key"]):
            above = eff[eff["price_usd"] > best["price_usd"]]
            if len(above):
                nxt = above.iloc[0]
                u = upgrade(part, best["model_key"], nxt["model_key"])
                out.append({
                    "kind": kind, "topic": "Upgrade",
                    "text": f"Moving from {_name(best['model_key'])} to {_name(nxt['model_key'])} "
                            f"costs {u['price_pct']:.0f}% more "
                            f"for {u['score_pct']:.0f}% more score.",
                    "detail": f"{_usd(best['price_usd'])} to {_usd(nxt['price_usd'])} "
                              f"(+{_usd(u['price_diff_usd'])}); {best['score']:,.0f} to "
                              f"{nxt['score']:,.0f} points. The next model up on the efficiency "
                              "frontier from the best-value pick.",
                })  # fmt: skip
    hs = history_summary(history)
    if hs["trend_ok"]:
        for kind, label in KIND_SHORT.items():
            ms = [m for m in hs["models"] if m["kind"] == kind and m["trend_ok"]]
            if len(ms) < MIN_FOR_MARKET_INSIGHT:
                continue
            up = [m for m in ms if m["change_pct"] > 100 * STABLE_PRICE_SHARE]
            down = [m for m in ms if m["change_pct"] < -100 * STABLE_PRICE_SHARE]
            out.append({
                "kind": kind, "topic": "Prices",
                "text": f"Between {hs['days'][0]} and {hs['days'][-1]}, {len(up)} {label}s got "
                        f"more expensive and {len(down)} got cheaper, of {len(ms)} tracked.",
                "detail": "Change is first observed lowest listing to last; under "
                          f"{100 * STABLE_PRICE_SHARE:.0f}% either way counts as stable.",
            })  # fmt: skip
    return out


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    cols = [c for c in cols if c in df.columns]
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
        for r in df[cols].to_dict("records")
    ]


def payload(bench: pd.DataFrame, value: pd.DataFrame, history: pd.DataFrame) -> dict:
    """Everything the page's charts need, as plain JSON-friendly values."""
    rk = ranked(value)
    hist = history.copy()
    if len(hist):
        hist["date"] = hist["date"].astype(str).str[:10]
    return {
        "models": _records(
            rk,
            [
                "kind",
                "model_key",
                "maker",
                "price_usd",
                "median_price_usd",
                "score",
                "p25_score",
                "p75_score",
                "n_runs",
                "score_per_dollar",
                "confidence",
                "n_listings",
                "n_retailers",
                "collected_date",
                "retailer",
                "listing_url",
                "backend",
            ],
        ),  # fmt: skip
        "history": _records(hist, ["date", "kind", "model_key", "price_usd", "n_listings"]),
        "history_summary": history_summary(history),
        "brackets": DEFAULT_BRACKETS,
        "gates": {
            "percentiles": MIN_FOR_PERCENTILES,
            "history_days": MIN_HISTORY_DAYS,
            "stable_share": STABLE_PRICE_SHARE,
        },
        "kinds": KINDS,
    }


# --- HTML for the parts of the page that do not change with the filters ---------------------


def _h(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))  # fmt: skip


def _card(title: str, body: str, cls: str = "") -> str:
    extra = f" {cls}" if cls else ""
    return f'<div class="vd-card{extra}"><div class="vd-card-title">{_h(title)}</div>{body}</div>'


def overview_html(bench: pd.DataFrame, value: pd.DataFrame, history: pd.DataFrame) -> str:
    """Market overview cards: coverage per kind, best value per kind, price freshness."""
    fun = funnel(bench, value).set_index("kind")
    rk = ranked(value)
    cards = []
    for kind, label in KIND_SHORT.items():
        f = fun.loc[kind]
        pct = 100 * f["ranked"] / max(int(f["price_checked"]), 1)
        cards.append(_card(f"{label} coverage", (
            f'<div class="vd-big">{f["ranked"]} <span>ranked ({pct:.0f}% of checked)</span></div>'
            f'<div class="vd-mini-bar"><span style="width:{min(pct, 100):.1f}%"></span></div>'
            f'<div class="vd-small">{f["benchmarked"]} benchmarked · '
            f'{f["price_checked"]} price-checked · {f["priced"]} priced</div>'
        ), cls=f"vd-card-{kind}"))  # fmt: skip
    for kind, label in KIND_SHORT.items():
        part = rk[rk["kind"] == kind]
        if part.empty:
            none = '<div class="vd-small">No ranked models yet.</div>'
            cards.append(_card(f"Best {label} value", none, cls=f"vd-card-{kind}"))
            continue
        b = part.loc[part["score_per_dollar"].idxmax()]
        body = (
            f'<div class="vd-big">{_h(display_name(b["model_key"]))}</div>'
            f'<div class="vd-small"><strong>{b["score_per_dollar"]:.2f} pts/$</strong> · '
            f"${b['price_usd']:,.2f} · {b['score']:,.0f} pts</div>"
        )
        if b.get("confidence") == "low":
            normal = part[part["confidence"] != "low"]
            alt = ""
            if len(normal):
                n = normal.loc[normal["score_per_dollar"].idxmax()]
                n_name = _h(display_name(n["model_key"]))
                alt = (
                    f" Best with 2+ listings: <strong>{n_name}</strong> "
                    f"({n['score_per_dollar']:.2f} pts/$, ${n['price_usd']:,.0f})."
                )
            body += f'<div class="vd-small vd-warn">Single listing.{alt}</div>'
        cards.append(_card(f"Best {label} value", body, cls=f"vd-card-{kind}"))
    fr = freshness(bench, value, history)
    counts = fr["status"].value_counts().to_dict() if len(fr) else {}
    last = max(history["date"].astype(str).str[:10]) if len(history) else "never"
    fresh_pct = min(100 * counts.get("fresh", 0) / max(len(fr), 1), 100)
    cards.append(_card("Price freshness", (
        f'<div class="vd-big">{_h(last)}</div>'
        f'<div class="vd-mini-bar vd-mini-bar-ok">'
        f'<span style="width:{fresh_pct:.1f}%"></span></div>'
        f'<div class="vd-small">Of {len(fr)} price-checked models: '
        f'<strong>{counts.get("fresh", 0)} fresh</strong>, {counts.get("stale", 0)} stale, '
        f'{counts.get("never priced", 0)} unmatched (≤{config.PRICE_MAX_AGE_DAYS}d window).</div>'
    ), cls="vd-card-fresh"))  # fmt: skip
    return '<div class="vd-cards">' + "".join(cards) + "</div>"


def insights_html(bench: pd.DataFrame, value: pd.DataFrame, history: pd.DataFrame) -> str:
    items = insights(bench, value, history)
    if not items:
        return '<div class="vd-state">No insight clears its minimum sample yet.</div>'
    out = []
    for kind, label in KINDS.items():
        part = [i for i in items if i["kind"] == kind]
        if not part:
            continue
        out.append(f'<h4 class="vd-h4">{_h(label)}</h4>')
        for i in part:
            out.append(
                f'<details class="vd-insight"><summary><span class="vd-tag">{_h(i["topic"])}</span>'
                f"{_h(i['text'])}</summary><p>{_h(i['detail'])}</p></details>"
            )
    hs = history_summary(history)
    if not hs["trend_ok"]:
        out.append(
            f'<p class="vd-note vd-muted">Price-trend insights start once price history spans '
            f"{MIN_HISTORY_DAYS} days. It spans {hs['span_days']} today.</p>"
        )
    return "".join(out)


def coverage_html(bench: pd.DataFrame, value: pd.DataFrame, history: pd.DataFrame) -> str:
    """Funnel per kind as bars sized by count, plus the never-matched models."""
    fun = funnel(bench, value)
    fr = freshness(bench, value, history)
    steps = [
        ("benchmarked", "With a benchmark score"),
        ("price_checked", "Price-checked by the collectors"),
        ("priced", "With a matching fresh price"),
        ("ranked", "Ranked"),
    ]
    out = []
    for _, f in fun.iterrows():
        top = max(int(f["benchmarked"]), 1)
        rows = "".join(
            f'<div class="vd-funnel-row"><span class="vd-funnel-label">{_h(t)}</span>'
            f'<span class="vd-funnel-bar"><span style="width:{100 * f[k] / top:.1f}%">'
            "</span></span>"
            f'<span class="vd-funnel-n">{int(f[k])}</span></div>'
            for k, t in steps
        )
        missing = fr[(fr["kind"] == f["kind"]) & (fr["status"] != "fresh")]
        miss = ""
        if len(missing):
            names = ", ".join(_h(display_name(k)) for k in missing["model_key"])
            miss = (f'<details class="vd-insight"><summary>{len(missing)} price-checked '
                    f'{KIND_SHORT[f["kind"]]}s without a fresh matching price</summary>'
                    f"<p>{names}</p></details>")  # fmt: skip
        head = f'<h4 class="vd-h4">{_h(KINDS[f["kind"]])}</h4>'
        out.append(f'<div class="vd-funnel">{head}{rows}{miss}</div>')
    return '<div class="vd-funnels">' + "".join(out) + "</div>"
