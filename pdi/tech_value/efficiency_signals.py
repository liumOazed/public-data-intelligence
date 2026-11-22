"""Where efficiency is heading: rates of change, not forecasts.

Every signal measures how fast something has been moving, compares the recent pace with the pace
before it, and says which way it points. None extends a line into the future. Each comes with
the numbers it rests on, the method, and how many launches or years it uses, so a reader can
judge how much weight it carries.

* Frontier pace: how fast the best points per rated watt improves per launch year, recent
  years against the years before. For servers, the same on SPECpower's measured results.
* TDP creep: whether rated power rises from one launch to the next *within the same product
  class* (Ryzen 7 X, Core i7 K, GeForce xx70 and so on), so a shift in which chips get
  benchmarked does not pass for creep.
* Vendor gap: whether the gap in median efficiency between the two main makers of each kind
  is widening or closing. Apple is excluded from the whole page by the owner's decision.
* Price of efficiency: what one point per rated watt costs today, cohort by launch cohort.
  Daily prices only began in November 2026, so the over-time version waits for history.
* Turning points: whether the frontier, or rated power, bends in a particular year, by
  comparing a straight line with a line that changes slope once (BIC), and which architectures
  and process nodes arrived that year.

Slopes are Theil-Sen (the median of pairwise slopes), so one odd year cannot swing them, with
a 90% interval.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from pdi.common import findings

WINDOW = 4  # launch years in the "recent" and the "before" windows
SP_WINDOW = 6  # publication years for SPECpower
STEADY_PTS = 3.0  # a change of pace smaller than this, in % per year, is "steady"
MIN_CLASS_YEARS = 3
# A launch year with fewer benchmarked chips than this is left out of the yearly series: the
# current year fills up over months, and one early chip would otherwise stand for the year.
MIN_PER_YEAR = 3
BIC_STRONG, BIC_WEAK = 6.0, 2.0
MIN_HISTORY_DAYS = 56
PAIRS = {"cpu": ("amd", "intel"), "gpu": ("nvidia", "amd")}
VENDOR = {"amd": "AMD", "intel": "Intel", "nvidia": "NVIDIA"}
UNIT = {"cpu": "processor", "gpu": "graphics card"}

SIGNAL_COLUMNS = [
    "signal", "kind", "title", "direction", "sentence", "value", "unit", "recent", "prior",
    "lo", "hi", "n", "window", "method", "label", "label_reason",
]  # fmt: skip
SERIES_COLUMNS = ["signal", "kind", "group", "x", "y", "fit"]


def _pct(slope_log: float) -> float:
    return float(np.expm1(slope_log) * 100)


def theil(x, y, ci: float = 0.90) -> tuple[float, float, float, float]:
    """Theil-Sen slope, intercept and its interval, for y on x."""
    res = theilslopes(np.asarray(y, float), np.asarray(x, float), ci)
    return float(res[0]), float(res[1]), float(res[2]), float(res[3])


def _pace_word(recent: float, prior: float) -> str:
    if recent < prior - STEADY_PTS:
        return "slowing"
    if recent > prior + STEADY_PTS:
        return "speeding up"
    return "steady"


def _signal(**kw) -> dict:
    label = kw.pop("finding", None) or findings.explanation(
        "A fitted slope that describes how fast the past moved. It is not a forecast."
    )
    return {**{c: None for c in SIGNAL_COLUMNS}, **kw, "label": label["label"],
            "label_reason": label["reason"]}  # fmt: skip


def _series(rows: list, signal: str, kind: str, group: str, x, y, fit=None) -> None:
    fit = [None] * len(x) if fit is None else fit
    for a, b, f in zip(x, y, fit, strict=True):
        rows.append({"signal": signal, "kind": kind, "group": group, "x": float(a),
                     "y": None if b is None or not np.isfinite(b) else float(b),
                     "fit": None if f is None else float(f)})  # fmt: skip


# --- Frontier pace --------------------------------------------------------------------------------
def pace(years, best, window: int) -> dict | None:
    """Annual % change of the yearly best over the last `window` years and the `window` before."""
    s = pd.Series(np.asarray(best, float), index=np.asarray(years, int)).dropna().sort_index()
    s = s[s > 0]
    if len(s) < window + 3:
        return None
    recent, prior = s.iloc[-window:], s.iloc[-2 * window : -window]
    r = theil(recent.index, np.log(recent.to_numpy()))
    p = theil(prior.index, np.log(prior.to_numpy()))
    return {"recent": _pct(r[0]), "prior": _pct(p[0]), "lo": _pct(r[2]), "hi": _pct(r[3]),
            "recent_years": (int(recent.index[0]), int(recent.index[-1])),
            "prior_years": (int(prior.index[0]), int(prior.index[-1]))}  # fmt: skip


def _rated(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    part = df[(df["kind"] == kind) & df["watts"].notna()]
    n = part.groupby("year")["model_key"].transform("size")
    return part[n >= MIN_PER_YEAR]


def frontier_pace(df: pd.DataFrame, kind: str, rows: list) -> dict | None:
    part = _rated(df, kind)
    best = part.groupby("year")["score_per_watt"].max()
    p = pace(best.index, best.to_numpy(), WINDOW)
    _series(rows, "frontier_pace", kind, "Best of the year", best.index, best.to_numpy(),
            np.maximum.accumulate(best.to_numpy()))  # fmt: skip
    if p is None:
        return None
    word = _pace_word(p["recent"], p["prior"])
    ry, py = p["recent_years"], p["prior_years"]
    return _signal(
        signal="frontier_pace", kind=kind, title="Frontier pace", direction=word,
        value=p["recent"], unit="% per launch year", recent=p["recent"], prior=p["prior"],
        lo=p["lo"], hi=p["hi"], n=int(len(best)), window=f"{ry[0]}–{ry[1]} vs {py[0]}–{py[1]}",
        sentence=(
            f"The most efficient {UNIT[kind]} of each launch year improved {p['recent']:+.0f}% a "
            f"year over {ry[0]}–{ry[1]}, against {p['prior']:+.0f}% a year over {py[0]}–{py[1]}: "
            f"{word}."
        ),
        method="Theil-Sen slope of log best points per rated watt on launch year, per window.",
    )  # fmt: skip


def specpower_pace(trend: pd.DataFrame, rows: list) -> dict | None:
    if trend is None or trend.empty:
        return None
    t = trend.sort_values("year")
    _series(rows, "specpower_pace", "cpu", "Best measured", t["year"], t["best"])
    p = pace(t["year"], t["best"], SP_WINDOW)
    if p is None:
        return None
    word = _pace_word(p["recent"], p["prior"])
    ry, py = p["recent_years"], p["prior_years"]

    def dbl(pct: float) -> str:
        return (
            f"doubling every {np.log(2) / np.log1p(pct / 100):.1f} years"
            if pct > 0
            else "not rising"
        )

    return _signal(
        signal="specpower_pace", kind="cpu", title="Measured server frontier", direction=word,
        value=p["recent"], unit="% per year", recent=p["recent"], prior=p["prior"], lo=p["lo"],
        hi=p["hi"], n=int(len(t)), window=f"{ry[0]}–{ry[1]} vs {py[0]}–{py[1]}",
        sentence=(
            f"The best measured server result improved {p['recent']:+.0f}% a year in "
            f"{ry[0]}–{ry[1]} ({dbl(p['recent'])}), against {p['prior']:+.0f}% a year in "
            f"{py[0]}–{py[1]} ({dbl(p['prior'])}): {word}."
        ),
        method="Theil-Sen slope of log best ssj_ops per watt on year published (SPECpower).",
    )  # fmt: skip


# --- TDP creep ------------------------------------------------------------------------------------
def product_class(kind: str, key: str) -> str | None:
    """Like-for-like class: 'Ryzen 7 X', 'Core 7 K', 'GeForce xx70 Ti', 'Radeon x800 XT'."""
    if kind == "cpu":
        if m := re.match(r"amd ryzen (\d) \d{4}(\w*)$", key):
            sfx = m.group(2)
            tag = (
                "X3D" if "x3d" in sfx else "X" if sfx.startswith("x") else "G" if "g" in sfx else ""
            )
            return f"Ryzen {m.group(1)} {tag}".strip()
        if m := re.match(r"intel core (?:i|ultra )(\d)[- ]\d{3,5}(\w*)$", key):
            sfx = m.group(2)
            return f"Core {m.group(1)}{' K' if sfx.startswith('k') else ''}"
        if key.startswith("amd threadripper"):
            return "Threadripper Pro" if " pro " in key else "Threadripper"
        return None
    if m := re.match(r"nvidia (?:geforce )?(?:gtx|rtx) (\d{1,2})(\d)0( ti| super)?$", key):
        return f"GeForce xx{m.group(2)}0{(m.group(3) or '').title()}"
    if m := re.match(r"amd (?:radeon )?rx (\d)(\d)00( xt| gre)?$", key):
        return f"Radeon x{m.group(2)}00{(m.group(3) or '').upper()}"
    return None


def tdp_creep(df: pd.DataFrame, kind: str, rows: list) -> dict | None:
    part = df[(df["kind"] == kind) & df["watts"].notna()].copy()
    part["cls"] = [product_class(kind, k) for k in part["model_key"]]
    part = part[part["cls"].notna()]
    slopes = []
    for cls, g in part.groupby("cls"):
        by_year = g.groupby("year")["watts"].median()
        if len(by_year) < MIN_CLASS_YEARS:
            continue
        s = theil(by_year.index, np.log(by_year.to_numpy()))
        slopes.append((cls, s[0], len(by_year)))
        _series(rows, "tdp_creep", kind, cls, by_year.index, by_year.to_numpy())
    if len(slopes) < 3:
        return None
    pct = np.array([_pct(s) for _, s, _ in slopes])
    med = float(np.median(pct))
    share_up = float(np.mean(pct > 0) * 100)
    word = (
        "climbing"
        if med > 1 and share_up >= 60
        else "falling"
        if med < -1 and share_up <= 40
        else "flat"
    )
    top = sorted(zip(pct, [c for c, _, _ in slopes], strict=True), reverse=True)
    return _signal(
        signal="tdp_creep", kind=kind, title="TDP creep", direction=word, value=med,
        unit="% per launch year", lo=float(np.quantile(pct, 0.25)),
        hi=float(np.quantile(pct, 0.75)),
        n=len(slopes), window="all launch years",
        sentence=(
            f"Within the same product class, rated power changed a median {med:+.1f}% per launch "
            f"year across {len(slopes)} classes, and rose in {share_up:.0f}% of them: {word}. "
            f"Steepest: {top[0][1]} ({top[0][0]:+.1f}% a year)."
        ),
        method="Theil-Sen slope of log median rated power on launch year within each product "
        "class with 3 or more launch years; median across classes, with the middle half.",
    )  # fmt: skip


# --- Vendor gap -----------------------------------------------------------------------------------
def vendor_gap(df: pd.DataFrame, kind: str, rows: list) -> dict | None:
    a, b = PAIRS[kind]
    part = df[(df["kind"] == kind) & df["watts"].notna() & df["vendor"].isin([a, b])]
    g = part.groupby(["year", "vendor"])["log_ppw"].agg(["median", "size"]).reset_index()
    g = g[g["size"] >= 2]
    wide = g.pivot(index="year", columns="vendor", values="median").dropna()
    if len(wide) < 5 or a not in wide or b not in wide:
        return None
    gap = (wide[a] - wide[b]).sort_index()
    for v in (a, b):
        _series(rows, "vendor_gap", kind, VENDOR[v], wide.index, np.exp(wide[v].to_numpy()))
    recent = gap.iloc[-2 * WINDOW :]
    lead = a if gap.iloc[-1] > 0 else b
    sign = 1 if lead == a else -1
    s, _, lo, hi = theil(recent.index, sign * recent.to_numpy())
    pts = _pct(s)
    if lo > 0:
        word = "widening"
    elif hi < 0:
        word = "closing"
    else:
        word = "no clear change"
    other = b if lead == a else a
    now = float(np.expm1(abs(gap.iloc[-1])) * 100)
    return _signal(
        signal="vendor_gap", kind=kind, title="Vendor gap", direction=word, value=now,
        unit=f"% {VENDOR[lead]} lead", recent=pts, lo=_pct(lo), hi=_pct(hi), n=int(len(recent)),
        window=f"{int(recent.index[0])}–{int(recent.index[-1])}",
        sentence=(
            f"In {int(gap.index[-1])} launches, {VENDOR[lead]}'s median {UNIT[kind]} did "
            f"{now:.0f}% "
            f"more points per rated watt than {VENDOR[other]}'s. Over "
            f"{int(recent.index[0])}–{int(recent.index[-1])} the lead moved {pts:+.1f} "
            f"percentage points a year: {word}."
            + (" Intel's rating is base power, which its chips boost far above, so the "
               "label flatters Intel." if kind == "cpu" else "")
        ),
        method="Yearly median log points per rated watt per maker (years with 2 or more chips "
        "from each); Theil-Sen slope of the gap, oriented so a positive slope widens the "
        "current leader's lead. 'Widening' or 'closing' only when the 90% interval excludes zero.",
    )  # fmt: skip


# --- Price of efficiency --------------------------------------------------------------------------
def price_of_efficiency(df: pd.DataFrame, kind: str, history: pd.DataFrame | None,
                        rows: list) -> list[dict]:  # fmt: skip
    out = []
    part = df[(df["kind"] == kind) & df["watts"].notna() & (df["price_usd"] > 0)].copy()
    if len(part):
        part["usd_per_ppw"] = part["price_usd"] / part["score_per_watt"]
        cohort = part.groupby("year")["usd_per_ppw"].min()
        cohort = cohort[cohort.index > cohort.index.max() - 2 * WINDOW]
        _series(rows, "price_cohort", kind, "Cheapest per point per watt", cohort.index,
                cohort.to_numpy())  # fmt: skip
        if len(cohort) >= 4:
            s, _, lo, hi = theil(cohort.index, np.log(cohort.to_numpy()))
            word = "falling" if hi < 0 else "rising" if lo > 0 else "no clear change"
            out.append(_signal(
                signal="price_cohort", kind=kind, title="Price of efficiency", direction=word,
                value=_pct(s), unit="% per launch year", lo=_pct(lo), hi=_pct(hi),
                n=int(len(cohort)), window=f"{int(cohort.index[0])}–{int(cohort.index[-1])}",
                sentence=(
                    f"At today's prices, one point per rated watt costs {_pct(s):+.0f}% more with "
                    f"each newer launch year ({int(cohort.index[0])}–{int(cohort.index[-1])}): "
                    f"{word}." if s >= 0 else
                    f"At today's prices, one point per rated watt costs {abs(_pct(s)):.0f}% less "
                    f"with each newer launch year ({int(cohort.index[0])}–"
                    f"{int(cohort.index[-1])}): {word}."
                ),
                method="Cheapest current price divided by points per rated watt, per launch "
                "year; Theil-Sen slope of its log over the last 8 launch years. Today's prices, "
                "not launch prices.",
            ))  # fmt: skip
    if history is not None and len(history):
        h = history[history["kind"] == kind].merge(
            df[df["watts"].notna()][["kind", "model_key", "score_per_watt"]],
            on=["kind", "model_key"],
        )
        daily = (h["price_usd"] / h["score_per_watt"]).groupby(h["date"]).min().sort_index()
        days = (
            (pd.to_datetime(daily.index.max()) - pd.to_datetime(daily.index.min())).days
            if len(daily)
            else 0
        )
        if len(daily):
            x = (pd.to_datetime(daily.index) - pd.to_datetime(daily.index.min())).days
            _series(rows, "price_daily", kind, "Cheapest per point per watt", x, daily.to_numpy())
        if days >= MIN_HISTORY_DAYS:
            s, _, lo, hi = theil(x, np.log(daily.to_numpy()))
            per30 = _pct(30 * s)
            word = "falling" if hi < 0 else "rising" if lo > 0 else "no clear change"
            out.append(_signal(
                signal="price_daily", kind=kind, title="Price of efficiency over time",
                direction=word, value=per30, unit="% per 30 days", lo=_pct(30 * lo),
                hi=_pct(30 * hi), n=int(len(daily)),
                window=f"{daily.index.min()} – {daily.index.max()}",
                sentence=f"The cheapest point per rated watt moved {per30:+.1f}% per 30 days: "
                f"{word}.",
                method="Daily lowest price divided by points per rated watt, cheapest model each "
                "day; Theil-Sen slope of its log.",
            ))  # fmt: skip
        else:
            out.append(_signal(
                signal="price_daily", kind=kind, title="Price of efficiency over time",
                direction="collecting", n=int(len(daily)), window=f"{days} days so far",
                sentence=f"Daily prices cover {days} days. The over-time trend appears after "
                f"{MIN_HISTORY_DAYS} days.",
                method="Waits for 8 weeks of daily prices.",
                finding=findings.observation("No trend computed yet."),
            ))  # fmt: skip
    return out


# --- Turning points -------------------------------------------------------------------------------
def turning_point(x, y, min_side: int = 3) -> dict | None:
    """Best single change of slope in y over x, compared with a straight line by BIC.

    The bent line is continuous: y = a + b*x + c*max(0, x - t), with t at one of the observed
    x. ``delta_bic`` is the straight line's BIC minus the bent line's (the break year counts as a
    parameter), so positive favours a bend.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 2 * min_side + 1:
        return None

    def sse(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        return float(r @ r), beta

    ones = np.ones(n)
    s1, b1 = sse(np.column_stack([ones, x]))
    best = None
    for t in np.unique(x)[min_side - 1 : -min_side]:
        if (x <= t).sum() < min_side or (x > t).sum() < min_side:
            continue
        s2, b2 = sse(np.column_stack([ones, x, np.maximum(0, x - t)]))
        if best is None or s2 < best[0]:
            best = (s2, t, b2)
    if best is None:
        return None
    s2, t, b2 = best
    eps = 1e-12
    bic1 = n * np.log(s1 / n + eps) + 2 * np.log(n)
    bic2 = n * np.log(s2 / n + eps) + 4 * np.log(n)
    d = float(bic1 - bic2)
    strength = "strong" if d >= BIC_STRONG else "possible" if d >= BIC_WEAK else "none"
    return {"year": int(t), "delta_bic": d, "strength": strength,
            "before": _pct(b2[1]), "after": _pct(b2[1] + b2[2]), "line": _pct(b1[1]),
            "fit": (b2[0] + b2[1] * x + b2[2] * np.maximum(0, x - t)).tolist(),
            "x": x.tolist()}  # fmt: skip


def _arrivals(df: pd.DataFrame, kind: str, year: int) -> str:
    part = df[df["kind"] == kind]
    first = part.groupby("arch")["year"].min()
    archs = sorted(first[(first >= year) & (first <= year + 1)].index)
    nodes = {}
    for a in archs:
        nodes.setdefault(part.loc[part["arch"] == a, "node"].mode().iloc[0], []).append(a)
    return "; ".join(f"{', '.join(v)} on {k}" for k, v in nodes.items())


def turning_points(df: pd.DataFrame, kind: str, rows: list, sp_trend=None) -> list[dict]:
    out = []
    part = _rated(df, kind)
    series = [
        ("turn_frontier", "Turning point: frontier", part.groupby("year")["score_per_watt"].max(),
         "best points per rated watt"),
        ("turn_power", "Turning point: rated power", part.groupby("year")["watts"].median(),
         "median rated power"),
    ]  # fmt: skip
    if kind == "cpu" and sp_trend is not None and len(sp_trend):
        t = sp_trend.set_index("year")["best"].sort_index()
        series.append(("turn_specpower", "Turning point: measured servers", t,
                       "best measured ssj_ops per watt"))  # fmt: skip
    for sig, title, s, what in series:
        s = s[s > 0]
        tp = turning_point(s.index, np.log(s.to_numpy()))
        if tp is None:
            continue
        _series(rows, sig, kind, what, tp["x"], np.exp(np.log(s.to_numpy())), np.exp(tp["fit"]))
        if tp["strength"] == "none":
            sentence = (f"No clear bend in {what}: one straight line fits as well "
                        f"({tp['line']:+.0f}% a year).")  # fmt: skip
            direction = "no bend"
        else:
            arrived = _arrivals(df, kind, tp["year"]) if sig != "turn_specpower" else ""
            sentence = (
                f"{tp['strength'].capitalize()} bend in {what} at {tp['year']}: "
                f"{tp['before']:+.0f}% a year before, {tp['after']:+.0f}% after."
                + (f" Arriving then: {arrived}." if arrived else "")
            )
            direction = "bends up" if tp["after"] > tp["before"] else "bends down"
        out.append(_signal(
            signal=sig, kind=kind, title=title, direction=direction, value=float(tp["year"]),
            unit="year", recent=tp["after"], prior=tp["before"], n=int(len(s)),
            window=f"{int(s.index.min())}–{int(s.index.max())}", sentence=sentence,
            method="Straight line against a line with one change of slope, by BIC (bend counted "
            f"as a parameter): 'strong' at a BIC gain of {BIC_STRONG:g} or more, 'possible' at "
            f"{BIC_WEAK:g}. Gain here: {tp['delta_bic']:.1f}.",
        ))  # fmt: skip
    return out


def build(df: pd.DataFrame, sp_trend: pd.DataFrame | None = None,
          history: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    signals, rows = [], []
    if df.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS), pd.DataFrame(columns=SERIES_COLUMNS)
    for kind in ("gpu", "cpu"):
        signals.append(frontier_pace(df, kind, rows))
        if kind == "cpu":
            signals.append(specpower_pace(sp_trend, rows))
        signals.append(tdp_creep(df, kind, rows))
        signals.append(vendor_gap(df, kind, rows))
        signals += price_of_efficiency(df, kind, history, rows)
        signals += turning_points(df, kind, rows, sp_trend)
    sig = pd.DataFrame([s for s in signals if s], columns=SIGNAL_COLUMNS)
    return sig, pd.DataFrame(rows, columns=SERIES_COLUMNS)
