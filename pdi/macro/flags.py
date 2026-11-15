"""Flags: which series moved unusually fast against their own history, and which economy moved
against its neighbours.

Everything here is arithmetic on published values, with no fitted model, so every flag is an
observation. The thresholds live in pdi/macro/config.py and were written before any flag was
computed.

For each series and month:

    move      = 100 x change in log level (level series) or change in value (rates)
    expected  = median move over the previous ten years; for seasonal series, the median move
                in the same calendar month over those years
    surprise  = move - expected
    z         = surprise / (1.4826 x median absolute deviation of past surprises)

A month is flagged when |z| >= FLAG_Z. Against neighbours, the country's z minus the median z
of the other Southeast Asian countries that month is scored the same way against its own
history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdi.common import findings
from pdi.macro import config

MAD_TO_SD = 1.4826


def monthly_values(obs: pd.DataFrame, series_id: str) -> pd.Series:
    """One value per calendar month on a complete monthly index (gaps become NaN)."""
    g = obs.loc[obs["series_id"] == series_id].set_index("period")["value"].sort_index()
    if g.empty:
        return g
    full = pd.date_range(g.index.min(), g.index.max(), freq="MS")
    return g.reindex(full)


def moves(values: pd.Series, transform: str) -> pd.Series:
    if transform == "log":
        return 100 * np.log(values).diff()
    if transform == "diff":
        return values.diff()
    raise ValueError(f"unknown transform {transform!r}")


def _robust_scale(resid: np.ndarray) -> float:
    resid = resid[~np.isnan(resid)]
    if len(resid) < 2:
        return np.nan
    scale = MAD_TO_SD * np.median(np.abs(resid - np.median(resid)))
    if scale <= 1e-12:
        # Series that rarely move (policy rates) have a zero median deviation; fall back to the
        # standard deviation so that a change is scored against how often changes happen.
        scale = float(np.std(resid, ddof=1))
    return scale if scale > 1e-12 else np.nan


def surprises(x: pd.Series, seasonal: str) -> pd.DataFrame:
    """Score every month of a move series against the ten years before it."""
    idx = x.index
    vals = x.to_numpy(dtype=float)
    months = idx.month.to_numpy()
    out = np.full((len(x), 3), np.nan)  # expected, scale, z
    for i in range(len(x)):
        if np.isnan(vals[i]):
            continue
        lo = max(0, i - config.HISTORY_MONTHS)
        w, wm = vals[lo:i], months[lo:i]
        ok = ~np.isnan(w)
        if ok.sum() < config.MIN_HISTORY_MONTHS:
            continue
        w, wm = w[ok], wm[ok]
        if seasonal == "nsa":
            same = w[wm == months[i]]
            if len(same) < config.MIN_SAME_MONTH_YEARS:
                continue
            expected = float(np.median(same))
            med_by_month = {m: np.median(w[wm == m]) for m in np.unique(wm)}
            resid = w - np.array([med_by_month[m] for m in wm])
        else:
            expected = float(np.median(w))
            resid = w - expected
        scale = _robust_scale(resid)
        if np.isnan(scale):
            continue
        out[i] = (expected, scale, (vals[i] - expected) / scale)
    df = pd.DataFrame(out, index=idx, columns=["expected", "scale", "z"])
    df.insert(0, "move", vals)
    df["surprise"] = df["move"] - df["expected"]
    return df


def outcome(surprise: pd.Series, at: pd.Timestamp) -> str:
    """Whether a flagged surprise held over the next FOLLOW_MONTHS months or was given back."""
    pos = surprise.index.get_loc(at)
    after = surprise.iloc[pos + 1 : pos + 1 + config.FOLLOW_MONTHS]
    if len(after) < config.FOLLOW_MONTHS or after.isna().any():
        return "too early"
    s = surprise.iloc[pos]
    given_back = -np.sign(s) * after.sum()
    return "reversed" if given_back > config.REVERSAL_SHARE * abs(s) else "held"


def score_all(obs: pd.DataFrame) -> pd.DataFrame:
    """Every series, every month: move, expected move, surprise and z."""
    cat = config.catalogue()
    frames = []
    for sid in sorted(obs["series_id"].unique()):
        s = cat[sid]
        x = moves(monthly_values(obs, sid), s.transform)
        df = surprises(x, s.seasonal)
        df = df.assign(series_id=sid, country=s.country, indicator=s.indicator)
        frames.append(df.rename_axis("period").reset_index())
    return pd.concat(frames, ignore_index=True)


def neighbour_gaps(scores: pd.DataFrame) -> pd.DataFrame:
    """For each comparable indicator, each country's z minus the median z of its neighbours,
    and that gap scored against its own history."""
    frames = []
    sea = scores[scores["country"].isin(config.SEA)]
    for ind in config.NEIGHBOUR_INDICATORS:
        wide = sea[sea["indicator"] == ind].pivot(index="period", columns="country", values="z")
        if wide.shape[1] < config.MIN_NEIGHBOURS + 1:
            continue
        wide = wide.asfreq("MS")
        for c in wide.columns:
            others = wide.drop(columns=c)
            n = others.notna().sum(axis=1)
            med = others.median(axis=1).where(n >= config.MIN_NEIGHBOURS)
            gap = (wide[c] - med).rename("gap")
            sc = surprises(gap, "sa")
            frames.append(
                pd.DataFrame(
                    {
                        "period": wide.index,
                        "series_id": f"{c}.{ind}",
                        "country": c,
                        "indicator": ind,
                        "own_z": wide[c].to_numpy(),
                        "neighbours_median_z": med.to_numpy(),
                        "n_neighbours": n.to_numpy(),
                        "gap": gap.to_numpy(),
                        "gap_surprise": sc["surprise"].to_numpy(),
                        "z": sc["z"].to_numpy(),
                    }
                )
            )
    cols = ["period", "series_id", "country", "indicator", "own_z", "neighbours_median_z",
            "n_neighbours", "gap", "gap_surprise", "z"]  # fmt: skip
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def _lower(name: str) -> str:
    """Lower-case the first letter only, so "US dollar" and "CPI" keep their capitals."""
    return name[:1].lower() + name[1:]


def _fmt_move(value: float, s: config.Series) -> str:
    if s.transform == "log":
        return f"{abs(value):.2f}%"
    unit = "percentage points" if s.units == "percent" else s.units
    return f"{abs(value):.2f} {unit}"


def own_sentence(row, s: config.Series) -> str:
    verb = s.up_means if row["move"] > 0 else s.down_means
    if abs(row["move"]) < 1e-12:
        verb = "did not move"
    usual = "that calendar month" if s.seasonal == "nsa" else "a month"
    exp = row["expected"]
    exp_txt = f"{'+' if exp >= 0 else '-'}{_fmt_move(exp, s)}"
    move_txt = "" if verb == "did not move" else f" {_fmt_move(row['move'], s)}"
    return (
        f"{config.COUNTRIES[s.country]}: {_lower(s.name)} {verb}{move_txt} in "
        f"{row['period']:%B %Y}, against a usual {exp_txt} for {usual}. That is "
        f"{abs(row['z']):.1f} times its usual spread over the previous ten years."
    )


def neighbour_sentence(row, s: config.Series) -> str:
    way = "more" if row["gap"] > 0 else "less"
    return (
        f"{config.COUNTRIES[s.country]}: {_lower(s.name)} moved against its neighbours in "
        f"{row['period']:%B %Y}. Its surprise was {row['own_z']:+.1f} spreads while the median "
        f"of {int(row['n_neighbours'])} other Southeast Asian economies was "
        f"{row['neighbours_median_z']:+.1f}, so it rose {way} than the region by a gap "
        f"{abs(row['z']):.1f} times that gap's usual spread."
    )


def build(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (scores, neighbour gaps, flags). Flags carry an outcome for the backtest."""
    cat = config.catalogue()
    scores = score_all(obs)
    gaps = neighbour_gaps(scores)
    rows = []

    for sid, g in scores.groupby("series_id"):
        s = cat[sid]
        sur = g.set_index("period")["surprise"]
        hit = g[g["z"].abs() >= config.FLAG_Z]
        for _, r in hit.iterrows():
            rows.append(
                {
                    "series_id": sid,
                    "country": s.country,
                    "indicator": s.indicator,
                    "period": r["period"],
                    "kind": "own history",
                    "move": r["move"],
                    "expected": r["expected"],
                    "z": r["z"],
                    "outcome": outcome(sur, r["period"]),
                    "sentence": own_sentence(r, s),
                }
            )
    for sid, g in gaps.groupby("series_id"):
        s = cat[sid]
        sur = g.set_index("period")["gap_surprise"]
        hit = g[g["z"].abs() >= config.FLAG_Z]
        for _, r in hit.iterrows():
            rows.append(
                {
                    "series_id": sid,
                    "country": s.country,
                    "indicator": s.indicator,
                    "period": r["period"],
                    "kind": "neighbours",
                    "move": np.nan,
                    "expected": np.nan,
                    "z": r["z"],
                    "outcome": outcome(sur, r["period"]),
                    "sentence": neighbour_sentence(r, s),
                }
            )
    flags = pd.DataFrame(rows)
    if not flags.empty:
        flags = flags.sort_values(["period", "kind", "series_id"], ascending=[False, True, True])
        flags["label"] = findings.OBSERVATION
    return scores, gaps, flags.reset_index(drop=True)


FLAG_FINDING = findings.observation(
    "Measured moves compared with each series' own published history. No model is involved, "
    "so a flag says something unusual happened, not why, and not what comes next."
)
