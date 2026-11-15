"""Turn stored electricity demand into the tables the Energy Demand page reads.

Everything here is arithmetic on published values (an observation). Nothing is fitted and
nothing is forecast.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pdi.common import findings
from pdi.common.export import write_table
from pdi.energy import config
from pdi.macro import flags

M = config.MODULE
SOURCES = [config.SOURCE]
LIMITS = [
    "One source for every country, so the definition is the same, but each country's figures "
    "reach the source from a different national publisher and with a different delay.",
    "Indonesia is not covered: the source has no monthly demand for it.",
    "Southeast Asian histories are short (they start between 2009 and 2019), so 'unusual' "
    "there is judged against five to seven earlier years of the same calendar month, not ten.",
    "Recent months are revised by the source. Values are shown as last published; every value "
    "is also kept as first seen.",
    "Demand here is electricity generated plus net imports. It includes losses on the grid "
    "and is not the same as electricity sold to customers.",
]
FINDING = findings.observation(
    "Arithmetic on published monthly figures: demand per day, its change on a year earlier, "
    "and how far the latest monthly move is from what that calendar month usually brings. "
    "No model is fitted."
)


def per_country(g: pd.DataFrame) -> pd.DataFrame:
    """One country's months on a complete monthly index, with the derived columns."""
    s = g.set_index("period")["demand_twh"].sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
    # Per day, so that a 28-day February is not read as a fall in demand.
    per_day = 1000 * s / s.index.days_in_month  # GWh per day
    base = per_day[per_day.index.year == config.INDEX_BASE_YEAR]
    base_mean = base.mean() if base.notna().sum() == 12 else np.nan
    scored = flags.surprises(100 * np.log(per_day).diff(), "nsa")
    year = s.rolling(12, min_periods=12).sum()
    return pd.DataFrame(
        {
            "period": s.index,
            "demand_twh": s.to_numpy(),
            "gwh_per_day": per_day.to_numpy(),
            "yoy_pct": (100 * (per_day / per_day.shift(12) - 1)).to_numpy(),
            "year_yoy_pct": (100 * (year / year.shift(12) - 1)).to_numpy(),
            "index": (100 * per_day / base_mean).to_numpy(),
            "move": scored["move"].to_numpy(),
            "expected_move": scored["expected"].to_numpy(),
            "z": scored["z"].to_numpy(),
        }
    )


def build(demand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for country, g in demand.groupby("country"):
        frames.append(per_country(g).assign(country=country, name=config.COUNTRIES[country]))
    history = pd.concat(frames, ignore_index=True)
    have = history.dropna(subset=["demand_twh"])
    latest = have.sort_values("period").groupby("country").tail(1).copy()
    first = have.groupby("country")["period"].min()
    latest["first_month"] = latest["country"].map(first)
    latest["months"] = latest["country"].map(have.groupby("country").size())
    latest["flagged"] = latest["z"].abs() >= config.FLAG_Z
    latest = latest.rename(columns={"period": "latest_month"})[
        ["country", "name", "latest_month", "demand_twh", "gwh_per_day", "yoy_pct",
         "year_yoy_pct", "move", "expected_move", "z", "flagged", "first_month", "months"]
    ].sort_values("year_yoy_pct", ascending=False, ignore_index=True)  # fmt: skip
    history = history[
        ["country", "name", "period", "demand_twh", "gwh_per_day", "yoy_pct", "year_yoy_pct",
         "index", "z"]
    ]  # fmt: skip
    return latest, history


def sentence(row) -> str:
    direction = "up" if row["yoy_pct"] >= 0 else "down"
    text = (
        f"{row['name']}: {row['demand_twh']:.1f} TWh in {row['latest_month']:%B %Y}, "
        f"{direction} {abs(row['yoy_pct']):.1f}% per day on a year earlier"
    )
    if pd.notna(row["z"]) and abs(row["z"]) >= config.FLAG_Z:
        text += (
            f"; the move from the month before was {abs(row['z']):.1f} spreads from what that "
            "calendar month usually brings"
        )
    return text + "."


def main() -> None:
    retrieved = json.loads(config.RETRIEVED.read_text()) if config.RETRIEVED.exists() else {}
    if not config.DEMAND.exists():
        reason = "Electricity demand has not been collected yet."
        for table in ("demand_latest", "demand_history"):
            write_table(M, table, pd.DataFrame(), sources=SOURCES, description=reason,
                        retrieved_at=retrieved, extra={"status": reason})  # fmt: skip
        print(reason)
        return
    latest, history = build(pd.read_parquet(config.DEMAND))
    latest["sentence"] = [sentence(r) for _, r in latest.iterrows()]
    extra = {
        "finding": FINDING,
        "flag_z": config.FLAG_Z,
        "index_base_year": config.INDEX_BASE_YEAR,
        "n_countries": int(len(latest)),
        "not_covered": [config.COUNTRIES[c] for c in config.ISO3 if c not in set(latest.country)],
    }
    write_table(
        M, "demand_latest", latest, sources=SOURCES, retrieved_at=retrieved, limitations=LIMITS,
        description="Latest month of electricity demand for each economy, with its change on a "
        "year earlier and how unusual the latest monthly move was.", extra=extra,
    )  # fmt: skip
    write_table(
        M, "demand_history", history, sources=SOURCES, retrieved_at=retrieved,
        limitations=LIMITS, description="Monthly electricity demand for each economy: TWh, GWh "
        f"per day, change on a year earlier, and an index with {config.INDEX_BASE_YEAR} = 100.",
        extra=extra,
    )  # fmt: skip
    print(f"exported demand for {len(latest)} economies, {len(history):,} months")


if __name__ == "__main__":
    main()
