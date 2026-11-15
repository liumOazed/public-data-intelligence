"""Share-shift arithmetic: each partner's share of US imports, by chapter and in total."""

from __future__ import annotations

import pandas as pd

from pdi.trade import config

WORLD = "WLD"


def twelve_month(imports: pd.DataFrame) -> pd.DataFrame:
    """Trailing twelve-month imports for every partner and chapter, plus chapter "ALL".

    A chapter a partner did not ship in a month is absent from the source; it is a zero here.
    """
    allc = imports.groupby(["partner", "period"], as_index=False)["value"].sum()
    allc["chapter"] = "ALL"
    df = pd.concat([imports, allc], ignore_index=True)
    months = pd.date_range(df["period"].min(), df["period"].max(), freq="MS")
    wide = df.pivot_table(
        index="period", columns=["partner", "chapter"], values="value", aggfunc="sum"
    )
    every = pd.MultiIndex.from_product(
        [sorted(df["partner"].unique()), sorted(df["chapter"].unique())],
        names=["partner", "chapter"],
    )
    wide = wide.reindex(index=months, columns=every).fillna(0.0)
    rolled = wide.rolling(12, min_periods=12).sum().dropna(how="all")
    out = rolled.stack(["partner", "chapter"], future_stack=True).rename("usd_12m").reset_index()
    return out.rename(columns={"level_0": "period"})


def with_shares(rolled: pd.DataFrame) -> pd.DataFrame:
    world = rolled[rolled["partner"] == WORLD].rename(columns={"usd_12m": "world_12m"})
    df = rolled[rolled["partner"] != WORLD].merge(
        world[["period", "chapter", "world_12m"]], on=["period", "chapter"], how="left"
    )
    df["share_pct"] = 100 * df["usd_12m"] / df["world_12m"].where(df["world_12m"] > 0)
    return df


def shifts(shares: pd.DataFrame) -> pd.DataFrame:
    """Latest twelve months against the base year and against a year earlier."""
    latest = shares["period"].max()
    base = pd.Timestamp(f"{config.BASE_YEAR}-12-01")
    keys = ["partner", "chapter"]

    def at(when, suffix):
        cols = {"share_pct": f"share_{suffix}", "usd_12m": f"usd_{suffix}"}
        return shares[shares["period"] == when][[*keys, *cols]].rename(columns=cols)

    now = shares[shares["period"] == latest][[*keys, "usd_12m", "world_12m", "share_pct"]]
    out = now.merge(at(base, "base"), on=keys, how="left").merge(
        at(latest - pd.DateOffset(years=1), "year_ago"), on=keys, how="left"
    )
    out["shift_since_base"] = out["share_pct"] - out["share_base"]
    out["shift_on_year"] = out["share_pct"] - out["share_year_ago"]
    out["period"] = latest
    return out


def china_and_sea(shift: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """One row per product chapter: China's shift since the base year beside each SEA economy's."""
    goods = shift[~shift["chapter"].isin((*config.SPECIAL_CHAPTERS, "ALL"))]
    wide = goods.pivot(index="chapter", columns="partner", values="shift_since_base")
    world = goods.groupby("chapter")["world_12m"].first()
    out = pd.DataFrame({"chapter": wide.index, "world_12m": world.reindex(wide.index).to_numpy()})
    out["name"] = out["chapter"].map(names).fillna("")
    for p in ("CN", *config.SEA):
        out[p] = wide[p].to_numpy() if p in wide else float("nan")
    out["sea_total"] = out[list(config.SEA)].sum(axis=1)
    out["largest_gainer"] = out[list(config.SEA)].idxmax(axis=1)
    out = out[out["world_12m"] >= config.MIN_CHAPTER_USD]
    return out.sort_values("CN", ignore_index=True)
