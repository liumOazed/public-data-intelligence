"""Join retail prices to benchmark scores: score per dollar, with the price date on every row."""

from __future__ import annotations

import duckdb
import pandas as pd

from pdi.tech_value import config
from pdi.tech_value.names import model_key

# CPU renders are compared with CPU renders. For GPUs each backend (OptiX, CUDA, HIP, oneAPI)
# is a separate measurement; the table shows the backend with the most runs for that card.
CPU_BACKEND = "CPU"


def load_prices() -> pd.DataFrame:
    files = sorted(config.PRICES_DIR.glob("*.parquet")) if config.PRICES_DIR.exists() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def benchmark_by_model(blender: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """One row per recognised model, from a single Blender version.

    The version used is the one covering the most recognised models. Scores from other
    versions are not blended in: a model without enough runs on that version has no score.
    """
    if blender.empty:
        return pd.DataFrame(), None
    df = blender.copy()
    keys = df["device_name"].map(model_key)
    df["kind"] = [k[0] if k else None for k in keys]
    df["model_key"] = [k[1] if k else None for k in keys]
    df = df[df["model_key"].notna()]
    df = df[(df["kind"] == "gpu") == (df["device_type"] != CPU_BACKEND)]
    if df.empty:
        return pd.DataFrame(), None
    version = df.groupby("blender_version")["model_key"].nunique().idxmax()
    df = df[df["blender_version"] == version]
    # Several device strings can map to one model ("AMD Ryzen 9 7950X 16-Core Processor").
    # Keep the backend and spelling with the most runs rather than averaging across them.
    df = df.sort_values("n_runs", ascending=False).drop_duplicates(["kind", "model_key"])
    cols = ["kind", "model_key", "device_name", "device_type", "blender_version", "n_runs"]
    return df[cols + ["median_score", "p25_score", "p75_score"]], version


def value_table(prices: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    """Latest price per model joined to its benchmark score.

    Price is the cheapest new listing available online on the latest collection date.
    Models with a price and no score stay in the table with an empty score, so coverage is
    visible instead of silently dropped.
    """
    if prices.empty:
        return pd.DataFrame()
    con = duckdb.connect()
    con.register("prices", prices)
    if bench.empty:
        bench = pd.DataFrame(
            columns=[
                "kind",
                "model_key",
                "device_name",
                "device_type",
                "blender_version",
                "n_runs",
                "median_score",
                "p25_score",
                "p75_score",
            ]
        ).astype({"median_score": float, "n_runs": float, "p25_score": float, "p75_score": float})
    con.register("bench", bench)
    return con.execute(
        """
        WITH latest AS (
            SELECT * FROM prices
            WHERE date = (SELECT max(date) FROM prices)
              AND model_key IS NOT NULL AND online_availability
        ),
        cheapest AS (
            SELECT category AS kind, model_key,
                   count(*) AS n_listings,
                   min(sale_price) AS price_usd,
                   arg_min(name, sale_price) AS listing_name,
                   arg_min(url, sale_price) AS listing_url,
                   arg_min(price_update_date, sale_price) AS price_date,
                   arg_min(regular_price, sale_price) AS regular_price_usd,
                   max(date) AS collected_date
            FROM latest GROUP BY ALL
        )
        SELECT c.kind, c.model_key, c.price_usd, c.price_date, c.collected_date,
               c.regular_price_usd, c.n_listings, c.listing_name, c.listing_url,
               b.median_score AS score, b.p25_score, b.p75_score, b.n_runs,
               b.device_type AS backend, b.blender_version,
               b.median_score / c.price_usd AS score_per_dollar
        FROM cheapest c LEFT JOIN bench b USING (kind, model_key)
        ORDER BY score_per_dollar DESC NULLS LAST, c.price_usd
        """
    ).df()


def price_history(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily lowest new, online-available price per model."""
    if prices.empty:
        return pd.DataFrame()
    con = duckdb.connect()
    con.register("prices", prices)
    return con.execute(
        """
        SELECT date, category AS kind, model_key,
               min(sale_price) AS price_usd, count(*) AS n_listings
        FROM prices
        WHERE model_key IS NOT NULL AND online_availability
        GROUP BY ALL ORDER BY model_key, date
        """
    ).df()


def coverage(prices: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """How much of what is on sale could be matched: listings -> models -> benchmark scores."""
    if prices.empty:
        return pd.DataFrame()
    latest = prices[prices["date"] == prices["date"].max()]
    rows = []
    for kind, part in latest.groupby("category"):
        t = table[table["kind"] == kind] if not table.empty else table
        rows.append(
            {
                "kind": kind,
                "listings": int(len(part)),
                "listings_recognised": int(part["model_key"].notna().sum()),
                "models_priced": int(len(t)),
                "models_with_score": int(t["score"].notna().sum()) if len(t) else 0,
            }
        )
    out = pd.DataFrame(rows)
    out["share_with_score"] = out["models_with_score"] / out["models_priced"].where(
        out["models_priced"] > 0
    )
    return out
