"""US average electricity price, the default rate of the efficiency page's cost calculator.

The series is BLS average price data, "Electricity per kilowatt-hour in U.S. city average"
(APU000072610), read through FRED like the macro series. It is published by a federal agency,
so it is public domain. Collected daily with the retail prices; a failed download leaves the
stored series as it was and the page falls back to its last stored month.

    python -m pdi.tech_value.electricity
"""

from __future__ import annotations

import io
import json
import os
from datetime import date

import pandas as pd

from pdi.common.fetch import CollectError, fetch, read_meta
from pdi.tech_value import config

SERIES = "APU000072610"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
KEY_ENV = "FRED_API_KEY"
START = "2015-01-01"
COLUMNS = ["period", "usd_per_kwh", "retrieved_at"]


def parse_csv(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    if SERIES not in raw.columns:
        raise CollectError(f"FRED CSV for {SERIES} has columns {list(raw.columns)}")
    df = pd.DataFrame(
        {
            "period": pd.to_datetime(raw[raw.columns[0]], errors="coerce"),
            # FRED writes "." for a missing month.
            "usd_per_kwh": pd.to_numeric(raw[SERIES], errors="coerce"),
        }
    ).dropna()
    return df.reset_index(drop=True)


def parse_api(payload: dict) -> pd.DataFrame:
    obs = payload.get("observations")
    if obs is None:
        raise CollectError(f"FRED API answer for {SERIES} has no observations")
    df = pd.DataFrame(
        {
            "period": pd.to_datetime([o["date"] for o in obs], errors="coerce"),
            "usd_per_kwh": pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce"),
        }
    ).dropna()
    return df.reset_index(drop=True)


def latest_rate(df: pd.DataFrame) -> dict | None:
    """The newest month's price, or None when nothing has been collected."""
    if df is None or df.empty:
        return None
    row = df.sort_values("period").iloc[-1]
    return {
        "usd_per_kwh": round(float(row["usd_per_kwh"]), 4),
        "period": pd.Timestamp(row["period"]).strftime("%Y-%m"),
        "retrieved_at": row.get("retrieved_at"),
    }


def load() -> pd.DataFrame:
    path = config.ELECTRICITY_TABLE
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)


def collect(today: date | None = None) -> pd.DataFrame:
    today = today or date.today()
    stem = config.RAW_DIR / "electricity" / f"{today.isoformat()}_{SERIES}"
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        params = {"series_id": SERIES, "api_key": key, "file_type": "json",
                  "observation_start": START}  # fmt: skip
        path = fetch(FRED_API, stem.with_suffix(".json"), params=params)
        df = parse_api(json.loads(path.read_text()))
    else:
        path = fetch(FRED_CSV, stem.with_suffix(".csv"), params={"id": SERIES, "cosd": START})
        df = parse_csv(path.read_text())
    if df.empty:
        raise CollectError(f"FRED returned no observations for {SERIES}")
    df["retrieved_at"] = read_meta(path)["retrieved_at"]
    config.ELECTRICITY_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_parquet(config.ELECTRICITY_TABLE, index=False)
    return df


def main() -> None:
    df = collect()
    rate = latest_rate(df)
    print(f"electricity: {len(df)} months, latest {rate['period']} at ${rate['usd_per_kwh']}/kWh")


if __name__ == "__main__":
    main()
