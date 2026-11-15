"""Collect step for the Macro Monitor: one function per source, each saving the raw response
untouched and returning a tidy table of (series_id, period, value).

Sources, all free and with reuse terms recorded in sources/registry.yaml:

* FRED (US). With a FRED_API_KEY secret the official API is used, which also lets the
  collector read each series' notes and stop on a copyrighted series. Without one, FRED's
  public CSV download is used for the same series.
* BIS (Southeast Asia). Policy rates, consumer prices and exchange rates on one definition
  for every country, so countries can be compared.
* OpenDOSM (Malaysia). Industrial output, trade, unemployment and producer prices.
* IMF (Southeast Asia). Exports, imports and reserves for all six economies on one
  definition, and Vietnam's consumer prices.
* Indeed Hiring Lab (US). Daily job postings index, averaged to months.

Run as `python -m pdi.macro.collect`.
"""

from __future__ import annotations

import io
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

from pdi.common.fetch import CollectError, fetch, read_meta
from pdi.macro import config

TIDY = ["series_id", "period", "value"]


def _month(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values), errors="coerce").dt.to_period("M").dt.to_timestamp()


def _monthly_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Daily or weekly rows -> one row per month, the mean of that month's values.

    The current month is dropped while it is incomplete, so a partial month never looks like a
    move.
    """
    out = (
        df.assign(period=_month(df["period"]).values)
        .groupby(["series_id", "period"], as_index=False)["value"]
        .mean()
    )
    this_month = pd.Timestamp(date.today()).to_period("M").to_timestamp()
    return out[out["period"] < this_month]


# --- FRED --------------------------------------------------------------------------------------


def parse_fred_csv(text: str, series: config.Series) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    date_col = raw.columns[0]
    if series.source_code not in raw.columns:
        raise CollectError(f"FRED CSV for {series.source_code} has columns {list(raw.columns)}")
    df = pd.DataFrame(
        {
            "series_id": series.id,
            "period": pd.to_datetime(raw[date_col], errors="coerce"),
            # FRED writes "." (or nothing) for a missing value.
            "value": pd.to_numeric(raw[series.source_code], errors="coerce"),
        }
    ).dropna()
    return _monthly_mean(df) if series.daily else df.assign(period=_month(df["period"]).values)


def parse_fred_api(payload: dict, series: config.Series) -> pd.DataFrame:
    obs = payload.get("observations")
    if obs is None:
        raise CollectError(f"FRED API answer for {series.source_code} has no observations")
    df = pd.DataFrame(
        {
            "series_id": series.id,
            "period": _month([o["date"] for o in obs]).values,
            "value": pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce"),
        }
    ).dropna()
    return df


def check_fred_notes(info: dict, series: config.Series) -> None:
    """Stop on a series whose notes say it is under someone else's copyright."""
    seriess = info.get("seriess") or []
    if not seriess:
        raise CollectError(f"FRED series {series.source_code} not found")
    notes = str(seriess[0].get("notes", ""))
    if "copyright" in notes.lower():
        raise CollectError(
            f"FRED series {series.source_code} is marked as copyrighted in its notes; it may not "
            "be republished without the owner's permission. Remove it from the catalogue."
        )


def collect_fred(raw_dir: Path, today: date) -> tuple[pd.DataFrame, str | None]:
    key = os.environ.get(config.FRED_KEY_ENV, "").strip()
    frames, retrieved = [], None
    for s in config.US_SERIES:
        stem = raw_dir / "fred" / f"{today.isoformat()}_{s.source_code}"
        if key:
            info_path = fetch(
                f"{config.FRED_API}/series",
                Path(f"{stem}_series.json"),
                params={"series_id": s.source_code, "api_key": key, "file_type": "json"},
            )
            check_fred_notes(json.loads(info_path.read_text()), s)
            params = {
                "series_id": s.source_code,
                "api_key": key,
                "file_type": "json",
                "observation_start": f"{config.HISTORY_START}-01",
            }
            if s.daily:
                params |= {"frequency": "m", "aggregation_method": "avg"}
            path = fetch(
                f"{config.FRED_API}/series/observations", Path(f"{stem}.json"), params=params
            )
            df = parse_fred_api(json.loads(path.read_text()), s)
            if s.daily:
                df = _monthly_mean(df)  # drops the incomplete current month
        else:
            path = fetch(
                config.FRED_CSV,
                Path(f"{stem}.csv"),
                params={"id": s.source_code, "cosd": f"{config.HISTORY_START}-01"},
            )
            df = parse_fred_csv(path.read_text(), s)
        retrieved = read_meta(path)["retrieved_at"]
        frames.append(df)
    via = "API" if key else "public CSV download (no FRED_API_KEY set)"
    print(f"FRED: {len(frames)} series via the {via}")
    return pd.concat(frames, ignore_index=True), retrieved


# --- BIS ---------------------------------------------------------------------------------------


def bis_key(template: str, countries) -> str:
    return template.replace("{c}", "+".join(countries))


def parse_bis_csv(text: str, indicator: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    need = {"REF_AREA", "TIME_PERIOD", "OBS_VALUE"}
    if not need <= set(raw.columns):
        raise CollectError(f"BIS {indicator} answer has columns {list(raw.columns)}")
    if "COLLECTION" in raw.columns:
        # Exchange rates come as period averages ("A") or end of period ("E"). Averages are
        # used where the BIS has them; Malaysia's table only has end of period.
        has_avg = raw.groupby("REF_AREA")["COLLECTION"].transform(lambda c: (c == "A").any())
        raw = raw[(raw["COLLECTION"] == "A") | ~has_avg]
    raw = raw[raw["REF_AREA"].isin(config.SEA)]
    df = pd.DataFrame(
        {
            "series_id": raw["REF_AREA"] + "." + indicator,
            "period": _month(raw["TIME_PERIOD"]).values,
            "value": pd.to_numeric(raw["OBS_VALUE"], errors="coerce").values,
        }
    ).dropna()
    if df.duplicated(["series_id", "period"]).any():
        raise CollectError(f"BIS {indicator} answer has more than one value per country and month")
    return df


def collect_bis(raw_dir: Path, today: date) -> tuple[pd.DataFrame, str | None]:
    frames, retrieved = [], None
    for ind, (flow, template, *_rest) in config.BIS_INDICATORS.items():
        key = bis_key(template, config.SEA)
        path = fetch(
            f"{config.BIS_API}/{flow}/{key}/all",
            raw_dir / "bis" / f"{today.isoformat()}_{flow}.csv",
            params={"format": "csv", "detail": "dataonly", "startPeriod": config.HISTORY_START},
        )
        retrieved = read_meta(path)["retrieved_at"]
        df = parse_bis_csv(path.read_text(), ind)
        ids = sorted(df["series_id"].unique())
        print(f"BIS {flow}: {len(ids)} countries ({ids})")
        frames.append(df)
    return pd.concat(frames, ignore_index=True), retrieved


# --- OpenDOSM ----------------------------------------------------------------------------------


def parse_dosm(rows: list[dict], spec: tuple) -> pd.DataFrame:
    ds, filters, col, ind, *_ = spec
    raw = pd.DataFrame(rows)
    if raw.empty or "date" not in raw.columns or col not in raw.columns:
        raise CollectError(f"OpenDOSM {ds} answer has columns {list(raw.columns)}")
    for k, v in filters.items():
        raw = raw[raw[k] == v]
    df = pd.DataFrame(
        {
            "series_id": f"MY.{ind}",
            "period": _month(raw["date"]).values,
            "value": pd.to_numeric(raw[col], errors="coerce").values,
        }
    ).dropna()
    if df.duplicated(["period"]).any():
        raise CollectError(f"OpenDOSM {ds}:{col} has more than one value per month")
    return df


def collect_dosm(raw_dir: Path, today: date) -> tuple[pd.DataFrame, str | None]:
    frames, retrieved, paths = [], None, {}
    for spec in config.DOSM_SERIES:
        ds = spec[0]
        if ds not in paths:
            paths[ds] = fetch(
                config.DOSM_API,
                raw_dir / "opendosm" / f"{today.isoformat()}_{ds}.json",
                params={"id": ds, "limit": 100000},
            )
            retrieved = read_meta(paths[ds])["retrieved_at"]
        frames.append(parse_dosm(json.loads(paths[ds].read_text()), spec))
    print(f"OpenDOSM: {len(frames)} series")
    return pd.concat(frames, ignore_index=True), retrieved


# --- IMF ---------------------------------------------------------------------------------------


def parse_imf_csv(text: str, indicator: str) -> pd.DataFrame:
    """SDMX-CSV from the IMF data API -> tidy rows for one indicator.

    Periods come as "2026-M06". OBS_VALUE is in units (dollars, index points); the SCALE column
    only says how the IMF displays the number, so it is not applied.
    """
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    need = {"COUNTRY", "TIME_PERIOD", "OBS_VALUE"}
    if not need <= set(raw.columns):
        raise CollectError(f"IMF {indicator} answer has columns {list(raw.columns)[:12]}")
    back = {v: k for k, v in config.ISO3.items()}
    raw = raw[raw["COUNTRY"].isin(back)]
    df = pd.DataFrame(
        {
            "series_id": raw["COUNTRY"].map(back) + "." + indicator,
            "period": _month(raw["TIME_PERIOD"].str.replace("-M", "-", regex=False)).values,
            "value": pd.to_numeric(raw["OBS_VALUE"], errors="coerce").values,
        }
    ).dropna()
    if df.empty:
        raise CollectError(f"IMF {indicator} answer has no usable rows")
    if df.duplicated(["series_id", "period"]).any():
        raise CollectError(f"IMF {indicator} answer has more than one value per country and month")
    return df


def collect_imf(raw_dir: Path, today: date) -> tuple[pd.DataFrame, str | None]:
    frames, retrieved = [], None
    for ind, (flow, template, countries, *_rest) in config.IMF_INDICATORS.items():
        key = template.replace("{c}", "+".join(config.ISO3[c] for c in countries))
        path = fetch(
            f"{config.IMF_API}/IMF.STA,{flow}/{key}",
            raw_dir / "imf" / f"{today.isoformat()}_{flow}_{ind}.csv",
            params={"startPeriod": config.IMF_HISTORY_START},
            headers={"Accept": config.IMF_CSV_ACCEPT},
        )
        retrieved = read_meta(path)["retrieved_at"]
        df = parse_imf_csv(path.read_text(), ind)
        missing = sorted(set(countries) - {s.split(".")[0] for s in df["series_id"]})
        if missing:
            raise CollectError(f"IMF {flow} {ind}: no data for {missing}")
        print(f"IMF {flow} {ind}: {df['series_id'].nunique()} countries")
        frames.append(df)
    return pd.concat(frames, ignore_index=True), retrieved


# --- Indeed ------------------------------------------------------------------------------------


def parse_indeed(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    need = {"date", "indeed_job_postings_index_SA", "variable"}
    if not need <= set(raw.columns):
        raise CollectError(f"Indeed file has columns {list(raw.columns)}")
    raw = raw[raw["variable"].str.strip() == "total postings"]
    df = pd.DataFrame(
        {
            "series_id": "US.job_postings",
            "period": pd.to_datetime(raw["date"], errors="coerce").values,
            "value": pd.to_numeric(raw["indeed_job_postings_index_SA"], errors="coerce").values,
        }
    ).dropna()
    return _monthly_mean(df)


def collect_indeed(raw_dir: Path, today: date) -> tuple[pd.DataFrame, str | None]:
    path = fetch(config.INDEED_URL, raw_dir / "indeed" / f"{today.isoformat()}_US.csv")
    return parse_indeed(path.read_text()), read_meta(path)["retrieved_at"]


COLLECTORS = {
    "fred": collect_fred,
    "bis": collect_bis,
    "opendosm": collect_dosm,
    "imf": collect_imf,
    "indeed_postings": collect_indeed,
}


def collect_all(today: date | None = None, raw_dir: Path | None = None):
    today = today or date.today()
    raw_dir = raw_dir or config.RAW_DIR
    frames, retrieved = [], {}
    for source, fn in COLLECTORS.items():
        df, when = fn(raw_dir, today)
        frames.append(df)
        retrieved[source] = when
    return pd.concat(frames, ignore_index=True)[TIDY], retrieved


def main() -> None:
    from pdi.macro import store

    if os.environ.get("PDI_FIRE_DRILL"):
        raise CollectError("fire drill: the Macro Monitor collector was told to fail")
    tidy, retrieved = collect_all()
    store.update(tidy, retrieved)


if __name__ == "__main__":
    main()
