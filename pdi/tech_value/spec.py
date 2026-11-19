"""SPEC CPU 2017: collect the published-results dump and turn it into one tidy table.

The dump has one row per published result. Cache and memory are free text
("1280 KB I+D on chip per core", "512 GB (16 x 32 GB 2Rx4 PC5-4800B-R)"), so most of this file
is parsing, and every parser is tested against saved sample text.
"""

from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from pdi.common.fetch import fetch, latest, read_meta
from pdi.tech_value import config

# Column names as SPEC writes them, mapped to ours. A missing one stops the run.
RAW_COLUMNS = {
    "Benchmark": "benchmark",
    "Hardware Vendor": "hardware_vendor",
    "System": "system",
    "# Cores": "cores",
    "# Chips": "chips",
    "# Enabled Threads Per Core": "threads_per_core",
    "Processor": "processor",
    "Processor MHz": "mhz",
    "1st Level Cache": "l1_text",
    "2nd Level Cache": "l2_text",
    "3rd Level Cache": "l3_text",
    "Memory": "memory_text",
    "Operating System": "os",
    "Compiler": "compiler",
    "HW Avail": "hw_avail",
    "Test Date": "test_date",
    "Published": "published",
    "Result": "peak_score",
    "Baseline": "base_score",
    "Disclosures": "disclosures",
}

_SIZE = re.compile(r"([\d.]+)\s*(KB|MB|GB|TB)\b", re.I)
_PER_N_CORES = re.compile(r"per\s+(\d+)\s+cores?", re.I)
_HREF = re.compile(r'href="?([^">\s]+\.(?:html|txt|pdf))', re.I)
_TO_MB = {"KB": 1 / 1024, "MB": 1.0, "GB": 1024.0, "TB": 1024.0 * 1024}

FAMILIES = (
    ("xeon platinum", "Xeon Platinum"),
    ("xeon gold", "Xeon Gold"),
    ("xeon silver", "Xeon Silver"),
    ("xeon bronze", "Xeon Bronze"),
    ("xeon max", "Xeon Max"),
    ("xeon 6", "Xeon 6"),
    ("xeon", "Xeon (other)"),
    ("epyc", "EPYC"),
    ("threadripper", "Threadripper"),
    ("ryzen", "Ryzen"),
    ("core", "Core"),
    ("ampere", "Ampere"),
)

TIDY_SCHEMA = pa.DataFrameSchema(
    {
        "benchmark": pa.Column(str, pa.Check.eq(config.SPEC_BENCHMARK)),
        "processor": pa.Column(str, pa.Check.str_length(min_value=3)),
        "vendor": pa.Column(str, pa.Check.isin(["Intel", "AMD", "Other"])),
        "family": pa.Column(str),
        "cores": pa.Column(int, pa.Check.in_range(1, 4096)),
        "chips": pa.Column(int, pa.Check.in_range(1, 64)),
        "threads_per_core": pa.Column(int, pa.Check.in_range(1, 8)),
        # A value outside this range means MHz was read as GHz or the column moved.
        "mhz": pa.Column(float, pa.Check.in_range(500, 8000)),
        "l2_mb_per_core": pa.Column(float, pa.Check.in_range(0, 64), nullable=True),
        "l3_mb_per_core": pa.Column(float, pa.Check.in_range(0, 512), nullable=True),
        "memory_gb": pa.Column(float, pa.Check.in_range(1, 65536), nullable=True),
        # SPEC CPU 2017 has been run on hardware from as early as 2007.
        "hw_avail": pa.Column("datetime64[ns]", pa.Check.ge(pd.Timestamp("2000-01-01"))),
        "base_score": pa.Column(float, pa.Check.in_range(0.5, 100000)),
    },
    strict=False,
    coerce=True,
)


def collect(today: date | None = None) -> Path:
    """Save the full results dump for one suite, untouched."""
    today = today or date.today()
    dest = config.RAW_DIR / "spec" / f"{today.isoformat()}_{config.SPEC_CONF}.csv"
    return fetch(
        config.SPEC_ENDPOINT,
        dest,
        params={"conf": config.SPEC_CONF, "op": "dump", "format": "csvdump"},
        timeout=300,
    )


def cache_mb_per_core(text: str | None, cores_per_chip: float) -> float | None:
    """Parse a SPEC cache description into MB per core.

    "32 KB I + 48 KB D on chip per core"  -> 0.078
    "2 MB I+D on chip per core"           -> 2.0
    "60 MB I+D on chip per chip"          -> 60 / cores_per_chip
    "16 MB I+D on chip per 8 cores"       -> 2.0
    "None"                                -> 0.0
    """
    if text is None or not str(text).strip():
        return None
    text = str(text)
    if text.strip().lower() in ("none", "n/a", "--"):
        return 0.0
    sizes = _SIZE.findall(text)
    if not sizes:
        return None
    total = sum(float(n) * _TO_MB[u.upper()] for n, u in sizes)
    lower = text.lower()
    if m := _PER_N_CORES.search(lower):
        return total / int(m.group(1))
    if "per chip" in lower or "per socket" in lower:
        return total / cores_per_chip if cores_per_chip else None
    # "per core" is also what the dump means when it gives no scope.
    return total


def memory_gb(text: str | None) -> float | None:
    """ "512 GB (16 x 32 GB 2Rx4 PC5-4800B-R)" -> 512.0. Only the leading total is read."""
    if text is None:
        return None
    m = _SIZE.match(str(text).strip())
    if not m:
        return None
    return float(m.group(1)) * _TO_MB[m.group(2).upper()] / 1024


def vendor_of(processor: str) -> str:
    p = processor.lower()
    if "intel" in p or "xeon" in p:
        return "Intel"
    if "amd" in p or "epyc" in p or "ryzen" in p:
        return "AMD"
    return "Other"


def family_of(processor: str) -> str:
    p = processor.lower()
    for needle, label in FAMILIES:
        if needle in p:
            return label
    return "Other"


def disclosure_url(cell: str | None) -> str | None:
    if not cell:
        return None
    m = _HREF.search(str(cell))
    if not m:
        return None
    href = m.group(1)
    return href if href.startswith("http") else "https://www.spec.org" + href


def parse(raw_csv: bytes | str) -> pd.DataFrame:
    """Raw dump -> tidy table: one row per published result, standard names and units."""
    buf = io.BytesIO(raw_csv) if isinstance(raw_csv, bytes) else io.StringIO(raw_csv)
    raw = pd.read_csv(buf, dtype=str, encoding_errors="replace")
    raw.columns = [c.strip() for c in raw.columns]
    # Current dumps carry the links in "Disclosure" and leave "Disclosures" empty.
    if "Disclosure" in raw.columns:
        links = raw["Disclosure"]
        raw["Disclosures"] = links.where(links.notna(), raw.get("Disclosures"))
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"SPEC dump is missing columns {missing}. Columns present: {list(raw.columns)}"
        )
    df = raw[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS)
    n_raw = len(df)

    for col in ("cores", "chips", "threads_per_core", "mhz", "base_score", "peak_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("hw_avail", "test_date", "published"):
        df[col] = pd.to_datetime(df[col].str.strip(), format="%b-%Y", errors="coerce")

    # Non-compliant and withdrawn results carry no score. They are dropped, not repaired.
    df = df[
        (df["benchmark"].str.strip() == config.SPEC_BENCHMARK)
        & (df["base_score"] > 0)
        & (df["cores"] > 0)
        & (df["chips"] > 0)
        & (df["mhz"] > 0)
        & df["hw_avail"].notna()
        & df["processor"].notna()
    ].copy()
    if len(df) < 0.5 * n_raw:
        raise ValueError(
            f"only {len(df)} of {n_raw} SPEC rows are usable; the dump format has probably changed"
        )

    df["benchmark"] = df["benchmark"].str.strip()
    df["processor"] = df["processor"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["vendor"] = df["processor"].map(vendor_of)
    df["family"] = df["processor"].map(family_of)
    df["threads_per_core"] = df["threads_per_core"].fillna(1)
    per_chip = df["cores"] / df["chips"]
    df["l2_mb_per_core"] = [
        cache_mb_per_core(t, c) for t, c in zip(df["l2_text"], per_chip, strict=True)
    ]
    df["l3_mb_per_core"] = [
        cache_mb_per_core(t, c) for t, c in zip(df["l3_text"], per_chip, strict=True)
    ]
    df["memory_gb"] = df["memory_text"].map(memory_gb)
    df["disclosure_url"] = df["disclosures"].map(disclosure_url)
    df = df.drop(columns=["disclosures", "l1_text"])
    for col in ("cores", "chips", "threads_per_core"):
        df[col] = df[col].astype(int)
    df = TIDY_SCHEMA.validate(df)
    return df.sort_values(["hw_avail", "processor", "base_score"]).reset_index(drop=True)


def transform() -> pd.DataFrame:
    """Latest raw dump -> committed tidy table."""
    raw_path = latest(config.RAW_DIR / "spec", f"*_{config.SPEC_CONF}.csv")
    df = parse(raw_path.read_bytes())
    previous = len(pd.read_parquet(config.SPEC_TABLE)) if config.SPEC_TABLE.exists() else 0
    # Published results are only ever added. A shrinking table means a bad download.
    if len(df) < 0.9 * previous:
        raise ValueError(f"SPEC table would shrink from {previous} to {len(df)} rows; stopping")
    df["retrieved_at"] = read_meta(raw_path)["retrieved_at"]
    config.SPEC_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.SPEC_TABLE, index=False)
    return df


if __name__ == "__main__":
    path = collect()
    print(f"collected {path}")
    print(f"tidy rows: {len(transform())}")
