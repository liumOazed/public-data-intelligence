"""Blender Open Data: collect the public snapshot and aggregate it to one row per device.

The snapshot is a zip holding a JSON-lines file with one benchmark launch per line. A launch
renders several scenes; the score Blender shows is the sum of samples per minute across the
three standard scenes. Here a launch counts only if it has exactly one compute device and all
three scenes, and results are kept apart by Blender version and by compute backend.
"""

from __future__ import annotations

import io
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import IO, Iterable

import pandas as pd
import pandera.pandas as pa

from pdi.common.fetch import fetch, latest, read_meta
from pdi.tech_value import config

TIDY_SCHEMA = pa.DataFrameSchema(
    {
        "device_name": pa.Column(str, pa.Check.str_length(min_value=2)),
        "device_type": pa.Column(str),
        "blender_version": pa.Column(str, pa.Check.str_matches(r"^\d+\.\d+")),
        "n_runs": pa.Column(int, pa.Check.ge(config.BLENDER_MIN_RUNS)),
        "median_score": pa.Column(float, pa.Check.in_range(0.1, 1e6)),
        "p25_score": pa.Column(float),
        "p75_score": pa.Column(float),
    },
    strict=False,
    coerce=True,
)


def collect(today: date | None = None) -> Path:
    today = today or date.today()
    dest = config.RAW_DIR / "blender" / f"{today.isoformat()}_opendata.zip"
    return fetch(config.BLENDER_SNAPSHOT_URL, dest, timeout=900)


def _minor(version: str) -> str:
    """ "4.2.1" -> "4.2". Patch releases of one version render the same scenes the same way."""
    parts = str(version).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(version)


def parse_launch(record: dict) -> tuple[tuple[str, str, str], float] | str:
    """One launch -> ((device_name, device_type, blender_version), score), or a skip reason."""
    data = record.get("data", record)
    if not isinstance(data, list) or not data:
        return "old_schema"  # pre-3.0 launches report render time, a different measurement
    scenes: dict[str, float] = {}
    keys = set()
    for entry in data:
        stats = entry.get("stats") if isinstance(entry, dict) else None
        if (
            isinstance(stats, dict)
            and "samples_per_minute" not in stats
            and ("total_render_time" in stats or "render_time_no_sync" in stats)
        ):
            # Blender 2.x launches in the newer file layout still report render time, not
            # samples per minute. A different measurement, not a broken record.
            return "old_schema"
        try:
            devices = entry["device_info"]["compute_devices"]
            if len(devices) != 1:
                return "multi_device"
            device = devices[0]
            name = device["name"] if isinstance(device, dict) else str(device)
            keys.add(
                (
                    " ".join(name.split()),
                    str(entry["device_info"]["device_type"]).upper(),
                    _minor(entry["blender_version"]["version"]),
                )
            )
            scenes[str(entry["scene"]["label"]).lower()] = float(
                entry["stats"]["samples_per_minute"]
            )
        except (KeyError, TypeError, ValueError, IndexError):
            return "unreadable"
    if len(keys) != 1:
        return "mixed_launch"
    if set(scenes) != set(config.BLENDER_SCENES):
        return "incomplete_scenes"
    score = sum(scenes.values())
    if not score > 0:
        return "zero_score"
    return keys.pop(), score


def aggregate(lines: Iterable[bytes | str]) -> tuple[pd.DataFrame, Counter]:
    """Stream launches and reduce them to medians per device, backend and Blender version."""
    scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    skipped: Counter = Counter()
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            skipped["bad_json"] += 1
            continue
        parsed = parse_launch(record)
        if isinstance(parsed, str):
            skipped[parsed] += 1
        else:
            scores[parsed[0]].append(parsed[1])
            skipped["used"] += 1

    rows = []
    for (name, dtype, version), values in scores.items():
        if len(values) < config.BLENDER_MIN_RUNS:
            continue
        q = statistics.quantiles(values, n=4)
        rows.append(
            {
                "device_name": name,
                "device_type": dtype,
                "blender_version": version,
                "n_runs": len(values),
                "median_score": statistics.median(values),
                "p25_score": q[0],
                "p75_score": q[2],
            }
        )
    return pd.DataFrame(rows), skipped


def _open_jsonl(zip_path: Path) -> IO[bytes]:
    archive = zipfile.ZipFile(zip_path)
    members = [n for n in archive.namelist() if n.lower().endswith((".jsonl", ".json"))]
    if not members:
        raise ValueError(f"no JSON-lines file in {zip_path.name}: {archive.namelist()[:10]}")
    # The data file is by far the largest member.
    biggest = max(members, key=lambda n: archive.getinfo(n).file_size)
    return archive.open(biggest)


def transform() -> pd.DataFrame:
    raw_path = latest(config.RAW_DIR / "blender", "*_opendata.zip")
    with _open_jsonl(raw_path) as handle:
        df, skipped = aggregate(io.BufferedReader(handle))
    print(f"blender launches: {dict(skipped)}")
    total = sum(skipped.values())
    if skipped["used"] < 1000 or skipped["unreadable"] > 0.2 * total:
        raise ValueError(
            f"Blender snapshot looks wrong: {dict(skipped)}. The launch format has probably changed"
        )
    df = TIDY_SCHEMA.validate(df)
    df["retrieved_at"] = read_meta(raw_path)["retrieved_at"]
    df = df.sort_values(["blender_version", "median_score"], ascending=[False, False])
    config.BLENDER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(config.BLENDER_TABLE, index=False)
    return df


if __name__ == "__main__":
    path = collect()
    print(f"collected {path}")
    print(f"device rows: {len(transform())}")
