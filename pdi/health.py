"""Silent-failure check: is the published data as fresh as the schedule says it should be?

A green workflow run is not proof that anything was collected. A collector can return the same
cached answer, a source can stop updating, or an export can quietly write an empty table. This
reads only the published contract files in site/data/v1/ and checks three things:

1. Every source each file cites was retrieved within its allowed age.
2. Every file was regenerated within its allowed age.
3. The tables a page cannot do without are not empty (and the Macro Monitor still has at least
   as many series as the collector's own floor).

It uses only the standard library, so the heartbeat workflow can run it without installing the
project. Run as `python -m pdi.health`; it exits 1 and prints one line per problem when any check
fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SITE_DATA = Path(__file__).resolve().parents[1] / "site" / "data" / "v1"

# How old a source's retrieval may be, in hours, before it counts as a silent failure. Daily
# collectors get two days (the heartbeat's own 48-hour rule); weekly ones get nine (a week plus
# the scheduler's slack). A source not listed here gets the weekly allowance.
SOURCE_MAX_AGE_H = {
    "centralcomputer": 48,
    "fred": 48,
    "bis": 48,
    "opendosm": 48,
    "indeed_postings": 48,
    "imf": 48,
}
WEEKLY_MAX_AGE_H = 9 * 24
# Every exported file is rewritten by the daily run.
FILE_MAX_AGE_H = 48
# Tables that must never be empty. Values are the minimum row count.
MIN_ROWS = {
    ("tech-value", "value"): 1,
    ("tech-value", "benchmarks"): 1,
    ("macro", "series"): 60,  # same floor as pdi.macro.config.MIN_SERIES_COLLECTED
    ("macro", "history"): 1,
    ("trade", "partners"): 7,  # China and the six economies in pdi.trade.config.PARTNERS
    ("energy", "demand_latest"): 6,  # the six economies pdi.energy.config.REQUIRED names
    ("procurement", "monthly"): 100,  # every month since October 2016
}


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    out = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return out if out.tzinfo else out.replace(tzinfo=timezone.utc)


def check(site_data: Path = SITE_DATA, now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    problems: list[str] = []
    files = sorted(site_data.glob("*/*.json"))
    if not files:
        return [f"no exported tables under {site_data}"]
    seen: set[tuple[str, str]] = set()
    for path in files:
        module, table = path.parent.name, path.stem
        seen.add((module, table))
        try:
            payload = json.loads(path.read_text())
            meta = payload["meta"]
            rows = len(payload["data"])
        except (ValueError, KeyError, TypeError) as exc:
            problems.append(f"{module}/{table}: unreadable ({type(exc).__name__})")
            continue
        if meta.get("one_off"):
            # A one-off experiment's table (the Demand Radar) is not refreshed by design.
            seen.add((module, table))
            continue
        generated = _parse(meta.get("generated_at"))
        if generated is None or now - generated > timedelta(hours=FILE_MAX_AGE_H):
            problems.append(
                f"{module}/{table}: last generated {meta.get('generated_at') or 'never'}, "
                f"more than {FILE_MAX_AGE_H} hours ago"
            )
        for src in meta.get("sources", []):
            limit = SOURCE_MAX_AGE_H.get(src.get("id"), WEEKLY_MAX_AGE_H)
            when = _parse(src.get("retrieved_at"))
            if when is None or now - when > timedelta(hours=limit):
                problems.append(
                    f"{module}/{table}: source {src.get('id')} last retrieved "
                    f"{src.get('retrieved_at') or 'never'}, more than {limit} hours ago"
                )
        need = MIN_ROWS.get((module, table))
        if need is not None and rows < need:
            problems.append(f"{module}/{table}: {rows} rows, expected at least {need}")
    for module, table in MIN_ROWS:
        if (module, table) not in seen:
            problems.append(f"{module}/{table}: file missing")
    return _dedupe(problems)


def _dedupe(problems: list[str]) -> list[str]:
    """One stale source shows up in every file that cites it; report it once with a count."""
    out, by_source = [], {}
    for p in problems:
        if ": source " in p:
            _, rest = p.split(": source ", 1)
            by_source.setdefault(rest, 0)
            by_source[rest] += 1
        else:
            out.append(p)
    out += [f"source {rest} (cited in {n} tables)" for rest, n in by_source.items()]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site-data", type=Path, default=SITE_DATA)
    args = parser.parse_args(argv)
    problems = check(args.site_data)
    if problems:
        print("\n".join(f"- {p}" for p in problems))
        return 1
    print("all published tables are fresh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
