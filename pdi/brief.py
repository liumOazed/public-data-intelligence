"""Monthly "what changed" brief, written by code from the stored tables, never by hand.

One Markdown page per calendar month under site/briefs/, written by the first daily run of the
next month. Everything in it is read from tables the pipeline already keeps:

* data/store/macro/vintages.parquet: every macro value with the time this pipeline first saw
  it, so "new this month" and "revised this month" are exact, not guessed from release dates.
* data/store/macro/flags.parquet: every flag the fixed rule raised, with whether it held.
* site/data/v1/tech-value/price_history.json: the lowest listed price per model, day by day.
* site/data/v1/*/holdout.json: the result of each module's out-of-sample test.
* optionally a JSON list of workflow runs (from `gh run list`), for the pipeline health lines.

Every line carries its finding label (observation, explanation or prediction), taken from the
same rule the pages use.

Run as `python -m pdi.brief` (writes last month's brief if it does not exist yet) or
`python -m pdi.brief --month 2026-11 --force`.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from pdi.common import findings
from pdi.common.paths import SITE_DATA
from pdi.macro import config as macro_config

BRIEFS_DIR = Path(__file__).resolve().parents[1] / "site" / "briefs"
# A listed price that moved by at least this share between the first and last collection day
# of the month is reported. Smaller moves are counted, not listed.
PRICE_MOVE_SHARE = 0.05
MAX_LISTED = 12

OBS = findings.OBSERVATION


def _tag(label: str) -> str:
    """The same linked badge the pages use; briefs live one folder below the site root."""
    return f"[{label.capitalize()}](../methodology.qmd#finding-labels){{.finding .finding-{label}}}"


def _month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    return start, start + pd.offsets.MonthBegin(1)


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# --- Macro Monitor ---------------------------------------------------------------------------


def _keys(df: pd.DataFrame, col: str) -> set[tuple[str, pd.Timestamp]]:
    return set(zip(df["series_id"], pd.to_datetime(df[col]), strict=True))


def _isin(df: pd.DataFrame, col: str, keys: set) -> pd.Series:
    pairs = zip(df["series_id"], pd.to_datetime(df[col]), strict=True)
    return pd.Series([k in keys for k in pairs], index=df.index, dtype=bool)


def macro_section(month: str, vintages: pd.DataFrame, flags: pd.DataFrame) -> list[str]:
    start, end = _month_bounds(month)
    cat = macro_config.catalogue()
    v = vintages.assign(first_seen=pd.to_datetime(vintages["first_seen"], utc=True))
    out = ["## Macro Monitor", ""]
    if v.empty:
        return out + ["No macro readings have been stored yet.", ""]

    initial_load = v["first_seen"].min()
    seen_before = v[v["first_seen"] < start]
    in_month = v[(v["first_seen"] >= start) & (v["first_seen"] < end)]
    if start <= initial_load < end:
        # The first run stores every value the sources have ever published. Those are not
        # news, so this month counts only what arrived after the first day of collection.
        first_day = initial_load.normalize() + pd.offsets.Day(1)
        in_month = in_month[in_month["first_seen"] >= first_day]
        seen_before = v[v["first_seen"] < first_day]
        out += [
            f"Collection started on {initial_load:%-d %B %Y}. Values stored that day are the "
            "starting point, not news, so this brief counts only what arrived after it.",
            "",
        ]

    is_revision = _isin(in_month, "period", _keys(seen_before, "period"))
    new = in_month[~is_revision]
    revised = in_month[is_revision]

    gained = new.groupby("series_id")["period"].max()
    # Only series that have ever delivered a value; a catalogue entry the source does not
    # carry is listed on the Series page instead.
    stored = set(v["series_id"]) & set(cat)
    silent = sorted(stored - set(gained.index))
    out += [
        f"{_tag(OBS)} **{len(gained)} of {len(stored)} series gained at least one new month**; "
        f"{len(revised)} earlier values were revised by their publishers.",
        "",
    ]
    if silent:
        names = ", ".join(f"`{s}`" for s in silent[:MAX_LISTED])
        more = f" and {len(silent) - MAX_LISTED} more" if len(silent) > MAX_LISTED else ""
        out += [
            f"{_tag(OBS)} No new month this month: {names}{more}. Some series are published "
            "less than monthly or with long lags; a series that stays on this list two months "
            "running is worth checking at its source.",
            "",
        ]

    new_keys = _keys(new, "period")
    if not flags.empty:
        f = flags.assign(period=pd.to_datetime(flags["period"]))
        fresh = f[_isin(f, "period", new_keys)]
        fresh = fresh.sort_values("z", key=abs, ascending=False)
        out += ["### Unusual moves in this month's new readings", ""]
        if fresh.empty:
            out += [
                f"{_tag(OBS)} None. No new reading moved by {macro_config.FLAG_Z} spreads or "
                "more against its own history or its neighbours.",
                "",
            ]
        for _, r in fresh.head(MAX_LISTED).iterrows():
            out.append(f"- {_tag(r['label'])} {r['sentence']} ({r['kind']})")
        if len(fresh) > MAX_LISTED:
            out.append(f"- and {len(fresh) - MAX_LISTED} more on the Macro Monitor page.")
        out.append("")

        # A flag can be judged once the FOLLOW_MONTHS-th month after it is in. Those whose
        # deciding month arrived this month are reported with the verdict.
        f = f.assign(deciding=f["period"] + pd.DateOffset(months=macro_config.FOLLOW_MONTHS))
        judged = f[_isin(f, "deciding", new_keys) & f["outcome"].isin(["held", "reversed"])]
        if not judged.empty:
            held = int((judged["outcome"] == "held").sum())
            out += [
                "### Earlier flags that can now be judged",
                "",
                f"{_tag(OBS)} {len(judged)} flags reached their "
                f"{macro_config.FOLLOW_MONTHS}-month mark: {held} held and "
                f"{len(judged) - held} reversed (gave back more than half of the surprise).",
                "",
            ]
            for _, r in judged.sort_values("period").head(MAX_LISTED).iterrows():
                out.append(f"- {r['outcome'].capitalize()}: {r['sentence']}")
            out.append("")

    if not revised.empty:
        by_series = revised.groupby("series_id").size().sort_values(ascending=False)
        top = ", ".join(f"`{s}` ({n})" for s, n in by_series.head(MAX_LISTED).items())
        out += [
            "### Revisions",
            "",
            f"{_tag(OBS)} Values changed after first publication: {top}. The first-seen values "
            "are kept, so a later backtest can use what was known at the time.",
            "",
        ]
    return out


# --- Tech Value --------------------------------------------------------------------------------


def tech_section(month: str, history: pd.DataFrame) -> list[str]:
    start, end = _month_bounds(month)
    out = ["## Tech Value", ""]
    if history.empty:
        return out + ["No prices have been collected yet.", ""]
    h = history.assign(date=pd.to_datetime(history["date"], utc=True))
    h = h[(h["date"] >= start) & (h["date"] < end)]
    if h["date"].nunique() < 2:
        return out + [
            f"{_tag(OBS)} Prices were collected on {h['date'].nunique()} day(s) this month, "
            "too few to say what changed.",
            "",
        ]
    first_day, last_day = h["date"].min(), h["date"].max()
    first = h[h["date"] == first_day].set_index(["kind", "model_key"])["price_usd"]
    last = h[h["date"] == last_day].set_index(["kind", "model_key"])["price_usd"]
    both = pd.concat({"first": first, "last": last}, axis=1).dropna()
    both["change"] = both["last"] / both["first"] - 1
    moved = both[both["change"].abs() >= PRICE_MOVE_SHARE].sort_values("change")
    appeared = sorted(set(last.index) - set(first.index))
    vanished = sorted(set(first.index) - set(last.index))
    out += [
        f"{_tag(OBS)} Between {first_day:%-d %B} and {last_day:%-d %B}, the lowest listed price "
        f"of {len(moved)} of {len(both)} models moved by {PRICE_MOVE_SHARE:.0%} or more. "
        f"{len(appeared)} models gained a price and {len(vanished)} lost every listing.",
        "",
    ]
    if not moved.empty:
        out += ["| Part | Start of month | End of month | Change |", "|---|---:|---:|---:|"]
        for (kind, key), r in moved.head(MAX_LISTED).iterrows():
            out.append(
                f"| {key} ({kind.upper()}) | ${r['first']:,.2f} | ${r['last']:,.2f} | "
                f"{r['change']:+.1%} |"
            )
        out.append("")
    return out


# --- Tests and pipeline health ------------------------------------------------------------


def tests_section(site_data: Path) -> list[str]:
    out = ["## Out-of-sample tests", ""]
    tv = _load_json(site_data / "tech-value" / "holdout.json")
    if tv and tv["meta"].get("verdict"):
        verdict = tv["meta"]["verdict"]
        out.append(f"- {_tag(verdict['finding']['label'])} Tech Value: {verdict['statement']}")
    fp = _load_json(site_data / "tech-value" / "fair_price.json")
    for kind, verdict in ((fp or {}).get("meta", {}).get("verdicts") or {}).items():
        out.append(
            f"- {_tag(verdict['finding']['label'])} Fair price ({kind.upper()}): "
            f"{verdict['statement']}"
        )
    mm = _load_json(site_data / "macro" / "holdout.json")
    if mm and mm["meta"].get("verdict"):
        verdict = mm["meta"]["verdict"]
        out.append(
            f"- {_tag(verdict['finding']['label'])} Macro forecast test: {verdict['statement']}"
        )
    return out + [""] if len(out) > 2 else out + ["No test has run yet.", ""]


def health_section(month: str, runs: list[dict] | None) -> list[str]:
    out = ["## Pipeline health", ""]
    if runs is None:
        return out + ["The run history was not available when this brief was written.", ""]
    start, end = _month_bounds(month)
    df = pd.DataFrame(runs)
    if df.empty:
        return out + [f"{_tag(OBS)} No daily pipeline run on main this month.", ""]
    df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True)
    df = df[(df["createdAt"] >= start) & (df["createdAt"] < end)]
    days = pd.date_range(start, end - pd.offsets.Day(1), freq="D")
    ok_days = set(df.loc[df["conclusion"] == "success", "createdAt"].dt.normalize())
    missing = [d for d in days if d not in ok_days]
    scheduled = df[df["event"] == "schedule"]
    out += [
        f"{_tag(OBS)} The daily pipeline succeeded on {len(ok_days)} of {len(days)} days "
        f"({len(scheduled)} scheduled runs, {int((df['conclusion'] == 'failure').sum())} "
        "failed runs).",
        "",
    ]
    if missing and len(missing) < len(days):
        out += [f"Days with no successful run: {', '.join(f'{d:%-d}' for d in missing)}.", ""]
    return out


# --- Assembly ---------------------------------------------------------------------------------


def build(
    month: str,
    *,
    vintages: pd.DataFrame,
    flags: pd.DataFrame,
    history: pd.DataFrame,
    site_data: Path = SITE_DATA,
    runs: list[dict] | None = None,
    written: date | None = None,
) -> str:
    label = pd.Timestamp(f"{month}-01").strftime("%B %Y")
    written = written or date.today()
    head = [
        "---",
        f'title: "What changed in {label}"',
        f"date: {written.isoformat()}",
        f'description: "Written by the pipeline on {written:%-d %B %Y} from the stored tables."',
        "---",
        "",
        "This brief is generated by code from the tables the pipeline keeps; nobody edits it. "
        "Each line carries its label: an *observation* is measured, an *explanation* comes from "
        "a model that has not beaten its naive baseline out of sample, and a *prediction* comes "
        "from one that has.",
        "",
    ]
    body = (
        macro_section(month, vintages, flags)
        + tech_section(month, history)
        + tests_section(site_data)
        + health_section(month, runs)
    )
    return "\n".join(head + body).rstrip() + "\n"


def previous_month(today: date) -> str:
    return (pd.Timestamp(today).to_period("M") - 1).strftime("%Y-%m")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the monthly what-changed brief.")
    parser.add_argument("--month", help="YYYY-MM; default: last month")
    parser.add_argument("--force", action="store_true", help="overwrite an existing brief")
    parser.add_argument("--runs", type=Path, help="JSON from gh run list (optional)")
    parser.add_argument("--out-dir", type=Path, default=BRIEFS_DIR)
    args = parser.parse_args(argv)

    month = args.month or previous_month(date.today())
    path = args.out_dir / f"{month}.md"
    if path.exists() and not args.force:
        print(f"brief for {month} already exists: {path}")
        return 0
    vintages = (
        pd.read_parquet(macro_config.VINTAGES)
        if macro_config.VINTAGES.exists()
        else pd.DataFrame(columns=["series_id", "period", "value", "first_seen"])
    )
    # The pipeline started after that month ended: there is nothing to compare, no brief.
    if (
        vintages.empty
        or pd.to_datetime(vintages["first_seen"], utc=True).min() >= _month_bounds(month)[1]
    ):
        print(f"no data collected in {month}; no brief written")
        return 0
    flags = pd.read_parquet(macro_config.FLAGS) if macro_config.FLAGS.exists() else pd.DataFrame()
    hist_payload = _load_json(SITE_DATA / "tech-value" / "price_history.json")
    history = (
        pd.DataFrame(hist_payload["data"], columns=hist_payload["meta"]["columns"])
        if hist_payload
        else pd.DataFrame()
    )
    runs = json.loads(args.runs.read_text()) if args.runs and args.runs.exists() else None
    text = build(month, vintages=vintages, flags=flags, history=history, runs=runs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
