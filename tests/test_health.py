import json
from datetime import datetime, timedelta, timezone

from pdi import health

NOW = datetime(2026, 12, 1, 12, tzinfo=timezone.utc)


def _write(root, module, table, rows, generated, sources):
    path = root / module / f"{table}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": generated.isoformat(),
        "sources": [{"id": k, "retrieved_at": v.isoformat() if v else None} for k, v in sources],
    }
    path.write_text(json.dumps({"meta": meta, "data": [{"x": i} for i in range(rows)]}))


def _healthy(root, now=NOW):
    fresh = now - timedelta(hours=3)
    _write(root, "tech-value", "value", 5, fresh, [("centralcomputer", fresh)])
    _write(
        root, "tech-value", "benchmarks", 5, fresh, [("blender_opendata", now - timedelta(days=6))]
    )
    _write(root, "macro", "series", 70, fresh, [("fred", fresh), ("bis", fresh)])
    _write(root, "macro", "history", 10, fresh, [("fred", fresh)])


def test_healthy_tables_pass(tmp_path):
    _healthy(tmp_path)
    assert health.check(tmp_path, NOW) == []


def test_green_run_with_a_stale_daily_source_is_caught(tmp_path):
    _healthy(tmp_path)
    stale = NOW - timedelta(hours=60)
    fresh = NOW - timedelta(hours=1)
    _write(tmp_path, "macro", "series", 70, fresh, [("fred", stale), ("bis", fresh)])
    _write(tmp_path, "macro", "history", 10, fresh, [("fred", stale)])
    problems = health.check(tmp_path, NOW)
    # One problem for the source, counted across the two tables that cite it.
    assert len(problems) == 1 and "fred" in problems[0] and "2 tables" in problems[0]


def test_weekly_source_gets_nine_days(tmp_path):
    _healthy(tmp_path)
    fresh = NOW - timedelta(hours=1)
    _write(
        tmp_path,
        "tech-value",
        "benchmarks",
        5,
        fresh,
        [("blender_opendata", NOW - timedelta(days=10))],
    )
    assert any("blender_opendata" in p for p in health.check(tmp_path, NOW))


def test_empty_or_missing_tables_are_caught(tmp_path):
    _healthy(tmp_path)
    fresh = NOW - timedelta(hours=1)
    _write(tmp_path, "macro", "series", 12, fresh, [("fred", fresh)])
    (tmp_path / "tech-value" / "value.json").unlink()
    problems = health.check(tmp_path, NOW)
    assert any("macro/series: 12 rows" in p for p in problems)
    assert any("tech-value/value: file missing" in p for p in problems)


def test_a_file_the_daily_run_stopped_rewriting_is_caught(tmp_path):
    _healthy(tmp_path)
    old = NOW - timedelta(days=3)
    _write(tmp_path, "macro", "history", 10, old, [("fred", NOW - timedelta(hours=1))])
    assert any("macro/history: last generated" in p for p in health.check(tmp_path, NOW))


def test_never_retrieved_counts_as_stale(tmp_path):
    _healthy(tmp_path)
    fresh = NOW - timedelta(hours=1)
    _write(tmp_path, "tech-value", "value", 5, fresh, [("centralcomputer", None)])
    assert any("never" in p for p in health.check(tmp_path, NOW))
