import pandas as pd

from pdi import brief


def _vintages():
    rows = []
    # Initial load on 15 Oct: two series with history to September.
    for sid in ("US.cpi", "US.payrolls"):
        for p in pd.date_range("2026-01-01", "2026-09-01", freq="MS"):
            rows.append((sid, p, 100.0, "2026-10-15T09:00:00+00:00"))
    # November: US.cpi gains October, and its August value is revised. US.payrolls is silent.
    rows.append(("US.cpi", pd.Timestamp("2026-10-01"), 103.0, "2026-11-12T09:30:00+00:00"))
    rows.append(("US.cpi", pd.Timestamp("2026-08-01"), 100.4, "2026-11-12T09:30:00+00:00"))
    return pd.DataFrame(rows, columns=["series_id", "period", "value", "first_seen"])


def _flags():
    return pd.DataFrame(
        [
            {"series_id": "US.cpi", "period": pd.Timestamp("2026-10-01"), "kind": "own history",
             "z": 3.1, "outcome": "too early", "sentence": "CPI jumped in October.",
             "label": "observation"},
            {"series_id": "US.cpi", "period": pd.Timestamp("2026-07-01"), "kind": "own history",
             "z": -2.8, "outcome": "held", "sentence": "CPI fell in July.",
             "label": "observation"},
        ]
    )  # fmt: skip


def _history():
    return pd.DataFrame(
        [
            ("2026-11-01", "gpu", "nvidia rtx 5070", 600.0),
            ("2026-11-01", "cpu", "amd ryzen 7 9700x", 300.0),
            ("2026-11-01", "cpu", "intel core i5-14600k", 250.0),
            ("2026-11-30", "gpu", "nvidia rtx 5070", 540.0),
            ("2026-11-30", "cpu", "amd ryzen 7 9700x", 303.0),
            ("2026-11-30", "gpu", "amd rx 9070", 650.0),
        ],
        columns=["date", "kind", "model_key", "price_usd"],
    )


def test_november_brief_reports_what_changed(tmp_path):
    runs = [
        {"conclusion": "success", "createdAt": f"2026-11-{d:02d}T09:20:00Z", "event": "schedule"}
        for d in range(1, 31)
        if d != 7
    ] + [{"conclusion": "failure", "createdAt": "2026-11-07T09:20:00Z", "event": "schedule"}]
    text = brief.build(
        "2026-11", vintages=_vintages(), flags=_flags(), history=_history(),
        site_data=tmp_path, runs=runs,
    )  # fmt: skip
    assert 'title: "What changed in November 2026"' in text
    # New month and revision are told apart; the silent series is named.
    assert "**1 of 2 series gained at least one new month**; 1 earlier values" in text
    assert "`US.payrolls`" in text.split("No new month this month:")[1].splitlines()[0]
    # The October flag is on a new reading; the July flag is judged by October's arrival.
    assert "CPI jumped in October." in text
    assert "1 flags reached their 3-month mark: 1 held" in text
    # Price moves of 5% or more are listed; the 1% one is only counted.
    assert "| nvidia rtx 5070 (GPU) | $600.00 | $540.00 | -10.0% |" in text
    assert "ryzen 7 9700x (CPU)" not in text
    assert "1 models gained a price and 1 lost every listing" in text
    assert "succeeded on 29 of 30 days" in text and "no successful run: 7." in text
    # Every finding line carries a label badge.
    assert text.count("{.finding .finding-observation}") >= 5


def test_first_collection_day_is_not_news():
    v = _vintages()
    v = v[v["first_seen"] < "2026-11-01"].assign(first_seen="2026-11-15T09:00:00+00:00")
    text = brief.build("2026-11", vintages=v, flags=_flags(), history=_history().head(0))
    assert "Collection started on 15 November 2026" in text
    assert "**0 of 2 series gained" in text
    assert "No prices have been collected yet." in text


def test_main_skips_months_before_collection_and_existing_briefs(tmp_path, monkeypatch):
    v = _vintages()
    path = tmp_path / "v.parquet"
    v.to_parquet(path)
    monkeypatch.setattr(brief.macro_config, "VINTAGES", path)
    monkeypatch.setattr(brief.macro_config, "FLAGS", tmp_path / "none.parquet")
    monkeypatch.setattr(brief, "SITE_DATA", tmp_path)
    assert brief.main(["--month", "2026-09", "--out-dir", str(tmp_path)]) == 0
    assert not (tmp_path / "2026-09.md").exists()
    assert brief.main(["--month", "2026-11", "--out-dir", str(tmp_path)]) == 0
    first = (tmp_path / "2026-11.md").read_text()
    (tmp_path / "2026-11.md").write_text("kept")
    brief.main(["--month", "2026-11", "--out-dir", str(tmp_path)])
    assert (tmp_path / "2026-11.md").read_text() == "kept"
    assert "What changed in November 2026" in first


def test_previous_month():
    assert brief.previous_month(pd.Timestamp("2027-01-01").date()) == "2026-12"
