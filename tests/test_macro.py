from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdi.common.fetch import CollectError
from pdi.macro import collect, config, export, flags, store


def synthetic_obs(months: int = 200, seed: int = 0, shock: dict | None = None) -> pd.DataFrame:
    """Every catalogue series as a random walk with a known seasonal pattern; `shock` adds a
    jump to one series in its last month."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end="2026-09-01", periods=months, freq="MS")
    rows = []
    for sid, s in config.catalogue().items():
        step = rng.normal(0, 0.3, months)
        if s.seasonal == "nsa":
            step += 2.0 * (idx.month == 12)
        if shock and sid in shock:
            step[-1] += shock[sid]
        if s.transform == "log":
            vals = 100 * np.exp(np.cumsum(step) / 100)
        else:
            vals = 5 + np.cumsum(step) / 10
        rows.append(pd.DataFrame({"series_id": sid, "period": idx, "value": vals}))
    return pd.concat(rows, ignore_index=True)


def test_catalogue_covers_scope():
    cat = config.catalogue()
    assert {s.country for s in cat.values()} == set(config.COUNTRIES)
    assert sum(s.country == "US" for s in cat.values()) >= 25
    assert len(cat) >= config.MIN_SERIES_COLLECTED


def test_fred_csv_parser_handles_missing_values():
    s = config.catalogue()["US.cpi"]
    text = "observation_date,CPIAUCSL\n2026-07-01,320.1\n2026-08-01,.\n2026-09-01,321.0\n"
    df = collect.parse_fred_csv(text, s)
    assert list(df["value"]) == [320.1, 321.0]
    assert df["period"].iloc[-1] == pd.Timestamp("2026-09-01")


def test_fred_copyrighted_series_stops_the_run():
    s = config.catalogue()["US.cpi"]
    with pytest.raises(CollectError, match="copyright"):
        collect.check_fred_notes({"seriess": [{"notes": "Copyright, 2026, Somebody."}]}, s)
    collect.check_fred_notes({"seriess": [{"notes": "Public domain."}]}, s)


def test_bis_parser_prefers_period_average_and_keeps_scope():
    text = (
        "FREQ,REF_AREA,CURRENCY,COLLECTION,TIME_PERIOD,OBS_VALUE\n"
        "M,TH,THB,A,2026-09,33.2\nM,TH,THB,E,2026-09,33.9\n"
        "M,MY,MYR,E,2026-09,4.07\nM,CN,CNY,A,2026-09,7.1\n"
    )
    df = collect.parse_bis_csv(text, "fx_usd").set_index("series_id")["value"]
    assert df.to_dict() == {"TH.fx_usd": 33.2, "MY.fx_usd": 4.07}


def test_dosm_parser_filters_to_levels():
    spec = config.DOSM_SERIES[0]
    rows = [
        {"date": "2026-09-01", "series": "abs", "index": 149.9, "index_sa": 147.5},
        {"date": "2026-09-01", "series": "growth_yoy", "index": 6.3, "index_sa": 6.3},
    ]
    df = collect.parse_dosm(rows, spec)
    assert list(df["value"]) == [147.5]


def test_indeed_parser_drops_the_incomplete_month():
    today = pd.Timestamp.today().normalize()
    days = pd.date_range(today - pd.Timedelta(days=70), today)
    text = "date,jobcountry,indeed_job_postings_index_SA,indeed_job_postings_index_NSA,variable\n"
    text += "".join(f"{d:%Y-%m-%d},US,100,100,total postings\n" for d in days)
    text += "".join(f"{d:%Y-%m-%d},US,50,50,new postings\n" for d in days)
    df = collect.parse_indeed(text)
    assert (df["value"] == 100).all()
    assert df["period"].max() < today.to_period("M").to_timestamp()


def test_unit_jump_is_rejected():
    obs = synthetic_obs(80)
    bad = obs["series_id"].eq("ID.fx_usd") & obs["period"].ge("2026-01-01")
    obs.loc[bad, "value"] *= 1000
    with pytest.raises(CollectError, match="factor"):
        store.validate(obs)


def test_future_dates_are_rejected():
    obs = synthetic_obs(80)
    obs.loc[obs["series_id"] == "TH.cpi", "period"] += pd.DateOffset(years=200)
    import pandera.errors

    with pytest.raises(pandera.errors.SchemaError):
        store.validate(obs)


def test_vintages_only_append_changes(tmp_path):
    path = tmp_path / "v.parquet"
    df = synthetic_obs(70)[["series_id", "period", "value"]]
    assert len(store.append_vintages(df, "2026-11-01", path)) == len(df)
    assert len(store.append_vintages(df, "2026-11-08", path)) == 0
    revised = df.copy()
    revised.loc[5, "value"] += 0.5
    added = store.append_vintages(revised, "2026-11-15", path)
    assert len(added) == 1
    assert len(pd.read_parquet(path)) == len(df) + 1


def test_a_planted_shock_is_flagged_and_noise_mostly_is_not():
    obs = synthetic_obs(shock={"US.payrolls": 4.0, "TH.fx_usd": -5.0})
    scores, _, flagged = flags.build(obs)
    last = obs["period"].max()
    now = flagged[(flagged["period"] == last) & (flagged["kind"] == "own history")]
    assert {"US.payrolls", "TH.fx_usd"} <= set(now["series_id"])
    assert (flagged["label"] == "observation").all()
    scored = scores["z"].notna().sum()
    own = (flagged["kind"] == "own history").sum()
    assert own / scored < 0.05


def test_seasonal_series_are_compared_with_the_same_month():
    obs = synthetic_obs()
    scores, _, _ = flags.build(obs)
    dec = scores[(scores["series_id"] == "MY.exports") & (scores["period"].dt.month == 12)]
    # December always jumps by 2; scored against other Decembers that is not a surprise.
    assert dec["z"].abs().median() < 2.5
    assert dec["expected"].dropna().median() == pytest.approx(2.0, abs=0.5)


def test_neighbour_flag_fires_when_one_country_moves_alone():
    obs = synthetic_obs(shock={"ID.fx_usd": 6.0})
    _, gaps, flagged = flags.build(obs)
    last = obs["period"].max()
    hit = flagged[(flagged["kind"] == "neighbours") & (flagged["period"] == last)]
    assert "ID.fx_usd" in set(hit["series_id"])
    assert gaps.loc[gaps["series_id"] == "ID.fx_usd", "n_neighbours"].max() >= config.MIN_NEIGHBOURS


def test_outcome_rule():
    idx = pd.date_range("2026-01-01", periods=5, freq="MS")
    held = pd.Series([4.0, 0.2, -0.3, 0.1, 0.0], index=idx)
    back = pd.Series([4.0, -1.5, -1.0, 0.0, 0.0], index=idx)
    assert flags.outcome(held, idx[0]) == "held"
    assert flags.outcome(back, idx[0]) == "reversed"
    assert flags.outcome(held, idx[3]) == "too early"


def test_export_writes_every_table(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "write_table", _writer(tmp_path))
    monkeypatch.setattr(config, "FLAGS", tmp_path / "flags.parquet")
    monkeypatch.setattr(config, "STORE_DIR", tmp_path)
    monkeypatch.setattr(config, "AUTOML_SUMMARY", tmp_path / "none.json")
    export.export_flags(synthetic_obs(shock={"US.cpi": 3.0}), {})
    export.export_forecasts({})
    got = {p.stem for p in (tmp_path / "macro").glob("*.json")}
    assert got == {
        "findings",
        "flags_latest",
        "flags_backtest",
        "series",
        "history",
        "neighbours",
        "leaderboard",
        "holdout",
        "folds",
        "forecasts",
    }


def test_forecasts_are_published_only_after_a_passed_test(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(export, "write_table", _writer(tmp_path))
    for name in ("AUTOML_SUMMARY", "AUTOML_LEADERBOARD", "AUTOML_HOLDOUT", "AUTOML_FORECASTS"):
        monkeypatch.setattr(config, name, tmp_path / name)
    board = pd.DataFrame(
        {
            "model": ["ETS", "Naive"],
            "scaled_mae": [0.7, 0.9],
            "n_forecasts": [10, 10],
            "period": "validation",
            "kind": "automl",
        }
    )
    board.to_parquet(config.AUTOML_LEADERBOARD)
    board.to_parquet(config.AUTOML_HOLDOUT)
    pd.DataFrame(
        {
            "item_id": ["US.cpi"],
            "target_month": [pd.Timestamp("2026-10-01")],
            "forecast": [0.2],
            "baseline_forecast": [0.3],
        }
    ).to_parquet(config.AUTOML_FORECASTS)
    base = {
        "picked": "ETS",
        "baseline": "Naive",
        "published": "Naive",
        "folds": [],
        "holdout_n_forecasts": 10,
        "holdout_improvement": 0.01,
        "fold_spread": 0.05,
    }
    for passed, label, rows in ((False, "explanation", 0), (True, "prediction", 1)):
        config.AUTOML_SUMMARY.write_text(json.dumps({**base, "passed": passed}))
        export.export_forecasts({})
        out = json.loads((tmp_path / "macro" / "forecasts.json").read_text())
        assert out["meta"]["verdict"]["finding"]["label"] == label
        assert len(out["data"]) == rows


def _writer(tmp_path):
    from pdi.common.export import write_table

    def write(*args, **kwargs):
        return write_table(*args, out_dir=tmp_path, **kwargs)

    return write
