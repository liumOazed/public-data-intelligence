import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from pdi import unusual
from pdi.common import findings
from pdi.procurement import export as procurement_export
from pdi.tech_value import export as tech_export
from pdi.trade import export as trade_export

NOW = datetime(2026, 11, 23, 6, 0, tzinfo=timezone.utc)


def _item(module="macro", subject="US.x", label="observation", test_passed=None, period="2026-10"):
    row = findings.item(
        module=module, subject=subject, period=f"{period}-01", sentence="s", rule="r",
        measure=3.0, threshold=2.5, finding=findings.observation("o"), page="p.qmd",
        table="t.json",
    )  # fmt: skip
    row.update(label=label, test_passed=test_passed)
    return row


def _write(site_data, module, rows):
    d = site_data / module
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "columns": findings.ITEM_COLUMNS,
            "generated_at": NOW.isoformat(),
            "rule_text": "a rule.",
        },  # fmt: skip
        "data": rows,
    }
    (d / "findings.json").write_text(json.dumps(payload))


def test_item_refuses_a_prediction_without_a_passed_test():
    pred = findings.finding(findings.PREDICTION, "p")
    with pytest.raises(ValueError):
        findings.item(module="m", subject="s", period="2026-10-01", sentence="", rule="r",
                      measure=1, threshold=1, finding=pred, page="", table="")  # fmt: skip
    ok = findings.item(module="m", subject="s", period="2026-10-01", sentence="", rule="r",
                       measure=1, threshold=1, finding=pred, page="", table="",
                       test_passed=True)  # fmt: skip
    assert ok["label"] == "prediction" and ok["key"] == "m|s|r|2026-10-01"


def test_build_stops_on_an_untested_prediction(tmp_path):
    _write(tmp_path, "macro", [_item(label="prediction")])
    with pytest.raises(unusual.UntestedPrediction):
        unusual.build(tmp_path, {}, NOW)
    _write(tmp_path, "macro", [_item(label="prediction", test_passed=True)])
    items, meta, _ = unusual.build(tmp_path, {}, NOW)
    assert meta["predictions"] == 1 and len(items) == 1


def test_build_stops_on_an_unknown_label(tmp_path):
    _write(tmp_path, "macro", [_item(label="hunch")])
    with pytest.raises(unusual.UntestedPrediction):
        unusual.build(tmp_path, {}, NOW)


def test_first_build_is_standing_and_later_items_are_new(tmp_path):
    _write(tmp_path, "macro", [_item()])
    items, meta, seen = unusual.build(tmp_path, {}, NOW)
    assert meta["first_build"] and not items["new"].any()

    _write(tmp_path, "macro", [_item(), _item(subject="VN.y")])
    later = NOW + timedelta(days=1)
    items, meta, seen = unusual.build(tmp_path, seen, later)
    assert not meta["first_build"]
    assert items.set_index("subject")["new"].to_dict() == {"VN.y": True, "US.x": False}

    # A week and a day later the second item is still flagged, but no longer new.
    items, meta, _ = unusual.build(tmp_path, seen, later + timedelta(days=8))
    assert meta["new_items"] == 0 and len(items) == 2


def test_missing_module_is_reported_not_fatal(tmp_path):
    _write(tmp_path, "macro", [])
    items, meta, _ = unusual.build(tmp_path, {}, NOW)
    status = {m["module"]: m for m in meta["modules"]}
    assert items.empty and status["trade"]["status"] == "not exported yet"


def _seasonal(years=11, spike=None):
    idx = pd.date_range("2015-01-01", periods=12 * years, freq="MS")
    rng = np.random.default_rng(0)
    pattern = np.tile(np.linspace(0.9, 1.1, 12), years)
    values = 1e9 * pattern * (1 + 0.01 * rng.standard_normal(len(idx)))
    if spike:
        values[-1] *= spike
    return idx, values


def test_trade_flags_only_an_unusual_latest_month():
    idx, values = _seasonal(spike=1.4)
    totals = pd.DataFrame({"partner": "VN", "period": idx, "value": values})
    items, scored = trade_export.unusual_items(totals)
    assert len(items) == 1 and items[0]["label"] == "observation"
    assert "Vietnam rose" in items[0]["sentence"]

    idx, values = _seasonal()
    calm = pd.DataFrame({"partner": "VN", "period": idx, "value": values})
    assert trade_export.unusual_items(calm)[0] == []


def test_procurement_scores_the_latest_settled_month_only():
    idx, values = _seasonal(spike=0.3)
    by_month = pd.DataFrame({"period": idx, "obligations": values, "provisional": False})
    assert len(procurement_export.unusual_items(by_month)) == 1
    # The same collapse in a provisional month is not scored.
    by_month.loc[by_month.index[-1], "provisional"] = True
    assert procurement_export.unusual_items(by_month) == []


def test_price_moves_of_five_percent_or_more_within_seven_days():
    h = pd.DataFrame({
        "date": ["2026-11-01", "2026-11-08", "2026-11-15", "2026-11-10", "2026-11-15"],
        "kind": "gpu", "model_key": ["a", "a", "a", "b", "b"],
        "price_usd": [50.0, 100.0, 94.0, 100.0, 103.0], "n_listings": 1,
    })  # fmt: skip
    items = tech_export.unusual_items(h)
    assert [i["subject"] for i in items] == ["a"]
    assert "fell 6.0%" in items[0]["sentence"] and "8 Nov" in items[0]["sentence"]
