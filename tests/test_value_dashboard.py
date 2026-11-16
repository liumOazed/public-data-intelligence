import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdi import site
from pdi.tech_value import config
from pdi.tech_value import dashboard as d

JS = Path(__file__).resolve().parents[1] / "site" / "tech-value" / "value-dashboard.js"


def _value(rows):
    df = pd.DataFrame(rows, columns=["kind", "model_key", "price_usd", "score", "confidence"])
    df["score_per_dollar"] = df["score"] / df["price_usd"]
    return df


SPEC_EXAMPLE = _value(
    [
        ("gpu", "nvidia a", 500, 5000, "normal"),
        ("gpu", "nvidia b", 600, 4800, "normal"),
        ("gpu", "nvidia c", 700, 6500, "normal"),
        ("gpu", "nvidia d", 800, 7000, "normal"),
    ]
)


def test_score_per_dollar_matches_the_worked_example():
    df = d.ranked(
        _value([("gpu", "x a", 500, 5000, "normal"), ("gpu", "x b", 750, 6000, "normal")])
    )
    assert df["score_per_dollar"].tolist() == [10.0, 8.0]


def test_ranked_drops_missing_zero_and_unscored():
    df = _value(
        [
            ("gpu", "ok", 100, 1000, "normal"),
            ("gpu", "no price", None, 1000, "normal"),
            ("gpu", "zero price", 0, 1000, "normal"),
            ("gpu", "no score", 100, None, "normal"),
        ]
    )
    assert d.ranked(df)["model_key"].tolist() == ["ok"]


def test_frontier_marks_the_dominated_model():
    on = d.frontier(SPEC_EXAMPLE)
    assert dict(zip(SPEC_EXAMPLE["model_key"], on, strict=False)) == {
        "nvidia a": True,
        "nvidia b": False,
        "nvidia c": True,
        "nvidia d": True,
    }


def test_identical_models_do_not_dominate_each_other():
    df = _value([("cpu", "a", 100, 50, "normal"), ("cpu", "b", 100, 50, "normal")])
    assert d.frontier(df).tolist() == [True, True]


def test_best_dominator_states_the_real_gaps():
    dom = d.best_dominator(SPEC_EXAMPLE, "nvidia b")
    assert dom["model_key"] == "nvidia a"
    assert dom["more_score_pct"] == pytest.approx(100 * (5000 / 4800 - 1))
    assert dom["lower_price_pct"] == pytest.approx(100 * (1 - 500 / 600))
    assert d.best_dominator(SPEC_EXAMPLE, "nvidia a") is None


def test_upgrade_differences_and_value_change():
    u = d.upgrade(SPEC_EXAMPLE, "nvidia a", "nvidia d")
    assert u["price_diff_usd"] == 300
    assert u["price_pct"] == pytest.approx(60)
    assert u["score_pct"] == pytest.approx(40)
    assert u["value_pct"] == pytest.approx(100 * ((7000 / 800) / (5000 / 500) - 1))


def test_upgrade_refuses_to_mix_kinds():
    df = _value([("gpu", "a", 100, 50, "normal"), ("cpu", "b", 100, 50, "normal")])
    with pytest.raises(ValueError):
        d.upgrade(df, "a", "b")


def test_brackets_are_half_open_and_report_counts():
    br = d.brackets(SPEC_EXAMPLE, [600, 750])
    assert br["n"].tolist() == [1, 2, 1]  # 500 | 600, 700 | 800
    assert br.loc[1, "best_value"] == "nvidia c"
    assert br.loc[1, "top_score_model"] == "nvidia c"
    assert br.loc[1, "median_price"] == 650
    assert math.isinf(br.loc[2, "high"])


def test_quantile_and_share_below():
    vals = [1, 2, 3, 4, 10]
    for q in (0.25, 0.5, 0.75, 0.9):
        assert d.quantile(vals, q) == pytest.approx(float(np.quantile(vals, q)))
    assert d.share_below(vals, 10) == 1.0
    assert d.share_below(vals, 1) == 0.0
    assert d.share_below([5], 5) is None


def _bench(n_gpu=3, n_cpu=2):
    rows = [("gpu", f"nvidia g{i}", 100 - i) for i in range(n_gpu)]
    rows += [("cpu", f"amd c{i}", 100 - i) for i in range(n_cpu)]
    return pd.DataFrame(rows, columns=["kind", "model_key", "n_runs"])


def test_price_checked_takes_the_most_benchmarked(monkeypatch):
    monkeypatch.setattr(config, "CRAWLER_MODELS_PER_KIND", 2)
    pc = d.price_checked(_bench())
    assert pc[pc["kind"] == "gpu"]["model_key"].tolist() == ["nvidia g0", "nvidia g1"]


def test_funnel_and_freshness_separate_the_universes():
    bench = _bench()
    value = _value([("gpu", "nvidia g0", 300, 3000, "normal")])
    history = pd.DataFrame(
        [
            ("2026-11-01", "gpu", "nvidia g1", 250.0, 1),  # seen once, then gone: stale
            ("2026-11-15", "gpu", "nvidia g0", 300.0, 2),
        ],
        columns=["date", "kind", "model_key", "price_usd", "n_listings"],
    )
    f = d.funnel(bench, value).set_index("kind")
    assert f.loc["gpu"].to_dict() == {
        "benchmarked": 3,
        "price_checked": 3,
        "priced": 1,
        "ranked": 1,
    }
    assert f.loc["cpu", "ranked"] == 0
    fr = d.freshness(bench, value, history)
    status = dict(zip(fr["model_key"], fr["status"], strict=False))
    assert status == {
        "nvidia g0": "fresh",
        "nvidia g1": "stale",
        "nvidia g2": "never priced",
        "amd c0": "never priced",
        "amd c1": "never priced",
    }


def test_history_summary_reports_gaps_and_gates_trends():
    history = pd.DataFrame(
        [
            ("2026-11-01", "gpu", "a", 100.0, 1),
            ("2026-11-02", "gpu", "a", 110.0, 1),
            ("2026-11-05", "gpu", "a", 90.0, 1),
        ],
        columns=["date", "kind", "model_key", "price_usd", "n_listings"],
    )
    hs = d.history_summary(history)
    assert hs["gaps"] == ["2026-11-03", "2026-11-04"]
    assert hs["span_days"] == 4 and not hs["trend_ok"]
    m = hs["models"][0]
    # Only observed points count: three observations, no interpolated days.
    assert m["n_obs"] == 3
    assert (m["low"], m["median"], m["high"]) == (90.0, 100.0, 110.0)
    assert m["change_usd"] == -10.0 and m["change_pct"] == pytest.approx(-10.0)
    assert not m["trend_ok"]


def test_insights_hold_back_market_claims_on_small_samples():
    value = SPEC_EXAMPLE.copy()  # 4 models: below the minimum for market insights
    items = d.insights(_bench(), value, pd.DataFrame())
    assert {i["topic"] for i in items} == {"Coverage"}


def test_insights_on_published_data_quote_the_real_numbers():
    value, _ = site.load("tech-value", "value")
    bench, _ = site.load("tech-value", "benchmarks")
    history, _ = site.load("tech-value", "price_history")
    if value.empty:
        pytest.skip("no published value table")
    items = d.insights(bench, value, history)
    rk = d.ranked(value)
    for kind in ("gpu", "cpu"):
        part = rk[rk["kind"] == kind]
        best = part.loc[part["score_per_dollar"].idxmax()]
        text = " ".join(i["text"] for i in items if i["kind"] == kind)
        assert d.display_name(best["model_key"]) in text
    # Two days of history: no price trend sentences.
    assert not any(i["topic"] == "Prices" for i in items)


def test_display_name():
    assert d.display_name("nvidia rtx 5060 ti") == "NVIDIA RTX 5060 Ti"
    assert d.display_name("intel core i7-14700kf") == "Intel Core i7-14700KF"
    assert d.display_name("amd ryzen 7 9800x3d") == "AMD Ryzen 7 9800X3D"
    assert d.display_name("amd rx 9070 xt") == "AMD RX 9070 XT"


def test_payload_is_json_and_ranked_only():
    value = pd.concat(
        [SPEC_EXAMPLE, _value([("gpu", "nvidia none", None, 100, "low")])], ignore_index=True
    )
    p = d.payload(_bench(), value, pd.DataFrame())
    json.dumps(p)
    assert [m["model_key"] for m in p["models"]] == SPEC_EXAMPLE["model_key"].tolist()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_browser_calculations_match_python():
    value, _ = site.load("tech-value", "value")
    if value.empty:
        pytest.skip("no published value table")
    rk = d.ranked(value)
    cases = []
    for kind in ("gpu", "cpu"):
        part = rk[rk["kind"] == kind].reset_index(drop=True)
        rows = part[["kind", "model_key", "price_usd", "score", "score_per_dollar"]].to_dict(
            "records"
        )
        cases.append({"rows": rows, "edges": d.DEFAULT_BRACKETS[kind]})
    script = f"""
const c = require({json.dumps(str(JS))});
const cases = {json.dumps(cases)};
const out = cases.map(({{rows, edges}}) => ({{
  frontier: c.frontier(rows),
  dom: rows.map((r) => c.bestDominator(rows, r.model_key)),
  q: [0.25, 0.5, 0.75, 0.9].map((q) => c.quantile(rows.map((r) => r.score), q)),
  below: rows.map((r) => c.shareBelow(rows.map((x) => x.score), r.score)),
  brackets: c.brackets(rows, edges).map((b) => [b.n, b.best_value || null, b.median_score || null]),
  names: rows.map((r) => c.displayName(r.model_key)),
}}));
console.log(JSON.stringify(out));
"""
    res = json.loads(
        subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    )
    for case, got in zip(cases, res, strict=False):
        part = pd.DataFrame(case["rows"])
        assert got["frontier"] == d.frontier(part).tolist()
        for key, dom in zip(part["model_key"], got["dom"], strict=False):
            want = d.best_dominator(part, key)
            if want is None:
                assert dom is None
            else:
                assert dom["model_key"] == want["model_key"]
                assert dom["more_score_pct"] == pytest.approx(want["more_score_pct"])
                assert dom["lower_price_pct"] == pytest.approx(want["lower_price_pct"])
        for q, v in zip((0.25, 0.5, 0.75, 0.9), got["q"], strict=False):
            assert v == pytest.approx(d.quantile(part["score"], q))
        assert got["below"] == pytest.approx(
            [d.share_below(part["score"].tolist(), s) for s in part["score"]]
        )
        br = d.brackets(part, case["edges"])
        want_br = [
            [
                int(r["n"]),
                r.get("best_value") if r["n"] else None,
                r.get("median_score") if r["n"] else None,
            ]
            for r in br.to_dict("records")
        ]
        assert [b[:2] for b in got["brackets"]] == [b[:2] for b in want_br]
        for g, w in zip(got["brackets"], want_br, strict=False):
            if w[2] is not None:
                assert g[2] == pytest.approx(w[2])
        assert got["names"] == [d.display_name(k) for k in part["model_key"]]
