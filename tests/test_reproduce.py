"""The stand-alone reproduction script must give the pipeline's own number."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from pdi.macro import config, flags

SCRIPT = Path(__file__).parents[1] / "scripts" / "reproduce_headline.py"
OBS = config.OBSERVATIONS


def _script():
    spec = importlib.util.spec_from_file_location("reproduce_headline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not OBS.exists(), reason="no stored observations")
def test_script_matches_pipeline(tmp_path):
    obs = pd.read_parquet(OBS)
    cpi = obs[obs["series_id"] == "US.cpi"].sort_values("period")
    csv = tmp_path / "CPIAUCSL.csv"
    pd.DataFrame({"observation_date": cpi["period"].dt.strftime("%Y-%m-%d"),
                  "CPIAUCSL": cpi["value"]}).to_csv(csv, index=False)  # fmt: skip
    got = _script().headline(str(csv))

    values = flags.monthly_values(obs, "US.cpi")
    scored = flags.surprises(flags.moves(values, "log"), "sa").dropna(subset=["z"])
    want = scored.iloc[-1]
    assert got["month"] == f"{scored.index[-1]:%Y-%m}"
    assert got["move"] == pytest.approx(want["move"])
    assert got["usual"] == pytest.approx(want["expected"])
    assert got["spreads"] == pytest.approx(want["z"])
