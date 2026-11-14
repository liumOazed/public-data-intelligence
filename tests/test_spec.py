import pandas as pd
import pandera.pandas as pa
import pytest

from pdi.tech_value import spec


@pytest.mark.parametrize(
    "text, cores_per_chip, expected",
    [
        ("32 KB I + 48 KB D on chip per core", 56, 80 / 1024),
        ("2 MB I+D on chip per core", 56, 2.0),
        ("1280 KB I+D on chip per core", 32, 1.25),
        ("105 MB I+D on chip per chip", 56, 105 / 56),
        ("16 MB I+D on chip per 8 cores", 64, 2.0),
        ("None", 8, 0.0),
        ("", 8, None),
        (None, 8, None),
    ],
)
def test_cache_mb_per_core(text, cores_per_chip, expected):
    got = spec.cache_mb_per_core(text, cores_per_chip)
    assert got == pytest.approx(expected) if expected is not None else got is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("512 GB (16 x 32 GB 2Rx8 PC5-4800B-R)", 512.0),
        ("1 TB (16 x 64 GB 2Rx4 PC4-3200AA-R)", 1024.0),
        ("unknown", None),
    ],
)
def test_memory_gb(text, expected):
    assert spec.memory_gb(text) == expected


def test_parse_sample(fixtures):
    df = spec.parse((fixtures / "spec_rint2017_sample.csv").read_bytes())
    # The withdrawn result (score 0) is dropped, not repaired.
    assert len(df) == 3
    row = df[df["processor"] == "Intel Xeon Platinum 8480+"].iloc[0]
    assert row["cores"] == 112 and row["chips"] == 2 and row["mhz"] == 2000
    assert row["base_score"] == 900 and row["peak_score"] == 930
    assert row["hw_avail"] == pd.Timestamp("2023-01-01")
    assert row["l3_mb_per_core"] == pytest.approx(105 / 56)
    assert row["vendor"] == "Intel" and row["family"] == "Xeon Platinum"
    assert row["disclosure_url"].startswith("https://www.spec.org/cpu2017/results/")
    # Double spaces in the processor name are collapsed so one CPU is one group.
    assert "Intel Xeon Gold 6338" in set(df["processor"])
    assert set(df["vendor"]) == {"Intel", "AMD"}


def test_parse_rejects_changed_columns(fixtures):
    text = (fixtures / "spec_rint2017_sample.csv").read_text().replace('"Baseline"', '"Base"')
    with pytest.raises(ValueError, match="missing columns"):
        spec.parse(text)


def test_parse_rejects_wrong_units(fixtures):
    # MHz delivered as GHz must stop the run, not flow into the model.
    text = (fixtures / "spec_rint2017_sample.csv").read_text().replace('"2000"', '"2.0"')
    with pytest.raises((ValueError, pa.errors.SchemaError)):
        spec.parse(text)
