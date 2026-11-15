from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "synthetic"
LIVE = Path(__file__).parent / "fixtures" / "live"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


def synthetic_spec(n_processors: int = 240, seed: int = 0, signal: bool = True) -> pd.DataFrame:
    """A made-up tidy SPEC table with a known data-generating process.

    With signal=True, score depends on cache and on a per-year efficiency gain, which the
    cores-times-clock baseline cannot see. With signal=False, score is cores x clock plus
    noise, so nothing should beat the baseline.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_processors):
        avail = pd.Timestamp("2018-01-01") + pd.DateOffset(days=int(rng.integers(0, 365 * 8)))
        cores_per_chip = int(rng.choice([8, 16, 24, 32, 48, 64, 96]))
        mhz = float(rng.choice([2000, 2400, 2800, 3200, 3600]))
        vendor = str(rng.choice(["Intel", "AMD"]))
        l3 = float(rng.choice([1.0, 2.0, 4.0]))
        l2 = float(rng.choice([0.5, 1.0, 2.0]))
        years = (avail - pd.Timestamp("2017-01-01")).days / 365.25
        proc_effect = rng.normal(0, 0.03)
        for _ in range(int(rng.integers(3, 7))):
            chips = int(rng.choice([1, 2]))
            cores = cores_per_chip * chips
            log_score = -6.0 + np.log(cores) + np.log(mhz)
            if signal:
                log_score += -0.1 * np.log(cores) + 0.15 * np.log1p(l3) + 0.06 * years
                log_score += 0.05 * (vendor == "AMD") + proc_effect
            log_score += rng.normal(0, 0.03)
            rows.append(
                {
                    "benchmark": "CINT2017rate",
                    "hardware_vendor": "Synthetic",
                    "system": f"Synthetic system {i}",
                    "processor": f"{vendor} Synthetic {i:04d}",
                    "vendor": vendor,
                    "family": "Synthetic",
                    "cores": cores,
                    "chips": chips,
                    "threads_per_core": 2,
                    "mhz": mhz,
                    "l2_mb_per_core": l2,
                    "l3_mb_per_core": l3,
                    "memory_gb": 256.0,
                    "memory_text": "256 GB",
                    "os": "Synthetic OS",
                    "compiler": "Synthetic compiler",
                    "hw_avail": avail.normalize().replace(day=1),
                    "test_date": avail.normalize().replace(day=1),
                    "base_score": float(np.exp(log_score)),
                    "peak_score": float(np.exp(log_score) * 1.03),
                    "disclosure_url": None,
                    "retrieved_at": "2026-01-01T00:00:00+00:00",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def spec_signal() -> pd.DataFrame:
    return synthetic_spec(signal=True)


@pytest.fixture
def spec_noise() -> pd.DataFrame:
    return synthetic_spec(signal=False)


@pytest.fixture
def live() -> Path:
    """Slices of real responses, captured by `python -m pdi.samples`."""
    return LIVE
