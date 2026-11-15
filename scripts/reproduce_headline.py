"""Reproduce one Macro Monitor number from scratch, without this repository's code.

    pip install pandas
    python reproduce_headline.py            # downloads CPIAUCSL from FRED's public CSV
    python reproduce_headline.py file.csv   # or reads a CSV you saved from FRED yourself

It prints the latest month's move in US consumer prices (CPI, all items, seasonally adjusted),
the usual move, and how many spreads apart they are. Compare the last line with the
"US.cpi" row of site/data/v1/macro/series.json (columns move, expected_move and z) for a run on
the same day. The rule, written down before any flag was computed:

    move     = 100 x change in the log of the index (close to the percent change)
    usual    = median move over the 120 calendar months before the month being scored
               (months with no published value are skipped)
    spread   = 1.4826 x median absolute deviation of (move - usual) over those 120 months
    spreads  = (move - usual) / spread; 2.5 or more is a flag
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL&cosd=1990-01-01"


def headline(csv_path_or_url: str = URL) -> dict:
    raw = pd.read_csv(csv_path_or_url)
    values = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.to_datetime(raw.iloc[:, 0])).dropna()
    # A month the publisher skipped stays a gap: October 2025 has no CPI (the US government
    # shutdown), so neither October's nor November's move exists. Without this line the
    # September-to-November change would count as one month's move.
    series = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="MS"))
    move = 100 * np.log(series).diff()
    latest = move.index[-1]
    past = move.iloc[-121:-1].dropna()  # the 120 calendar months before the latest one
    usual = float(np.median(past))
    spread = 1.4826 * float(np.median(np.abs(past - usual)))
    return {
        "month": f"{latest:%Y-%m}",
        "move": float(move.iloc[-1]),
        "usual": usual,
        "spread": spread,
        "spreads": (float(move.iloc[-1]) - usual) / spread,
    }


if __name__ == "__main__":
    h = headline(sys.argv[1] if len(sys.argv) > 1 else URL)
    print(
        f"US consumer prices, {h['month']}: moved {h['move']:+.3f}% against a usual "
        f"{h['usual']:+.3f}% (spread {h['spread']:.3f}), so {h['spreads']:.2f} spreads."
    )
