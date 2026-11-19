"""Shared helpers for the over- and under-performer rankings (CPU and GPU).

Everything here works on held-out predictions that already exist: it does not fit models.

* ``related_group`` keeps submissions of the same silicon in one held-out group, so a chip's
  single-socket "P" variant cannot leak into the fit that scores it.
* ``loo_interval`` turns the spread of other chips' held-out residuals into a prediction range
  for each chip, so the range is validated rather than assumed.
* ``extrapolation_flags`` names the specifications on which a chip lies outside every other chip
  the models were trained on.
* ``validation_table`` reports typical error by model and by hardware slice, next to a simple
  baseline.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

INTERVAL_QUANTILES = (0.10, 0.90)  # an 80% range
_P_SUFFIX = re.compile(r"(\d)P$")


def related_group(processor: str) -> str:
    """Name of the silicon a processor shares with its variants.

    Single-socket parts such as AMD EPYC "9454P" use the same die as the part without the P,
    so the two are held out together. A P part with no plain sibling stays a group of its own.
    """
    return _P_SUFFIX.sub(r"\1", str(processor).strip())


def loo_interval(resid_log: np.ndarray, q: tuple[float, float] = INTERVAL_QUANTILES):
    """For each chip, quantiles of every other chip's held-out residual (in logs)."""
    r = np.asarray(resid_log, dtype=float)
    lo = np.empty_like(r)
    hi = np.empty_like(r)
    for i in range(len(r)):
        others = np.delete(r, i)
        others = others[np.isfinite(others)]
        if len(others) < 5:
            lo[i] = hi[i] = np.nan
            continue
        lo[i], hi[i] = np.quantile(others, q)
    return lo, hi


def extrapolation_flags(
    chips: pd.DataFrame, results: pd.DataFrame, features: dict[str, str]
) -> pd.Series:
    """Comma-separated labels of the features on which a chip lies outside all other chips.

    ``chips`` is indexed by processor with one value per feature; ``results`` holds every result
    used for fitting, with a ``processor`` column. A chip is compared with the range of the
    *other* chips, which is the range the held-out models were trained on.
    """
    per_chip = results.groupby("processor")[list(features)].median()
    out = {}
    for proc, row in chips.iterrows():
        others = per_chip.drop(index=proc, errors="ignore")
        hits = []
        for col, label in features.items():
            v = row.get(col)
            if v is None or not np.isfinite(v):
                continue
            lo, hi = others[col].min(), others[col].max()
            if v > hi * 1.0001:
                hits.append(f"{label} above training range")
            elif v < lo * 0.9999:
                hits.append(f"{label} below training range")
        out[proc] = "; ".join(hits)
    return pd.Series(out, dtype=object)


def _abs_pct(resid_log: pd.Series) -> pd.Series:
    return np.expm1(resid_log.abs()) * 100


def validation_table(
    chips: pd.DataFrame, models: dict[str, str], slices: dict[str, str]
) -> pd.DataFrame:
    """Typical held-out error per model, overall and for each value of each slice column.

    ``chips`` has one row per ranked chip with a median residual (log) column per model and a
    ``<col>_covered`` boolean for whether the measured score fell in that model's 80% range.
    """
    rows = []
    groups = [("All", "All", chips)]
    for slice_name, col in slices.items():
        for value, part in chips.groupby(col, dropna=True):
            groups.append((slice_name, str(value), part))
    for slice_name, value, part in groups:
        for model_name, col in models.items():
            r = part[col].dropna()
            if r.empty:
                continue
            covered = part.get(f"{col}_covered")
            rows.append(
                {
                    "slice": slice_name,
                    "group": value,
                    "model": model_name,
                    "n_chips": int(len(r)),
                    "median_abs_pct_error": float(_abs_pct(r).median()),
                    "mean_abs_log_error": float(r.abs().mean()),
                    "range_coverage_pct": (
                        float(covered.dropna().mean() * 100)
                        if covered is not None and covered.notna().any()
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def add_model_summary(chips: pd.DataFrame, cols: dict[str, str], score_col: str) -> pd.DataFrame:
    """Add predicted scores, 80% ranges, range coverage and model spread to a chip table.

    ``cols`` maps a short model key (``ols``, ``structural``, ``nonlinear``) to the column with
    that model's median held-out residual (log). Predicted score = measured / exp(residual).
    """
    out = chips.copy()
    for key, col in cols.items():
        r = out[col].to_numpy(dtype=float)
        pred = out[score_col].to_numpy(dtype=float) / np.exp(r)
        lo, hi = loo_interval(r)
        out[f"pred_{key}"] = pred
        out[f"pred_{key}_lo"] = pred * np.exp(lo)
        out[f"pred_{key}_hi"] = pred * np.exp(hi)
        covered = (r >= lo) & (r <= hi)
        out[f"{col}_covered"] = np.where(np.isfinite(lo), covered, np.nan)
    pct = np.column_stack([np.expm1(out[c].to_numpy(dtype=float)) * 100 for c in cols.values()])
    out["model_spread_pts"] = pct.max(axis=1) - pct.min(axis=1)
    return out
