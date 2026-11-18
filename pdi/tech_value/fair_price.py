"""Fair price: what a part would cost if price followed benchmark performance alone.

For each kind of part (processors and graphics cards separately, because their Blender scores
are on different scales) the logarithm of price is modelled from the logarithm of the score.
A part priced below its prediction is cheap for what it does; one priced above is paying for
something the benchmark does not measure, or is simply overpriced.

The sample is a few dozen parts, so the rules are strict and simple:

* A naive baseline is always on the leaderboard: every part costs the same per point of score.
* Every model is scored by leave-one-out prediction, so a part's fair price always comes from
  a fit that never saw that part.
* Only parts with at least two listings are used for fitting. A single listing is often one
  marketplace seller's asking price for a discontinued part, and would bend the line.
* A model replaces the baseline only when its leave-one-out error is lower by more than
  twice the standard error of the gain. Otherwise the baseline sets the fair price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    QuantileRegressor,
    Ridge,
    TheilSenRegressor,
)
from sklearn.neighbors import KNeighborsRegressor

from pdi.common import findings
from pdi.tech_value import config, names

BASELINE = "Naive baseline (same price per point)"
Fit = Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]]

_WS_LINES = {"nvidia_ws", "amd_ws", "intel_arc_pro", "amd_threadripper"}
_WS_RE = r"\b(?:threadripper|pro|xeon|epyc)\b"


def _ordinal_gen_lag(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (model_gen_lag, ordinal_gen_rank) for each row in df.

    `ordinal_gen_rank` is the exact position (0 = current generation, 1 = previous generation,
    2+ = legacy generations) within each product line's active generation list.
    """
    pairs = list(zip(df["kind"], df["model_key"], strict=True))
    line_gens = names.active_generations(pairs)
    ordinal = np.zeros(len(df), dtype=float)
    for idx, (line, gen) in enumerate(zip(df["line"], df["gen"], strict=True)):
        if gen <= 0 or line not in line_gens:
            continue
        gens = [g for g in line_gens[line] if not (line == "amd_ryzen" and g == 8)]
        eff_gen = 7 if (line == "amd_ryzen" and gen == 8) else gen
        rank = float(len(gens))
        for pos, g in enumerate(gens):
            if eff_gen >= g:
                rank = float(pos)
                break
        ordinal[idx] = rank
    gen_step = np.where(df["line"] == "nvidia_geforce", df["gen"] / 10.0, df["gen"].astype(float))
    temp = df.assign(_gs=gen_step)
    max_gen = temp.groupby("line")["_gs"].transform("max").to_numpy()
    model_lag = np.where(gen_step > 0, (max_gen - gen_step).clip(min=0.0), 0.0)
    return model_lag, ordinal


def prepare(table: pd.DataFrame) -> pd.DataFrame:
    """Priced, scored parts with the columns the models use."""
    if table.empty:
        return pd.DataFrame()
    df = table[table["score"].notna() & table["price_usd"].notna()].copy()
    df["log_price"] = np.log(df["price_usd"].astype(float))
    df["log_score"] = np.log(df["score"].astype(float))
    df["vendor"] = df["model_key"].str.split().str[0]
    lines = [names.product_line(k, m) for k, m in zip(df["kind"], df["model_key"], strict=True)]
    df["line"] = [item[0] if item else "other" for item in lines]
    df["gen"] = [item[1] if item else 0 for item in lines]
    df["is_workstation"] = (
        df["line"].isin(_WS_LINES) | df["model_key"].str.contains(_WS_RE, case=False, na=False)
    ).astype(float)
    df["is_x3d"] = df["model_key"].str.endswith("3d", na=False).astype(float)
    model_lag, ordinal_rank = _ordinal_gen_lag(df)
    df["gen_lag"] = model_lag
    df["gen_rank"] = ordinal_rank.astype(int)
    df["gen_stage"] = np.select(
        [df["gen_rank"] == 0, df["gen_rank"] == 1],
        ["Current gen (0)", "Previous gen (-1)"],
        default="Legacy gen (-2+)",
    )
    df["regime"] = np.select(
        [df["is_workstation"] > 0, df["is_x3d"] > 0],
        ["Workstation / Pro", "3D V-Cache (X3D)"],
        default="Consumer / Mainstream",
    )
    df["trusted"] = df["n_listings"] >= config.FAIR_PRICE_MIN_LISTINGS
    return df.reset_index(drop=True)


def _design(df: pd.DataFrame, vendors: list[str]) -> np.ndarray:
    cols = [np.ones(len(df)), df["log_score"].to_numpy()]
    cols += [(df["vendor"] == v).to_numpy(float) for v in vendors]
    return np.column_stack(cols)


def _structural_features(train: pd.DataFrame):
    mu = float(train["log_score"].mean())
    sd = float(train["log_score"].std()) or 1.0
    counts = train["vendor"].value_counts()
    vendors = sorted(counts[counts >= 2].index)[1:]

    def transform(df: pd.DataFrame) -> np.ndarray:
        z = (df["log_score"].to_numpy() - mu) / sd
        cols = [
            z,
            z**2,
            df["is_workstation"].to_numpy(float),
            df["is_x3d"].to_numpy(float),
            df["gen_lag"].to_numpy(float),
        ]
        cols += [(df["vendor"] == v).to_numpy(float) for v in vendors]
        return np.column_stack(cols)

    return transform


def fit_baseline(train: pd.DataFrame):
    a = float(np.median(train["log_price"] - train["log_score"]))
    return lambda df: a + df["log_score"].to_numpy()


def _fit_line(train: pd.DataFrame, with_vendor: bool):
    # The first vendor is the reference level; a vendor seen once cannot have its own level.
    counts = train["vendor"].value_counts()
    vendors = sorted(counts[counts >= 2].index)[1:] if with_vendor else []
    beta, *_ = np.linalg.lstsq(_design(train, vendors), train["log_price"].to_numpy(), rcond=None)
    return lambda df: _design(df, vendors) @ beta


def fit_line(train: pd.DataFrame):
    return _fit_line(train, with_vendor=False)


def fit_line_vendor(train: pd.DataFrame):
    return _fit_line(train, with_vendor=True)


def _make_structural_fit(estimator_fn: Callable[[], object]) -> Fit:
    def fit(train: pd.DataFrame):
        transform = _structural_features(train)
        est = estimator_fn()
        est.fit(transform(train), train["log_price"].to_numpy())
        return lambda df: est.predict(transform(df))

    return fit


def fit_structural_quantile(train: pd.DataFrame):
    return _make_structural_fit(
        lambda: QuantileRegressor(quantile=0.5, alpha=0.012, solver="highs")
    )(train)


def fit_structural_quantile_p20(train: pd.DataFrame):
    return _make_structural_fit(
        lambda: QuantileRegressor(quantile=0.2, alpha=0.012, solver="highs")
    )(train)


def fit_structural_quantile_p80(train: pd.DataFrame):
    return _make_structural_fit(
        lambda: QuantileRegressor(quantile=0.8, alpha=0.012, solver="highs")
    )(train)


def fit_structural_huber(train: pd.DataFrame):
    return _make_structural_fit(lambda: HuberRegressor(alpha=0.5, max_iter=500))(train)


def fit_forest(train: pd.DataFrame):
    vendors = sorted(train["vendor"].unique())

    def features(df: pd.DataFrame) -> np.ndarray:
        return _design(df, vendors)[:, 1:]

    forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0)
    forest.fit(features(train), train["log_price"].to_numpy())
    return lambda df: forest.predict(features(df))


ENSEMBLE_NAME = "Weighted Ensemble (Top-3 LOO)"
FLOOR_NAME = "L1 Quantile P20 (Low Listed Band)"
CEILING_NAME = "L1 Quantile P80 (High Listed Band)"

MODEL_SPECS: dict[str, tuple[str, str, Fit]] = {
    BASELINE: ("Baseline", "Central fair price (P50)", fit_baseline),
    "Straight line (log price on log score)": ("Linear (L2)", "Central fair price (P50)", fit_line),
    "Straight line with brand": ("Linear (L2)", "Central fair price (P50)", fit_line_vendor),
    "L1 structural quantile (regime + gen lag)": (
        "Quantile (L1)",
        "Central fair price (P50)",
        fit_structural_quantile,
    ),
    "Huber structural regression": (
        "Robust (Huber)",
        "Central fair price (P50)",
        fit_structural_huber,
    ),
    "Bayesian Ridge (posterior structural)": (
        "Bayesian Linear",
        "Central fair price (P50)",
        _make_structural_fit(lambda: BayesianRidge()),
    ),
    "Theil-Sen robust median-slope": (
        "Robust (Nonparametric)",
        "Central fair price (P50)",
        _make_structural_fit(lambda: TheilSenRegressor(random_state=0)),
    ),
    "ElasticNet (L1 + L2 structural)": (
        "Linear (L1+L2)",
        "Central fair price (P50)",
        _make_structural_fit(lambda: ElasticNet(alpha=0.015, l1_ratio=0.5, max_iter=2000)),
    ),
    "Lasso (L1 sparse structural)": (
        "Linear (L1)",
        "Central fair price (P50)",
        _make_structural_fit(lambda: Lasso(alpha=0.01, max_iter=2000)),
    ),
    "Ridge (L2 regularized structural)": (
        "Linear (L2)",
        "Central fair price (P50)",
        _make_structural_fit(lambda: Ridge(alpha=1.0)),
    ),
    "Gradient Boosted Trees (Huber loss)": (
        "Boosting (Huber)",
        "Nonlinear regime check",
        _make_structural_fit(
            lambda: GradientBoostingRegressor(
                loss="huber", n_estimators=80, max_depth=2, learning_rate=0.08, random_state=0
            )
        ),
    ),
    "Gradient Boosted Trees (L1 loss)": (
        "Boosting (L1)",
        "Nonlinear regime check",
        _make_structural_fit(
            lambda: GradientBoostingRegressor(
                loss="absolute_error",
                n_estimators=80,
                max_depth=2,
                learning_rate=0.08,
                random_state=0,
            )
        ),
    ),
    "Extra Trees (structural)": (
        "Bagging (Trees)",
        "Nonlinear regime check",
        _make_structural_fit(
            lambda: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=0)
        ),
    ),
    "Random forest": ("Bagging (Trees)", "Nonlinear regime check", fit_forest),
    "K-Nearest Neighbors (distance-weighted)": (
        "Local Instance",
        "Local peer check",
        _make_structural_fit(lambda: KNeighborsRegressor(n_neighbors=5, weights="distance")),
    ),
    FLOOR_NAME: (
        "Quantile (Pinball τ=0.20)",
        "Conditional 20th percentile (low listed band)",
        fit_structural_quantile_p20,
    ),
    CEILING_NAME: (
        "Quantile (Pinball τ=0.80)",
        "Conditional 80th percentile (high listed band)",
        fit_structural_quantile_p80,
    ),
}

MODELS: dict[str, Fit] = {name: spec[2] for name, spec in MODEL_SPECS.items()}


def _pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    diff = y - pred
    return np.maximum(q * diff, (q - 1.0) * diff)


def leave_one_out(train: pd.DataFrame, fit: Fit) -> np.ndarray:
    out = np.empty(len(train))
    for i in range(len(train)):
        rest = train.drop(train.index[i])
        out[i] = fit(rest)(train.iloc[[i]])[0]
    return out


def _add_market_signals(
    out: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach Step 2 market signals (B: Seller Spread, D: Frontier & Momentum, C: Lifecycle)."""
    df = out.copy()
    p = df["price_usd"].astype(float).to_numpy()
    s = df["score"].astype(float).to_numpy()
    n_list = df["n_listings"].astype(int).to_numpy()
    n_ret = df["n_retailers"].astype(int).to_numpy()

    med_list = (
        df["median_price_usd"].astype(float).to_numpy()
        if "median_price_usd" in df.columns
        else p.copy()
    )
    src_med = (
        df["source_median_price_usd"].astype(float).to_numpy()
        if "source_median_price_usd" in df.columns
        else med_list.copy()
    )
    ret_floor = (
        df["retail_floor_usd"].astype(float).to_numpy()
        if "retail_floor_usd" in df.columns
        else np.full(len(df), np.nan)
    )
    mkt_floor = (
        df["marketplace_floor_usd"].astype(float).to_numpy()
        if "marketplace_floor_usd" in df.columns
        else np.full(len(df), np.nan)
    )

    # B. Cross-Seller & Listing Spread
    listing_spread_pct = np.where(
        (n_list >= 2) & (med_list > 0),
        np.maximum(0.0, (med_list - p) / med_list * 100.0),
        0.0,
    )
    source_spread_pct = np.where(
        (n_ret >= 2) & (src_med > 0),
        np.maximum(0.0, (src_med - p) / src_med * 100.0),
        0.0,
    )

    has_ret = ~np.isnan(ret_floor)
    has_mkt = ~np.isnan(mkt_floor)
    channel_signal = np.select(
        [
            has_ret & has_mkt & (ret_floor <= mkt_floor * 0.98),
            has_ret & has_mkt & (mkt_floor < ret_floor * 0.98),
            has_ret & has_mkt,
            has_ret & ~has_mkt,
            ~has_ret & has_mkt & (n_list >= 2),
        ],
        [
            "Retail undercuts marketplace",
            "Marketplace undercuts retail",
            "Retail & marketplace aligned",
            "Primary retail only",
            "Marketplace only (multi-listing)",
        ],
        default="Single marketplace listing",
    )

    # D. Two-Threshold Frontier Distance (Pareto Frontier vs. Regime Best Value Leader)
    drop_frontier_usd = np.zeros(len(df), dtype=float)
    drop_frontier_pct = np.zeros(len(df), dtype=float)
    blocking_rival = []

    for i in range(len(df)):
        # A rival j dominates or blocks i if score_j >= score_i and price_j < price_i
        faster_mask = (s >= s[i]) & (np.arange(len(df)) != i)
        if np.any(faster_mask):
            faster_idx = np.where(faster_mask)[0]
            best_j = faster_idx[np.argmin(p[faster_idx])]
            if p[best_j] < p[i]:
                diff = p[i] - p[best_j]
                drop_frontier_usd[i] = diff
                drop_frontier_pct[i] = (diff / p[i]) * 100.0
                blocking_rival.append(str(df.iloc[best_j]["model_key"]))
            else:
                blocking_rival.append("On Pareto frontier")
        else:
            blocking_rival.append("On Pareto frontier")

    spd = s / np.maximum(p, 1e-6)
    df["_spd"] = spd
    best_spd_by_regime = df.groupby("regime")["_spd"].transform("max").to_numpy()
    target_best_val_price = s / np.maximum(best_spd_by_regime, 1e-6)
    drop_best_val_usd = np.maximum(0.0, p - target_best_val_price)
    drop_best_val_pct = np.where(p > 0, (drop_best_val_usd / p) * 100.0, 0.0)
    df = df.drop(columns=["_spd"])

    # D. Sticker-Price Momentum (7-day observed price change, decoupled from model recalibration)
    price_change_7d_pct = np.zeros(len(df), dtype=float)
    if history is not None and not history.empty:
        kind_val = str(df["kind"].iloc[0]) if len(df) else ""
        h_kind = history[history["kind"] == kind_val].copy()
        if not h_kind.empty:
            h_kind["date"] = pd.to_datetime(h_kind["date"])
            h_kind = h_kind.sort_values("date")
            change_map: dict[str, float] = {}
            for m_key, g in h_kind.groupby("model_key"):
                p_now = float(g.iloc[-1]["price_usd"])
                prior = g[g["date"] < g.iloc[-1]["date"]]
                if not prior.empty and float(prior.iloc[0]["price_usd"]) > 0:
                    p_prev = float(prior.iloc[0]["price_usd"])
                    change_map[str(m_key)] = (p_now / p_prev - 1.0) * 100.0
            price_change_7d_pct = np.array(
                [change_map.get(str(m), 0.0) for m in df["model_key"]], dtype=float
            )

    # C. Generational Hand-Off & Market Wind Diagnostic (Observation-based, non-speculative)
    gap = df["gap_pct"].astype(float).to_numpy()
    gen_rank = df["gen_rank"].astype(int).to_numpy()
    lifecycle_signal = np.select(
        [
            (gen_rank >= 1) & (gap <= -config.FAIR_PRICE_GAP_PCT) & (n_list >= 2),
            (gen_rank >= 1) & (gap >= config.FAIR_PRICE_GAP_PCT) & (~has_ret),
            (gen_rank >= 1) & (gap >= config.FAIR_PRICE_GAP_PCT) & has_ret,
            (gen_rank == 0) & (gap <= -config.FAIR_PRICE_GAP_PCT),
            (drop_frontier_pct > 0) & (drop_frontier_pct <= 8.0),
        ],
        [
            "Prev-gen clearance sweet spot",
            "Thin-supply legacy premium (marketplace only)",
            "Legacy specialty / socket holdout premium",
            "Current-gen value leader",
            "Near Pareto frontier (<8% drop needed)",
        ],
        default="Normal generational curve",
    )

    df["listing_spread_pct"] = listing_spread_pct
    df["source_spread_pct"] = source_spread_pct
    df["channel_signal"] = channel_signal
    df["drop_to_frontier_usd"] = drop_frontier_usd
    df["drop_to_frontier_pct"] = drop_frontier_pct
    df["blocking_rival"] = blocking_rival
    df["drop_to_best_value_usd"] = drop_best_val_usd
    df["drop_to_best_value_pct"] = drop_best_val_pct
    df["price_change_7d_pct"] = price_change_7d_pct
    df["lifecycle_signal"] = lifecycle_signal
    return df


@dataclass
class Result:
    leaderboard: pd.DataFrame
    parts: pd.DataFrame
    verdicts: dict


def evaluate(table: pd.DataFrame, history: pd.DataFrame | None = None) -> Result:
    df = prepare(table)
    boards, parts, verdicts = [], [], {}
    central_candidates = [
        name for name, (_, role, _) in MODEL_SPECS.items() if name not in {FLOOR_NAME, CEILING_NAME}
    ]
    for kind in ("gpu", "cpu"):
        part = df[df["kind"] == kind] if len(df) else df
        train = part[part["trusted"]] if len(part) else part
        if len(train) < config.FAIR_PRICE_MIN_PARTS:
            verdicts[kind] = {
                "statement": f"Only {len(train)} parts have two or more listings, fewer than "
                f"the {config.FAIR_PRICE_MIN_PARTS} needed to fit a fair price.",
                "n_parts": int(len(train)),
                "finding": findings.observation(
                    "Too few parts to fit or test a fair price; only the measured prices are shown."
                ),
            }
            continue
        y = train["log_price"].to_numpy()
        oof = {name: leave_one_out(train, fit) for name, fit in MODELS.items()}
        err = {name: np.abs(y - pred) for name, pred in oof.items()}

        # Build Top-3 LOO Ensemble from non-baseline central candidates
        non_base = [n for n in central_candidates if n != BASELINE]
        top3 = sorted(non_base, key=lambda n: err[n].mean())[:3]
        oof[ENSEMBLE_NAME] = np.mean([oof[n] for n in top3], axis=0)
        err[ENSEMBLE_NAME] = np.abs(y - oof[ENSEMBLE_NAME])

        all_central = central_candidates + [ENSEMBLE_NAME]
        best = min(all_central, key=lambda name: err[name].mean())
        gain = err[BASELINE] - err[best]
        se = float(gain.std(ddof=1) / np.sqrt(len(gain))) if best != BASELINE else 0.0
        beats = best != BASELINE and float(gain.mean()) > 2 * se
        # A model that does not clearly beat the baseline is not used: the simpler rule wins.
        picked = best if beats else BASELINE

        # A. Leave-one-out empirical coverage test for the P20-P80 conditional band (target = 60%)
        oof_lo = np.minimum(oof[FLOOR_NAME], oof[picked] - 0.05)
        oof_hi = np.maximum(oof[CEILING_NAME], oof[picked] + 0.05)
        loo_in_band = (y >= oof_lo) & (y <= oof_hi)
        loo_coverage_pct = float(loo_in_band.mean() * 100.0)
        rel_band_width_pct = float(
            np.median((np.exp(oof_hi) - np.exp(oof_lo)) / np.exp(oof[picked])) * 100.0
        )

        # Rank central models by MAE and include boundary quantile models with pinball loss
        ordered_names = sorted(
            list(MODELS.keys()) + [ENSEMBLE_NAME],
            key=lambda n: (n in {FLOOR_NAME, CEILING_NAME}, err[n].mean()),
        )
        for rank_idx, name in enumerate(ordered_names, start=1):
            family, role = (
                ("Ensemble (Top-3)", "Consensus fair price (Top-3 blend)")
                if name == ENSEMBLE_NAME
                else (MODEL_SPECS[name][0], MODEL_SPECS[name][1])
            )
            g = err[BASELINE] - err[name]
            g_se = float(g.std(ddof=1) / np.sqrt(len(g))) if name != BASELINE else 0.0
            row_beats = (
                name not in {BASELINE, FLOOR_NAME, CEILING_NAME} and float(g.mean()) > 2 * g_se
            )
            q_val = 0.2 if name == FLOOR_NAME else (0.8 if name == CEILING_NAME else 0.5)
            pinball = float(_pinball_loss(y, oof[name], q_val).mean())
            boards.append(
                {
                    "kind": kind,
                    "rank": rank_idx,
                    "model": name,
                    "family": family,
                    "question": role,
                    "n_parts": int(len(train)),
                    "mae_log": float(err[name].mean()),
                    "pinball_loss": pinball,
                    "median_abs_pct_error": float(np.median(np.expm1(err[name])) * 100),
                    "gain_vs_baseline": float(g.mean()),
                    "beats_baseline": bool(row_beats),
                    "used": name == picked,
                    "is_baseline": name == BASELINE,
                }
            )
        n_beating = sum(
            1
            for n in all_central
            if n != BASELINE
            and float((err[BASELINE] - err[n]).mean())
            > 2 * float((err[BASELINE] - err[n]).std(ddof=1) / np.sqrt(len(train)))
        )
        typical = float(np.median(np.expm1(err[picked])) * 100)

        # Predict every part across the multi-question models
        full_fits = {name: MODELS[name](train) for name in MODELS}

        def _part_preds(
            model_name: str,
            _full_fits=full_fits,
            _part=part,
            _train=train,
            _oof=oof,
            _top3=top3,
        ) -> pd.Series:
            if model_name == ENSEMBLE_NAME:
                full_log = np.mean([_full_fits[m](_part) for m in _top3], axis=0)
            else:
                full_log = _full_fits[model_name](_part)
            s = pd.Series(np.exp(full_log), index=_part.index)
            s.loc[_train.index] = np.exp(_oof[model_name])
            return s

        fair = _part_preds(picked)
        ensemble_usd = _part_preds(ENSEMBLE_NAME)
        floor_p20_usd = _part_preds(FLOOR_NAME)
        ceiling_p80_usd = _part_preds(CEILING_NAME)
        huber_usd = _part_preds("Huber structural regression")
        bayesian_usd = _part_preds("Bayesian Ridge (posterior structural)")
        trees_usd = _part_preds("Extra Trees (structural)")

        floor_p20_usd = np.minimum(floor_p20_usd, fair * 0.95)
        ceiling_p80_usd = np.maximum(ceiling_p80_usd, fair * 1.05)

        top5 = sorted(non_base, key=lambda n: err[n].mean())[:5]
        top5_log_matrix = np.column_stack([np.log(_part_preds(m).to_numpy()) for m in top5])
        disagreement_pct = np.expm1(np.std(top5_log_matrix, axis=1, ddof=1)) * 100

        out = part.assign(
            fair_price_usd=fair,
            ensemble_price_usd=ensemble_usd,
            floor_p20_usd=floor_p20_usd,
            ceiling_p80_usd=ceiling_p80_usd,
            huber_price_usd=huber_usd,
            bayesian_price_usd=bayesian_usd,
            tree_price_usd=trees_usd,
            model_disagreement_pct=disagreement_pct,
        )
        out["gap_pct"] = (out["price_usd"] / out["fair_price_usd"] - 1) * 100
        out["confidence"] = np.where(out["trusted"], "two or more listings", "single listing")
        out["reading"] = np.select(
            [
                out["gap_pct"] <= -config.FAIR_PRICE_GAP_PCT,
                out["gap_pct"] >= config.FAIR_PRICE_GAP_PCT,
            ],
            ["below fair price", "above fair price"],
            default="near fair price",
        )
        out["price_zone"] = np.select(
            [
                out["price_usd"] <= out["floor_p20_usd"],
                out["price_usd"] >= out["ceiling_p80_usd"],
            ],
            ["Below P20 low-listed band", "Above P80 high-listed band"],
            default="Inside P20-P80 typical band",
        )
        out = _add_market_signals(out, history=history)

        # Historical Signal Validation: Do parts with high listing spread (>= 15%) experience
        # larger subsequent price drops than low-spread parts across collected history?
        high_spread_mask = (out["n_listings"] >= 2) & (out["listing_spread_pct"] >= 15.0)
        low_spread_mask = (out["n_listings"] >= 2) & (out["listing_spread_pct"] < 15.0)
        mean_chg_high = (
            float(out.loc[high_spread_mask, "price_change_7d_pct"].mean())
            if high_spread_mask.any()
            else 0.0
        )
        mean_chg_low = (
            float(out.loc[low_spread_mask, "price_change_7d_pct"].mean())
            if low_spread_mask.any()
            else 0.0
        )
        near_frontier_count = int(
            ((out["drop_to_frontier_pct"] > 0) & (out["drop_to_frontier_pct"] <= 10.0)).sum()
        )
        on_frontier_count = int((out["drop_to_frontier_usd"] == 0).sum())

        verdicts[kind] = {
            "used": picked,
            "best_challenger": best if best != BASELINE else None,
            "top3_ensemble_members": top3,
            "n_models_tested": len(ordered_names),
            "n_models_beating_baseline": int(n_beating),
            "n_parts": int(len(train)),
            "mae_log_baseline": float(err[BASELINE].mean()),
            "mae_log_best": float(err[best].mean()),
            "gain": float(gain.mean()),
            "gain_standard_error": se,
            "beats_baseline": bool(beats),
            "typical_error_pct": typical,
            "p20_p80_loo_coverage_pct": round(loo_coverage_pct, 1),
            "p20_p80_target_coverage_pct": 60.0,
            "p20_p80_relative_width_pct": round(rel_band_width_pct, 1),
            "market_signals": {
                "n_on_pareto_frontier": on_frontier_count,
                "n_within_10pct_of_frontier": near_frontier_count,
                "n_high_seller_spread_15pct": int(high_spread_mask.sum()),
                "mean_7d_change_high_spread_pct": round(mean_chg_high, 2),
                "mean_7d_change_low_spread_pct": round(mean_chg_low, 2),
            },
            "questions": {
                "central_fair_price": picked,
                "consensus_ensemble": ENSEMBLE_NAME,
                "discount_floor_p20": FLOOR_NAME,
                "scarcity_ceiling_p80": CEILING_NAME,
            },
            "finding": findings.from_test(
                beats,
                passed=f"{picked} won the {len(ordered_names)}-model regression tournament "
                f"({n_beating} models beat the naive baseline by > 2 SE; LOO P20–P80 band "
                f"coverage is {loo_coverage_pct:.0f}% vs 60% target).",
                failed="The fair price is the same price per point of score for every part, "
                "so the gap is score per dollar restated. No model beat that baseline out of "
                "sample.",
                otherwise=findings.OBSERVATION,
            ),
            "statement": (
                f"{picked} won the {len(ordered_names)}-model tournament (predicting within "
                f"{typical:.0f}% for a typical part; {n_beating} regressions beat the "
                f"same-price-per-point baseline by > 2 SE). Out-of-fold P20–P80 band coverage "
                f"is {loo_coverage_pct:.0f}% (60% nominal target, median band width "
                f"{rel_band_width_pct:.0f}%)."
                if beats
                else f"No model clearly beats the same-price-per-point baseline on "
                f"{len(train)} parts, so the baseline sets the fair price. It is typically "
                f"off by {typical:.0f}%, so read the gaps as rough."
            ),
        }
        parts.append(out)
    cols = [
        "kind",
        "model_key",
        "regime",
        "gen_stage",
        "price_usd",
        "fair_price_usd",
        "ensemble_price_usd",
        "floor_p20_usd",
        "ceiling_p80_usd",
        "huber_price_usd",
        "bayesian_price_usd",
        "tree_price_usd",
        "model_disagreement_pct",
        "gap_pct",
        "reading",
        "price_zone",
        "listing_spread_pct",
        "source_spread_pct",
        "channel_signal",
        "drop_to_frontier_usd",
        "drop_to_frontier_pct",
        "blocking_rival",
        "drop_to_best_value_usd",
        "drop_to_best_value_pct",
        "price_change_7d_pct",
        "lifecycle_signal",
        "confidence",
        "score",
        "n_listings",
        "n_retailers",
        "retailer",
        "listing_url",
    ]
    all_parts = (
        pd.concat(parts)[cols].sort_values(["kind", "gap_pct"]).reset_index(drop=True)
        if parts
        else pd.DataFrame(columns=cols)
    )
    return Result(pd.DataFrame(boards), all_parts, verdicts)
