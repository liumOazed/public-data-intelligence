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


def prepare(table: pd.DataFrame) -> pd.DataFrame:
    """Priced, scored parts with the columns the models use."""
    if table.empty:
        return pd.DataFrame()
    df = table[table["score"].notna() & table["price_usd"].notna()].copy()
    df["log_price"] = np.log(df["price_usd"].astype(float))
    df["log_score"] = np.log(df["score"].astype(float))
    df["vendor"] = df["model_key"].str.split().str[0]
    lines = [
        names.product_line(k, m)
        for k, m in zip(df["kind"], df["model_key"], strict=True)
    ]
    df["line"] = [item[0] if item else "other" for item in lines]
    df["gen"] = [item[1] if item else 0 for item in lines]
    df["is_workstation"] = (
        df["line"].isin(_WS_LINES)
        | df["model_key"].str.contains(_WS_RE, case=False, na=False)
    ).astype(float)
    df["is_x3d"] = df["model_key"].str.endswith("3d", na=False).astype(float)
    gen_step = np.where(df["line"] == "nvidia_geforce", df["gen"] / 10.0, df["gen"].astype(float))
    df["gen_step"] = gen_step
    max_gen = df.groupby("line")["gen_step"].transform("max")
    df["gen_lag"] = np.where(df["gen_step"] > 0, (max_gen - df["gen_step"]).clip(lower=0), 0.0)
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
FLOOR_NAME = "L1 Quantile P20 (Discount Floor)"
CEILING_NAME = "L1 Quantile P80 (Scarcity Ceiling)"

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
        "Quantile (L1)",
        "Discount floor (P20 buy zone)",
        fit_structural_quantile_p20,
    ),
    CEILING_NAME: (
        "Quantile (L1)",
        "Scarcity ceiling (P80 FOMO/EOL)",
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


@dataclass
class Result:
    leaderboard: pd.DataFrame
    parts: pd.DataFrame
    verdicts: dict


def evaluate(table: pd.DataFrame) -> Result:
    df = prepare(table)
    boards, parts, verdicts = [], [], {}
    central_candidates = [
        name
        for name, (_, role, _) in MODEL_SPECS.items()
        if name not in {FLOOR_NAME, CEILING_NAME}
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
                name not in {BASELINE, FLOOR_NAME, CEILING_NAME}
                and float(g.mean()) > 2 * g_se
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
            "questions": {
                "central_fair_price": picked,
                "consensus_ensemble": ENSEMBLE_NAME,
                "discount_floor_p20": FLOOR_NAME,
                "scarcity_ceiling_p80": CEILING_NAME,
            },
            "finding": findings.from_test(
                beats,
                passed=f"{picked} won the {len(ordered_names)}-model regression tournament "
                f"({n_beating} models beat the naive baseline by > 2 SE).",
                failed="The fair price is the same price per point of score for every part, "
                "so the gap is score per dollar restated. No model beat that baseline out of "
                "sample.",
                otherwise=findings.OBSERVATION,
            ),
            "statement": (
                f"{picked} won the {len(ordered_names)}-model tournament (predicting within "
                f"{typical:.0f}% for a typical part; {n_beating} regressions beat the "
                f"same-price-per-point baseline by > 2 SE)."
                if beats
                else f"No model clearly beats the same-price-per-point baseline on "
                f"{len(train)} parts, so the baseline sets the fair price. It is typically "
                f"off by {typical:.0f}%, so read the gaps as rough."
            ),
        }

        # Predict every part across the multi-question models:
        # 1. Picked best model (central fair price)
        # 2. Top-3 LOO Ensemble (consensus price)
        # 3. Top-5 central models spread (cross-model disagreement %)
        # 4. P20 Discount Floor & P80 Scarcity Ceiling
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

        # Ensure P20 <= Fair <= P80 monotonicity for display clarity
        floor_p20_usd = np.minimum(floor_p20_usd, fair * 0.95)
        ceiling_p80_usd = np.maximum(ceiling_p80_usd, fair * 1.05)

        top5 = sorted(non_base, key=lambda n: err[n].mean())[:5]
        top5_log_matrix = np.column_stack(
            [np.log(_part_preds(m).to_numpy()) for m in top5]
        )
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
            ["Below P20 discount floor", "Above P80 scarcity ceiling"],
            default="Inside P20-P80 normal band",
        )
        parts.append(out)
    cols = [
        "kind",
        "model_key",
        "regime",
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
