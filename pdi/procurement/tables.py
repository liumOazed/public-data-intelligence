"""Supplier name cleaning and the procurement metrics. Pure functions over the transactions.

The rules are fixed in pdi.procurement.config. Three kinds of money stay in three columns all
the way through: promised by an action (obligation), the award's ceiling, and paid (outlay).
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd

from pdi.procurement import config

_PUNCT = re.compile(r"[^A-Z0-9 ]+")


def fiscal_year(dates: pd.Series) -> pd.Series:
    """US fiscal year: October 2024 to September 2025 is fiscal 2025."""
    return (dates.dt.year + (dates.dt.month >= 10)).astype(int)


def clean_name(name: object) -> str:
    """A company name with case, punctuation and legal endings removed."""
    if not isinstance(name, str):
        return ""
    text = name.upper().replace("&", " AND ")
    text = _PUNCT.sub(" ", text.replace(".", "").replace("'", ""))
    words = text.split()
    while words and words[-1] in config.LEGAL_ENDINGS:
        words.pop()
    while words and words[0] == "THE":
        words.pop(0)
    return " ".join(words)


def _most_common(values: pd.Series) -> str:
    values = values[values.astype(bool)]
    return values.mode().iloc[0] if len(values) else ""


def with_suppliers(tx: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], dict]:
    """Add a `supplier` column: parent company where named, merged on identical cleaned names.

    Returns the transactions, a name for every supplier id, and counts of what was merged.
    """
    tx = tx.copy()
    for col in ("recipient_uei", "recipient_parent_uei", "recipient_name", "recipient_parent_name"):
        tx[col] = tx[col].fillna("").astype(str).str.strip()
    has_parent = tx["recipient_parent_uei"] != ""
    has_uei = tx["recipient_uei"] != ""
    first = tx["recipient_parent_uei"].where(has_parent, tx["recipient_uei"])
    # No registration at all: the id is a digest of the cleaned name, never the name itself.
    nameless = tx["recipient_name"].map(
        lambda n: "N" + hashlib.sha256(clean_name(n).encode()).hexdigest()[:12]
    )
    first = first.where(has_parent | has_uei, nameless)
    label = tx["recipient_parent_name"].where(has_parent, tx["recipient_name"])
    label = label.where(label != "", tx["recipient_name"])
    tx["_first"], tx["_label"] = first, label

    groups = tx.groupby("_first").agg(
        name=("_label", _most_common),
        size=("federal_action_obligation", lambda s: float(s.abs().sum())),
    )
    groups["clean"] = groups["name"].map(clean_name)
    # Within each cleaned name, every group takes the id of the largest one.
    ordered = groups.sort_values("size", ascending=False)
    lead = ordered.groupby("clean").apply(lambda g: g.index[0], include_groups=False)
    groups["supplier"] = [
        lead[c] if c else i for i, c in zip(groups.index, groups["clean"], strict=True)
    ]
    tx["supplier"] = tx["_first"].map(groups["supplier"])
    names = groups.loc[groups.index == groups["supplier"], "name"].to_dict()
    stats = {
        "transactions": int(len(tx)),
        "names_as_written": int(tx["recipient_name"].nunique()),
        "registrations": int(tx.loc[has_uei, "recipient_uei"].nunique()),
        "after_parent_grouping": int(groups.shape[0]),
        "suppliers": int(groups["supplier"].nunique()),
        "transactions_without_registration": int((~has_uei).sum()),
    }
    return tx.drop(columns=["_first", "_label"]), names, stats


def monthly(tx: pd.DataFrame) -> pd.DataFrame:
    out = (
        tx.assign(period=tx["action_date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["period", "awarding_agency_name", "product_or_service_code"], as_index=False)
        .agg(obligations=("federal_action_obligation", "sum"),
             actions=("federal_action_obligation", "size"))
    )  # fmt: skip
    return out.rename(columns={"awarding_agency_name": "agency", "product_or_service_code": "psc"})


def supplier_year(tx: pd.DataFrame) -> pd.DataFrame:
    """Net obligations per supplier and fiscal year; a row exists for any year with an action."""
    return (
        tx.assign(fiscal_year=fiscal_year(tx["action_date"]))
        .groupby(["fiscal_year", "supplier"], as_index=False)
        .agg(obligations=("federal_action_obligation", "sum"),
             actions=("federal_action_obligation", "size"))
    )  # fmt: skip


def concentration(sy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fy, g in sy[sy["obligations"] > 0].groupby("fiscal_year"):
        share = (100 * g["obligations"] / g["obligations"].sum()).sort_values(ascending=False)
        row = {"fiscal_year": int(fy), "suppliers": int(len(g)),
               "obligations": float(g["obligations"].sum()),
               "hhi": float((share**2).sum())}  # fmt: skip
        for n in config.TOP_N:
            row[f"top{n}_share_pct"] = float(share.head(n).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def entrants(sy: pd.DataFrame) -> pd.DataFrame:
    """Suppliers paid in a year that had no action in the lookback years before it."""
    first = int(sy["fiscal_year"].min()) + config.ENTRANT_LOOKBACK
    seen = sy.groupby("fiscal_year")["supplier"].agg(set)
    rows = []
    for fy, g in sy[(sy["obligations"] > 0) & (sy["fiscal_year"] >= first)].groupby("fiscal_year"):
        before = set().union(*(seen.get(y, set()) for y in range(fy - config.ENTRANT_LOOKBACK, fy)))
        new = ~g["supplier"].isin(before)
        rows.append(
            {"fiscal_year": int(fy), "entrants": int(new.sum()),
             "entrant_share_of_suppliers_pct": float(100 * new.mean()),
             "entrant_share_of_dollars_pct":
                 float(100 * g.loc[new, "obligations"].sum() / g["obligations"].sum())}
        )  # fmt: skip
    return pd.DataFrame(rows)


def competition(tx: pd.DataFrame) -> pd.DataFrame:
    """Newly obligated dollars (actions above zero) by how the award was competed."""
    new = tx[tx["federal_action_obligation"] > 0]
    extent = new["extent_competed"].fillna("").str.upper().str.strip()
    offers = pd.to_numeric(new["number_of_offers_received"], errors="coerce")
    kind = pd.Series("not_stated", index=new.index)
    kind[extent.isin(config.NOT_COMPETED)] = "not_competed"
    competed = extent.isin(config.COMPETED)
    kind[competed] = "competed_offers_not_recorded"
    kind[competed & (offers == 1)] = "competed_one_offer"
    kind[competed & (offers >= 2)] = "competed_two_or_more_offers"
    return (
        new.assign(kind=kind, fiscal_year=fiscal_year(new["action_date"]))
        .groupby(["fiscal_year", "kind"], as_index=False)
        .agg(new_dollars=("federal_action_obligation", "sum"),
             actions=("federal_action_obligation", "size"))
    )  # fmt: skip


def single_bidder(comp: pd.DataFrame) -> pd.DataFrame:
    wide = comp.pivot(index="fiscal_year", columns="kind", values="new_dollars").fillna(0.0)
    for col in ("competed_one_offer", "competed_two_or_more_offers", "not_competed",
                "competed_offers_not_recorded", "not_stated"):  # fmt: skip
        if col not in wide:
            wide[col] = 0.0
    known = wide["competed_one_offer"] + wide["competed_two_or_more_offers"]
    total = wide.sum(axis=1)
    return pd.DataFrame(
        {
            "fiscal_year": wide.index.astype(int),
            "new_dollars": total.to_numpy(),
            "single_bidder_share_pct": (100 * wide["competed_one_offer"] / known).to_numpy(),
            "not_competed_share_pct": (100 * wide["not_competed"] / total).to_numpy(),
            "competed_with_offers_recorded_pct": (100 * known / total).to_numpy(),
        }
    )


def cohorts(tx: pd.DataFrame) -> pd.DataFrame:
    """Promised, ceiling and paid for the awards that began in each fiscal year."""
    key = "contract_award_unique_key"
    ordered = tx.sort_values(["action_date", "modification_number"])
    start = (
        ordered[ordered["modification_number"].astype(str) == "0"].groupby(key)["action_date"].min()
    )
    last = ordered.groupby(key).agg(
        promised=("total_dollars_obligated", "last"),
        ceiling=("potential_total_value_of_award", "last"),
        paid=("total_outlayed_amount_for_overall_award", "last"),
    )
    awards = last.join(start.rename("start"), how="inner")
    awards["fiscal_year"] = fiscal_year(awards["start"])
    reported = awards["paid"].notna()
    awards["promised_where_paid_reported"] = awards["promised"].where(reported)
    out = awards.groupby("fiscal_year").agg(
        awards=("promised", "size"),
        promised=("promised", "sum"),
        ceiling=("ceiling", "sum"),
        awards_reporting_paid=("paid", "count"),
        promised_where_paid_reported=("promised_where_paid_reported", "sum"),
        paid=("paid", "sum"),
    )
    return out.reset_index()
