"""Federal Procurement: name cleaning, the metrics, and the check against the published figure."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from pdi.common.fetch import CollectError
from pdi.procurement import collect, config, tables


def _tx(rows: list[dict]) -> pd.DataFrame:
    base = {
        "contract_award_unique_key": "A1",
        "modification_number": "0",
        "action_date": "2024-11-05",
        "federal_action_obligation": 100.0,
        "total_dollars_obligated": 100.0,
        "potential_total_value_of_award": 200.0,
        "total_outlayed_amount_for_overall_award": None,
        "awarding_agency_name": "Department of Testing",
        "recipient_uei": "U1",
        "recipient_name": "ACME COMPUTERS, INC.",
        "recipient_parent_uei": "",
        "recipient_parent_name": "",
        "product_or_service_code": "7B22",
        "extent_competed": "FULL AND OPEN COMPETITION",
        "number_of_offers_received": "3",
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["contract_transaction_unique_key"] = [f"T{i}" for i in range(len(df))]
    df["action_date"] = pd.to_datetime(df["action_date"])
    return df


def test_clean_name_drops_case_punctuation_and_legal_endings():
    assert tables.clean_name("Dell Marketing L.P.") == "DELL MARKETING"
    assert tables.clean_name("FOUR POINTS TECHNOLOGY, L.L.C.") == "FOUR POINTS TECHNOLOGY"
    assert tables.clean_name("The AT&T Corp") == "AT AND T"
    assert tables.clean_name("Blue Tech Inc.") == tables.clean_name("BLUE TECH, INC")
    assert tables.clean_name(None) == ""


def test_suppliers_group_by_parent_then_by_cleaned_name():
    tx = _tx([
        {"recipient_uei": "U1", "recipient_name": "DELL FEDERAL SYSTEMS L.P",
         "recipient_parent_uei": "P1", "recipient_parent_name": "DELL TECHNOLOGIES INC."},
        {"recipient_uei": "U2", "recipient_name": "DELL MARKETING L.P.",
         "recipient_parent_uei": "P1", "recipient_parent_name": "DELL TECHNOLOGIES INC."},
        # The same company written differently, filed with no parent.
        {"recipient_uei": "U3", "recipient_name": "Dell Technologies, Inc"},
        {"recipient_uei": "U4", "recipient_name": "OTHER LLC", "federal_action_obligation": 5.0},
        {"recipient_uei": "", "recipient_name": "Jane Roe"},
    ])  # fmt: skip
    out, names, stats = tables.with_suppliers(tx)
    assert out["supplier"].iloc[:3].nunique() == 1
    assert stats["registrations"] == 4 and stats["after_parent_grouping"] == 4
    assert stats["suppliers"] == 3 and stats["transactions_without_registration"] == 1
    # A recipient with no registration is stored under a digest, never under the name.
    nameless = out["supplier"].iloc[4]
    assert nameless.startswith("N") and "ROE" not in nameless.upper()
    assert names[out["supplier"].iloc[0]].upper().startswith("DELL TECHNOLOGIES")


def test_fiscal_year_starts_in_october():
    d = pd.to_datetime(pd.Series(["2024-09-30", "2024-10-01", "2025-09-30"]))
    assert tables.fiscal_year(d).tolist() == [2024, 2025, 2025]


def test_concentration_ignores_suppliers_with_nothing_net():
    sy = pd.DataFrame({
        "fiscal_year": [2025] * 4,
        "supplier": list("ABCD"),
        "obligations": [50.0, 30.0, 20.0, -10.0],
        "actions": [1, 1, 1, 1],
    })  # fmt: skip
    row = tables.concentration(sy).iloc[0]
    assert row["suppliers"] == 3 and row["obligations"] == 100
    assert row["hhi"] == pytest.approx(50**2 + 30**2 + 20**2)
    assert row["top4_share_pct"] == pytest.approx(100)


def test_entrant_needs_a_clean_lookback():
    first = config.FIRST_FISCAL_YEAR
    judged = first + config.ENTRANT_LOOKBACK
    sy = pd.DataFrame({
        "fiscal_year": [first, judged, judged, judged],
        "supplier": ["OLD", "OLD", "NEW", "GONE"],
        "obligations": [10.0, 60.0, 40.0, -5.0],
        "actions": [1, 1, 1, 1],
    })  # fmt: skip
    out = tables.entrants(sy)
    assert out["fiscal_year"].tolist() == [judged]
    assert out["entrants"].iloc[0] == 1
    assert out["entrant_share_of_dollars_pct"].iloc[0] == pytest.approx(40)


def test_single_bidder_share_counts_only_competed_dollars_with_offers_recorded():
    tx = _tx([
        {"federal_action_obligation": 60.0, "number_of_offers_received": "1"},
        {"federal_action_obligation": 40.0, "number_of_offers_received": "4"},
        {"federal_action_obligation": 50.0, "number_of_offers_received": ""},
        {"federal_action_obligation": 50.0, "extent_competed": "NOT COMPETED",
         "number_of_offers_received": "1"},
        {"federal_action_obligation": -500.0, "number_of_offers_received": "1"},
    ])  # fmt: skip
    row = tables.single_bidder(tables.competition(tx)).iloc[0]
    assert row["new_dollars"] == 200, "money taken back is not a new dollar"
    assert row["single_bidder_share_pct"] == pytest.approx(60)
    assert row["not_competed_share_pct"] == pytest.approx(25)
    assert row["competed_with_offers_recorded_pct"] == pytest.approx(50)


def test_promised_ceiling_and_paid_stay_in_separate_columns():
    tx = _tx([
        {"contract_award_unique_key": "A1", "action_date": "2022-11-01",
         "total_dollars_obligated": 100.0, "potential_total_value_of_award": 400.0},
        {"contract_award_unique_key": "A1", "modification_number": "P00001",
         "action_date": "2023-12-01", "total_dollars_obligated": 250.0,
         "potential_total_value_of_award": 400.0,
         "total_outlayed_amount_for_overall_award": 90.0},
        {"contract_award_unique_key": "A2", "action_date": "2023-01-10",
         "total_dollars_obligated": 70.0, "potential_total_value_of_award": 70.0},
        # First action is outside the data: the award has no start year and is left out.
        {"contract_award_unique_key": "A3", "modification_number": "P00009",
         "action_date": "2023-02-01"},
    ])  # fmt: skip
    out = tables.cohorts(tx)
    assert out["fiscal_year"].tolist() == [2023]
    row = out.iloc[0]
    assert (row["awards"], row["promised"], row["ceiling"]) == (2, 320.0, 470.0)
    # Paid is compared only with what was promised on the awards that report a payment.
    assert row["awards_reporting_paid"] == 1
    assert (row["promised_where_paid_reported"], row["paid"]) == (250.0, 90.0)


def test_parse_keeps_only_configured_columns_and_stops_on_a_stray_code():
    raw = _tx([{}]).astype(str).assign(
        action_date="2024-11-05", total_outlayed_amount_for_overall_award="",
        recipient_address_line_1="1 Main St", primary_place_of_performance_zip_4="00000",
    )  # fmt: skip
    df = collect.parse(raw)
    assert list(df.columns) == config.COLUMNS
    assert pd.isna(df["total_outlayed_amount_for_overall_award"].iloc[0])
    with pytest.raises(CollectError, match="outside the slice"):
        collect.parse(raw.assign(product_or_service_code="7030"))
    with pytest.raises(CollectError, match="no column"):
        collect.parse(raw.drop(columns=["extent_competed"]))


def test_no_personal_column_is_requested():
    banned = ("address", "phone", "fax", "officer", "executive", "zip", "city", "congressional")
    assert not [c for c in config.COLUMNS if any(b in c for b in banned)]
    assert not set(config.PSC) & {"7030", "7A20", "7A21"}, "software is not hardware"


def test_published_months_are_fiscal_months():
    payload = {"results": [
        {"time_period": {"fiscal_year": "2025", "month": "1"}, "aggregated_amount": 10.0},
        {"time_period": {"fiscal_year": "2025", "month": "3"}, "aggregated_amount": 30.0},
        {"time_period": {"fiscal_year": "2025", "month": "4"}, "aggregated_amount": 40.0},
        {"time_period": {"fiscal_year": "2025", "month": "12"}, "aggregated_amount": 120.0},
    ]}  # fmt: skip
    out = collect.parse_published(payload)
    assert out["period"].dt.strftime("%Y-%m").tolist() == [
        "2024-10", "2024-12", "2025-01", "2025-09",
    ]  # fmt: skip


def test_check_flags_a_month_outside_the_limit_and_leaves_out_the_current_month():
    monthly = pd.DataFrame({
        "period": pd.to_datetime(["2025-08-01", "2025-09-01", "2025-10-01"]),
        "obligations": [100.0, 100.0, 5.0],
    })  # fmt: skip
    published = pd.DataFrame({
        "period": pd.to_datetime(["2025-08-01", "2025-09-01", "2025-10-01"]),
        "published": [100.5, 123.0, 5.0],
    })  # fmt: skip
    check = collect.check_totals(monthly, published, date(2025, 10, 20))
    assert check["period"].dt.month.tolist() == [8, 9]
    assert check["within_limit"].tolist() == [True, False]
    assert check["gap_share"].iloc[1] == pytest.approx(23 / 123)


def _month_tx():
    tx = _tx([
        {"contract_award_unique_key": "CONT_AWD_W1_9700_-NONE-_-NONE-", "modification_number": "0",
         "action_date": "2026-01-05", "federal_action_obligation": 1000.0},
        {"contract_award_unique_key": "CONT_AWD_W2_9700_X1_9700", "modification_number": "P00001",
         "action_date": "2026-01-06", "federal_action_obligation": -50.0},
    ])  # fmt: skip
    return tx


def test_reconcile_finds_second_copies_in_the_search():
    listed = pd.DataFrame({
        "Award ID": ["W1", "W1", "W2"], "Mod": ["0", "0", "P00001"],
        "Action Date": ["2026-01-05", "2026-01-05", "2026-01-06"],
        "Transaction Amount": [1000.0, 1000.0, -50.0],
    })  # fmt: skip
    row = collect.reconcile(pd.Timestamp("2026-01-01"), _month_tx(), listed)
    assert row["search"] == 1950 and row["download"] == 950
    assert row["extra_copies"] == 1 and row["extra_copies_sum"] == 1000
    assert row["search_rows_not_in_download"] == 0
    assert row["gap_after_copies_share"] == 0 and row["reconciled"]


def test_reconcile_fails_when_the_search_has_a_transaction_the_download_lacks():
    listed = pd.DataFrame({
        "Award ID": ["W1", "W2", "W3"], "Mod": ["0", "P00001", "0"],
        "Action Date": ["2026-01-05", "2026-01-06", "2026-01-07"],
        "Transaction Amount": [1000.0, -50.0, 0.5],
    })  # fmt: skip
    row = collect.reconcile(pd.Timestamp("2026-01-01"), _month_tx(), listed)
    assert row["search_rows_not_in_download"] == 1
    assert not row["reconciled"], "a missing row fails even when the money is tiny"
