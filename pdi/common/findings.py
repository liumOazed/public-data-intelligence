"""Finding labels shared by every module.

Each finding a page shows carries one of three labels, so a reader can tell a measured fact
from a fitted story from a tested forecast:

* observation: a measured value or simple arithmetic on measured values. No model.
* explanation: a fitted model describes a relationship, but it has not beaten the naive
  baseline out of sample, or the claim is about the fit rather than about new data.
* prediction: a model's output, used only when that model beat the naive baseline on data
  it never saw, by more than the noise level the module states in advance.

The label is decided here in code from the test result, never typed by hand on a page.
"""

from __future__ import annotations

OBSERVATION = "observation"
EXPLANATION = "explanation"
PREDICTION = "prediction"
LABELS = (OBSERVATION, EXPLANATION, PREDICTION)


def finding(label: str, reason: str) -> dict:
    if label not in LABELS:
        raise ValueError(f"unknown finding label {label!r}")
    return {"label": label, "reason": reason}


def observation(reason: str) -> dict:
    return finding(OBSERVATION, reason)


def explanation(reason: str) -> dict:
    return finding(EXPLANATION, reason)


def from_test(beats_baseline: bool, *, passed: str, failed: str, otherwise: str) -> dict:
    """Prediction only when the out-of-sample test passed; `otherwise` is the fallback label."""
    if beats_baseline:
        return finding(PREDICTION, passed)
    return finding(otherwise, failed)


# --- The shared finding format ----------------------------------------------------------------
#
# Every module exports the items its own fixed rule marks as unusual in one table,
# site/data/v1/<module>/findings.json, with these columns. The "What's unusual this week" page
# reads only these files.

ITEM_COLUMNS = [
    "key",  # stable id: module|subject|rule|period, so a later build can tell new from standing
    "module",
    "subject",  # what moved: a series, an economy, a partner, a model
    "period",  # the date the item refers to (a month, or a collection day), ISO
    "sentence",
    "rule",  # short name of the module's fixed rule
    "measure",  # the number the rule tested (spreads, or a share)
    "threshold",
    "label",
    "label_reason",
    "test_passed",  # True only when a prediction's out-of-sample test passed; else None
    "page",  # the page with the evidence, relative to the site root
    "table",  # the data file with the evidence, relative to site/data/v1/
]


def item(
    *,
    module: str,
    subject: str,
    period,
    sentence: str,
    rule: str,
    measure: float,
    threshold: float,
    finding: dict,
    page: str,
    table: str,
    test_passed: bool | None = None,
) -> dict:
    if finding["label"] == PREDICTION and test_passed is not True:
        raise ValueError(f"{module}: {subject} is labelled a prediction without a passed test")
    period = period.isoformat()[:10] if hasattr(period, "isoformat") else str(period)[:10]
    return {
        "key": f"{module}|{subject}|{rule}|{period}",
        "module": module,
        "subject": subject,
        "period": period,
        "sentence": sentence,
        "rule": rule,
        "measure": float(measure),
        "threshold": float(threshold),
        "label": finding["label"],
        "label_reason": finding["reason"],
        "test_passed": test_passed,
        "page": page,
        "table": table,
    }


def write_items(module: str, items: list[dict], *, rule: str, write=None, **table_kwargs):
    """Write one module's findings.json in the shared format.

    `write` is the module's own write_table, passed so that tests that redirect a module's
    writes also redirect this one.
    """
    import pandas as pd

    from pdi.common.export import write_table

    write_table = write or write_table
    df = pd.DataFrame(items, columns=ITEM_COLUMNS)
    extra = {**(table_kwargs.pop("extra", None) or {}), "rule_text": rule}
    return write_table(
        module,
        "findings",
        df,
        description=f"Items this module's fixed rule marks as unusual. Rule: {rule}",
        extra=extra,
        **table_kwargs,
    )
