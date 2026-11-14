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
