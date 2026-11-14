import pytest

from pdi import site
from pdi.common import findings


def test_prediction_needs_a_passed_test():
    passed = findings.from_test(True, passed="p", failed="f", otherwise=findings.EXPLANATION)
    failed = findings.from_test(False, passed="p", failed="f", otherwise=findings.EXPLANATION)
    assert passed == {"label": "prediction", "reason": "p"}
    assert failed == {"label": "explanation", "reason": "f"}


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        findings.finding("hunch", "x")


def test_badge_links_relative_to_page():
    badge = site.finding_badge(findings.observation("x"))
    assert badge.startswith("[Observation](../methodology.qmd#finding-labels)")
    assert ".finding-observation" in badge
    assert site.finding_line(None) == ""
