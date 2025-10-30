"""Tests for glossary integration and query expansion."""
import pytest
from src.query_rewrite import expand, is_definitional
from src.ontologies.clockify_glossary import ALIASES, extract_terms, _norm


def test_aliases_loaded():
    """Verify glossary aliases are loaded correctly."""
    assert "timesheet" in ALIASES or any("timesheet" in k for k in ALIASES.keys())
    assert "approval" in ALIASES or any("approval" in k for k in ALIASES.keys())


def test_expand_variants_cap():
    """Test query expansion caps at max_vars."""
    out = expand("How do I submit my timesheet for approval?", max_vars=5)
    assert isinstance(out, list)
    assert len(out) <= 5
    assert out[0].startswith("How do I submit")


def test_expand_preserves_original():
    """First variant should always be the original query."""
    q = "Help with project budgets"
    out = expand(q, max_vars=3)
    assert out[0] == q


def test_is_definitional():
    """Test definitional query detection."""
    assert is_definitional("What is billable rate?") is True
    assert is_definitional("Define timesheet") is True
    assert is_definitional("How do I enable SSO?") is False
    assert is_definitional("Log time to project") is False


def test_norm_consistency():
    """Test string normalization."""
    assert _norm("TimeSheet") == "timesheet"
    assert _norm("Billable-Rate") == "billablerate"
    assert _norm("PTO (Paid Time Off)") == "pto paid time off"


def test_extract_terms_from_glossary():
    """Test parsing glossary terms marked with #."""
    sample = "### Timesheet #\nA weekly record.\n### Timer #\nA clock."
    terms = extract_terms(sample)
    assert len(terms) >= 2
    assert any("timesheet" in t["norm"] for t in terms)
