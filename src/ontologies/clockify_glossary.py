"""Clockify glossary with term aliases and auto-aliasing logic."""
from __future__ import annotations
import re
import pathlib
from typing import Any


# Curated Clockify terms and their canonical aliases
CURATED = {
    "timesheet": ["weekly timesheet", "submit timesheet", "approve timesheet"],
    "approval": ["timesheet approval", "submit for approval", "manager approval"],
    "billable rate": ["bill rate", "billing rate", "hourly bill rate"],
    "billable hours": ["chargeable hours", "billed hours"],
    "project budget": ["budget estimate", "task-based estimate", "manual estimate"],
    "time rounding": ["round time", "15-minute rounding", "round up", "round to nearest"],
    "audit log": ["activity log", "change log"],
    "idle time detection": ["idle detection", "afk detection"],
    "pomodoro timer": ["pomodoro", "25-minute timer"],
    "pto": ["time off", "paid time off", "vacation", "holiday"],
    "kiosk": ["time clock", "clock in terminal", "pin code"],
    "sso": ["single sign-on", "saml", "oauth"],
    "member rate": ["user rate", "team member rate"],
    "project rate": ["per-project rate"],
    "workspace rate": ["org rate", "organization rate"],
    "cost rate": ["labor cost rate", "internal rate"],
    "scheduled report": ["email report", "weekly report", "daily report"],
    "detailed report": ["csv export", "excel export"],
    "summary report": ["grouped report"],
    "tags": ["labels", "categories"],
    "user group": ["group"],
    "webhooks": ["clockify webhooks", "http callback"],
}


def _norm(s: str) -> str:
    """Normalize string for comparison: lowercase, remove special chars, strip."""
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def extract_terms(raw: str) -> list[dict[str, str]]:
    """
    Extract terms from glossary text.
    Terms are lines ending with " #" (markdown convention).

    Args:
        raw: Raw glossary text

    Returns:
        List of {"term": str, "norm": str} dicts
    """
    terms = []
    for line in raw.splitlines():
        if line.strip().endswith("#"):
            t = line.strip().rstrip("#").strip()
            # Skip separator lines like "A | B"
            if t and not re.match(r"^[A-Z] \|", t):
                terms.append({"term": t, "norm": _norm(t)})
    return terms


def auto_alias(t: str) -> list[str]:
    """
    Auto-generate aliases for a term.

    Examples:
        "timesheet" -> ["timesheet", "timesheetin", "timesheet-", ...]
        "paid time off" -> ["paidtimeoff", "paid-time-off", "paidtimeof", "pto", ...]
    """
    base = _norm(t)
    outs = {base}

    # No-space variant
    outs.add(base.replace(" ", ""))

    # Hyphen variant
    outs.add(base.replace(" ", "-"))

    # Plural → singular
    if base.endswith("s"):
        outs.add(base[:-1])

    # Hardcoded common abbreviations
    if base in ("paid time off", "paidtimeoff"):
        outs.add("pto")
    if base in ("single sign on", "singlesignon"):
        outs.add("sso")

    return sorted(x for x in outs if x)


def build_aliases(terms: list[dict[str, str]]) -> dict[str, list[str]]:
    """
    Build canonical aliases for all terms.

    Args:
        terms: List of {"term": str, "norm": str} from extract_terms()

    Returns:
        Dict mapping normalized term → list of aliases
    """
    aliases = {}

    for t in terms:
        key = t["term"].lower()
        vals = auto_alias(t["term"])

        # Merge with curated if available
        if key in CURATED:
            curated_norms = {_norm(x) for x in CURATED[key]}
            vals = sorted(set(vals) | curated_norms)

        aliases[key] = vals

    return aliases


def load_aliases(glossary_path: str | None = None) -> dict[str, list[str]]:
    """
    Load and build aliases from glossary file, or fallback to curated only.

    Args:
        glossary_path: Path to glossary text file (default: docs/clockify_glossary.txt)

    Returns:
        Dict mapping term → list of normalized aliases
    """
    p = pathlib.Path(glossary_path or "docs/clockify_glossary.txt")

    if p.exists():
        try:
            raw = p.read_text(encoding="utf-8")
            terms = extract_terms(raw)
            return build_aliases(terms)
        except Exception:
            pass

    # Fallback to curated only
    base = [{"term": k} for k in CURATED.keys()]
    return build_aliases(base)


# Module-level singleton
ALIASES = load_aliases()
