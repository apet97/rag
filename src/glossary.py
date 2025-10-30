#!/usr/bin/env python3
"""Glossary management: load, detect, and expand terms."""

import csv
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Glossary:
    """Manage Clockify glossary terms and aliases."""

    def __init__(self, glossary_path: Optional[str] = None):
        """
        Initialize glossary from CSV.

        Args:
            glossary_path: Path to glossary CSV file (term,aliases,type,notes)
        """
        self.glossary_path = glossary_path or os.getenv("GLOSSARY_PATH", "data/glossary.csv")
        self.terms: Dict[str, str] = {}  # term -> canonical form
        self.aliases: Dict[str, str] = {}  # alias -> canonical form
        self.types: Dict[str, str] = {}  # canonical -> type
        self.notes: Dict[str, str] = {}  # canonical -> notes
        self._load_glossary()

    def _load_glossary(self):
        """Load glossary from CSV."""
        path = Path(self.glossary_path)
        if not path.exists():
            logger.warning(f"Glossary not found at {self.glossary_path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    term = row.get("term", "").strip()
                    aliases_str = row.get("aliases", "").strip()
                    term_type = row.get("type", "").strip()
                    notes = row.get("notes", "").strip()

                    if not term:
                        continue

                    canonical = self._normalize(term)
                    self.terms[canonical] = term
                    self.types[canonical] = term_type
                    self.notes[canonical] = notes

                    # Register aliases
                    for alias in aliases_str.split("|"):
                        alias = alias.strip()
                        if alias:
                            alias_normalized = self._normalize(alias)
                            self.aliases[alias_normalized] = canonical

            logger.info(f"Loaded {len(self.terms)} glossary terms from {self.glossary_path}")
        except Exception as e:
            logger.error(f"Failed to load glossary: {e}")

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison: lowercase, no special chars."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def detect_terms(self, text: str) -> Set[str]:
        """
        Detect glossary terms in text.

        Args:
            text: Text to search

        Returns:
            Set of canonical term forms found
        """
        normalized = self._normalize(text)
        found = set()

        # Check each term and alias
        all_keys = set(self.terms.keys()) | set(self.aliases.keys())
        for key in all_keys:
            if key in normalized:
                # Prefer exact word boundaries to avoid false positives
                canonical = self.terms.get(key) or self.aliases.get(key)
                if canonical:
                    found.add(canonical)

        return found

    def expand_query(self, query: str, max_variants: int = 3) -> List[str]:
        """
        Expand query with glossary aliases.

        Args:
            query: Original query
            max_variants: Max expansions to return (including original)

        Returns:
            List of query variations
        """
        variants = [query]  # Always include original

        detected = self.detect_terms(query)
        if not detected:
            return variants

        # For each detected term, add variant with canonical form
        for term in sorted(detected):
            if len(variants) >= max_variants:
                break

            canonical = self._normalize(term)
            if canonical in self.terms:
                display_term = self.terms[canonical]
                variant = query
                # Try to replace common variants with canonical
                for alias_norm, canonical_norm in self.aliases.items():
                    if canonical_norm == canonical:
                        # Replace alias with display term
                        pattern = r'\b' + re.escape(alias_norm.replace(" ", r'\s+')) + r'\b'
                        variant = re.sub(pattern, display_term, variant, flags=re.IGNORECASE)
                        if variant != query:
                            variants.append(variant)
                            break

        return variants[:max_variants]

    def get_term_info(self, term: str) -> Optional[Dict]:
        """
        Get full term information.

        Args:
            term: Term to look up (any form)

        Returns:
            Dict with term, type, notes or None
        """
        normalized = self._normalize(term)
        canonical = self.terms.get(normalized) or self.aliases.get(normalized)

        if not canonical:
            return None

        canonical_norm = self._normalize(canonical)
        return {
            "term": canonical,
            "canonical": canonical_norm,
            "type": self.types.get(canonical_norm, ""),
            "notes": self.notes.get(canonical_norm, ""),
        }

    def get_all_aliases(self, term: str) -> List[str]:
        """Get all known aliases for a term."""
        canonical_norm = self._normalize(term)
        canonical = self.terms.get(canonical_norm) or term

        aliases = [canonical]
        for alias_norm, canonical_norm_check in self.aliases.items():
            if canonical_norm_check == canonical_norm:
                # Find the original form
                for orig, norm in self.aliases.items():
                    if norm == canonical_norm and orig != canonical_norm:
                        aliases.append(orig)

        return list(set(aliases))


# Global glossary instance
_glossary_instance: Optional[Glossary] = None


def get_glossary() -> Glossary:
    """Get or create global glossary instance."""
    global _glossary_instance
    if _glossary_instance is None:
        _glossary_instance = Glossary()
    return _glossary_instance


if __name__ == "__main__":
    # Test the glossary
    glossary = get_glossary()

    # Test detection
    test_queries = [
        "What is PTO?",
        "How do I set billable rates?",
        "SSO integration",
        "Create a timesheet",
    ]

    for q in test_queries:
        detected = glossary.detect_terms(q)
        expanded = glossary.expand_query(q)
        print(f"Query: {q}")
        print(f"  Detected: {detected}")
        print(f"  Expanded: {expanded}")
        print()
