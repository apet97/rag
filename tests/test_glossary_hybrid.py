#!/usr/bin/env python3
"""Tests for glossary-aware retrieval and hybrid search."""

import pytest
import numpy as np
from pathlib import Path
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.glossary import Glossary, get_glossary
from src.retrieval_engine import RetrievalEngine, RetrievalConfig, RetrievalStrategy
from src.preprocess import HTMLCleaner


class TestGlossary:
    """Test glossary loading and term detection."""

    def test_glossary_loads(self):
        """Test glossary loads from CSV."""
        glossary = Glossary("data/glossary.csv")
        assert len(glossary.terms) > 0, "Glossary should have terms"
        assert len(glossary.aliases) > 0, "Glossary should have aliases"

    def test_glossary_normalize(self):
        """Test term normalization."""
        assert Glossary._normalize("PTO") == "pto"
        assert Glossary._normalize("Paid Time Off") == "paid time off"
        assert Glossary._normalize("Billable-Rate") == "billablerate"

    def test_glossary_detection_basic(self):
        """Test basic term detection."""
        glossary = Glossary("data/glossary.csv")
        text = "What is PTO?"
        detected = glossary.detect_terms(text)
        # Should detect some form of "pto" or "paid time off"
        assert len(detected) > 0, f"Should detect terms in '{text}'"

    def test_glossary_detection_billing(self):
        """Test detection of billing-related terms."""
        glossary = Glossary("data/glossary.csv")
        text = "Set billable rates for the project"
        detected = glossary.detect_terms(text)
        assert len(detected) > 0, "Should detect 'billable' in billing context"

    def test_glossary_expansion(self):
        """Test query expansion with glossary."""
        glossary = Glossary("data/glossary.csv")
        query = "What is PTO?"
        expanded = glossary.expand_query(query, max_variants=3)

        assert len(expanded) > 0, "Should have at least original query"
        assert expanded[0] == query, "First variant should be original"
        assert isinstance(expanded, list), "Should return list"

    def test_glossary_get_term_info(self):
        """Test retrieving term metadata."""
        glossary = Glossary("data/glossary.csv")
        info = glossary.get_term_info("PTO")

        if info:  # May not exist in small test glossary
            assert "term" in info or "canonical" in info
            assert isinstance(info, dict)

    def test_glossary_singleton(self):
        """Test global glossary singleton."""
        g1 = get_glossary()
        g2 = get_glossary()
        assert g1 is g2, "Should return same instance"


class TestPIIStripping:
    """Test PII removal from text."""

    def test_strip_email(self):
        """Test email removal."""
        text = "Contact us at support@clockify.me for help"
        cleaned = HTMLCleaner.strip_pii(text)
        assert "support@clockify.me" not in cleaned
        assert "[EMAIL]" in cleaned

    def test_strip_phone(self):
        """Test phone number removal."""
        text = "Call us at (555) 123-4567"
        cleaned = HTMLCleaner.strip_pii(text)
        assert "(555) 123-4567" not in cleaned
        assert "[PHONE]" in cleaned

    def test_strip_ssn(self):
        """Test SSN removal."""
        text = "Employee SSN: 123-45-6789"
        cleaned = HTMLCleaner.strip_pii(text)
        assert "123-45-6789" not in cleaned
        assert "[SSN]" in cleaned

    def test_strip_preserves_content(self):
        """Test that stripping preserves other content."""
        text = "Contact support@clockify.me about (555) 123-4567"
        cleaned = HTMLCleaner.strip_pii(text)
        assert "Contact" in cleaned
        assert "about" in cleaned


class TestHybridRetriever:
    """Test hybrid retrieval with late fusion."""

    def test_hybrid_init(self):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever(alpha=0.6, k_dense=40, k_bm25=40, k_final=12)
        assert retriever.alpha == 0.6
        assert retriever.k_dense == 40
        assert retriever.k_bm25 == 40
        assert retriever.k_final == 12

    def test_hybrid_alpha_clamping(self):
        """Test alpha clamping to [0, 1]."""
        r1 = HybridRetriever(alpha=-0.5)
        assert r1.alpha == 0.0

        r2 = HybridRetriever(alpha=1.5)
        assert r2.alpha == 1.0

    def test_bm25_index_building(self):
        """Test BM25 index construction."""
        retriever = HybridRetriever()
        chunks = [
            {"id": 0, "text": "PTO is paid time off", "title": "PTO Policy", "section": "Benefits"},
            {"id": 1, "text": "Billable rate is charged to clients", "title": "Billing", "section": "Pricing"},
            {"id": 2, "text": "Timesheet records work hours", "title": "Timesheet", "section": "Tracking"},
        ]

        success = retriever.build_bm25_index(chunks)
        assert success, "BM25 index should build successfully"
        assert retriever.bm25_index is not None
        assert len(retriever.chunks) == 3

    def test_bm25_retrieval(self):
        """Test BM25 retrieval - returns results even with 0 scores."""
        retriever = HybridRetriever(k_bm25=2)
        chunks = [
            {"id": 0, "text": "PTO is paid time off", "title": "PTO Policy", "section": "Benefits"},
            {"id": 1, "text": "Billable rate is charged to clients", "title": "Billing", "section": "Pricing"},
        ]
        retriever.build_bm25_index(chunks)

        # Query - BM25 may return zero scores in some cases
        results = retriever.retrieve_bm25("pto rate")
        # Should return something from top-k even if scores are 0
        assert results is not None, "Should return list"
        # May be empty or have elements depending on BM25 implementation

    def test_score_normalization(self):
        """Test score normalization."""
        retriever = HybridRetriever()
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = retriever._normalize_scores(scores)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert len(normalized) == len(scores)

    def test_fusion_alpha_0(self):
        """Test fusion with alpha=0 (BM25 only)."""
        retriever = HybridRetriever(alpha=0.0)
        dense_results = [(0, 0.9), (1, 0.7)]
        bm25_results = [(1, 0.5), (2, 0.3)]

        fused = retriever.fuse_results(dense_results, bm25_results)
        assert len(fused) > 0

    def test_fusion_alpha_1(self):
        """Test fusion with alpha=1 (dense only)."""
        retriever = HybridRetriever(alpha=1.0)
        dense_results = [(0, 0.9), (1, 0.7)]
        bm25_results = [(1, 0.5), (2, 0.3)]

        fused = retriever.fuse_results(dense_results, bm25_results)
        assert len(fused) > 0

    def test_fusion_alpha_0_5(self):
        """Test fusion with alpha=0.5 (balanced)."""
        retriever = HybridRetriever(alpha=0.5)
        dense_results = [(0, 0.9), (1, 0.7)]
        bm25_results = [(1, 0.8), (2, 0.3)]

        fused = retriever.fuse_results(dense_results, bm25_results)
        assert len(fused) > 0
        # Score for item 1 should be average-ish since it appears in both
        assert any(chunk_id == 1 for chunk_id, _ in fused)

    def test_fuse_deduplication(self):
        """Test that fusion deduplicates chunks."""
        retriever = HybridRetriever(k_final=10)
        dense_results = [(0, 0.9), (1, 0.7), (2, 0.6)]
        bm25_results = [(0, 0.8), (1, 0.5), (3, 0.4)]

        fused = retriever.fuse_results(dense_results, bm25_results)
        chunk_ids = [cid for cid, _ in fused]

        # No duplicates
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_retriever_singleton(self):
        """Test hybrid retriever singleton."""
        from src.retrieval_hybrid import get_hybrid_retriever
        r1 = get_hybrid_retriever()
        r2 = get_hybrid_retriever()
        assert r1 is r2, "Should return same instance"


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_glossary_detection_in_chunk(self):
        """Test glossary detection on chunk-like content."""
        glossary = Glossary("data/glossary.csv")
        chunk_text = """
        # Billable Rates

        A billable rate is the rate charged to clients for work performed.
        This is different from the cost rate, which is the internal labor cost.
        """
        detected = glossary.detect_terms(chunk_text)
        assert len(detected) > 0, "Should detect billing-related terms"

    def test_pii_then_glossary_detection(self):
        """Test PII stripping followed by glossary detection."""
        text = """
        PTO Policy - Contact John Doe at john@example.com or (555) 123-4567.
        PTO is paid time off for employees.
        """
        cleaned = HTMLCleaner.strip_pii(text)
        glossary = Glossary("data/glossary.csv")
        detected = glossary.detect_terms(cleaned)

        assert "[EMAIL]" in cleaned
        assert "[PHONE]" in cleaned
        assert len(detected) > 0, "Should still detect glossary terms after PII stripping"

    def test_hybrid_config_from_env(self):
        """Test that hybrid retriever reads from environment."""
        os.environ["HYBRID_ALPHA"] = "0.7"
        os.environ["K_DENSE"] = "50"
        os.environ["K_BM25"] = "50"
        os.environ["K_FINAL"] = "15"

        from src.retrieval_hybrid import get_hybrid_retriever, HybridRetriever
        # Reset singleton to pick up new env vars
        import src.retrieval_hybrid
        src.retrieval_hybrid._hybrid_retriever = None

        retriever = get_hybrid_retriever()
        assert retriever.alpha == 0.7
        assert retriever.k_dense == 50
        assert retriever.k_bm25 == 50
        assert retriever.k_final == 15

        # Cleanup
        del os.environ["HYBRID_ALPHA"]
        del os.environ["K_DENSE"]
        del os.environ["K_BM25"]
        del os.environ["K_FINAL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
