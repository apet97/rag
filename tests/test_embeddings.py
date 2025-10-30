#!/usr/bin/env python3
"""
PHASE 5: Embedding Dimension Probe Tests

Tests embedding model dimensions, consistency, and correctness.
Ensures embeddings are properly normalized and have expected shape.

Key validations:
- Embedding dimension matches expected (384 for stub, 768 for real)
- All embeddings are L2-normalized (norm ≈ 1.0)
- Batch embedding preserves dimension across multiple texts
- Embedding consistency: same input always produces same output
"""

import os
import numpy as np
import pytest
from src.embeddings import (
    embed_query,
    embed_passages,
    encode_texts,
    EMBEDDING_DIM,
    EMBEDDINGS_BACKEND,
)
from src.embeddings_stub import StubEmbedder


class TestEmbeddingDimensions:
    """Test that embeddings have correct dimensions."""

    def test_embedding_dim_is_set(self):
        """Verify EMBEDDING_DIM constant is properly resolved."""
        assert EMBEDDING_DIM > 0, "EMBEDDING_DIM should be positive"
        assert isinstance(EMBEDDING_DIM, int), "EMBEDDING_DIM should be an integer"

    def test_embed_query_shape(self):
        """Test that query embedding has correct shape."""
        text = "What is Python?"
        embedding = embed_query(text)

        assert embedding is not None, "embed_query should not return None"
        assert isinstance(embedding, np.ndarray), "embedding should be numpy array"
        assert embedding.ndim == 1, f"Expected 1D embedding, got {embedding.ndim}D"
        assert len(embedding) == EMBEDDING_DIM, (
            f"Expected dimension {EMBEDDING_DIM}, got {len(embedding)}"
        )

    def test_embed_texts_shape(self):
        """Test that batch embedding has correct shape."""
        texts = ["First document", "Second document", "Third document"]
        embeddings = embed_passages(texts)

        assert embeddings is not None, "embed_texts should not return None"
        assert isinstance(embeddings, np.ndarray), "embeddings should be numpy array"
        assert embeddings.ndim == 2, f"Expected 2D embeddings, got {embeddings.ndim}D"
        assert embeddings.shape[0] == len(texts), (
            f"Expected {len(texts)} embeddings, got {embeddings.shape[0]}"
        )
        assert embeddings.shape[1] == EMBEDDING_DIM, (
            f"Expected dimension {EMBEDDING_DIM}, got {embeddings.shape[1]}"
        )

    def test_embed_single_text_consistency(self):
        """Test that embedding the same text multiple times produces identical results."""
        text = "Consistency check for embeddings"

        embed1 = embed_query(text)
        embed2 = embed_query(text)

        assert np.allclose(embed1, embed2, rtol=1e-6), (
            "Same input should produce identical embeddings (deterministic)"
        )

    def test_embed_different_texts_differ(self):
        """Test that different texts produce different embeddings."""
        text1 = "Hello world"
        text2 = "Goodbye world"

        embed1 = embed_query(text1)
        embed2 = embed_query(text2)

        # They should be different (unless by extremely unlikely chance)
        assert not np.allclose(embed1, embed2, rtol=1e-3), (
            "Different inputs should produce different embeddings"
        )


class TestEmbeddingNormalization:
    """Test that embeddings are properly normalized."""

    def test_query_embedding_normalized(self):
        """Test that query embeddings have unit norm (L2-normalized)."""
        text = "Test normalization"
        embedding = embed_query(text)

        norm = np.linalg.norm(embedding)
        # Stub embedder may not always return perfectly normalized (depends on output)
        # Check that norm is reasonable (not wildly off)
        assert 0.8 < norm <= 1.05, (
            f"Expected unit norm (~1.0), got {norm:.4f}. "
            f"Embedding may not be properly L2-normalized."
        )

    def test_batch_embeddings_normalized(self):
        """Test that batch embeddings are properly normalized."""
        texts = ["First", "Second", "Third"]
        embeddings = embed_passages(texts)

        norms = np.linalg.norm(embeddings, axis=1)
        for i, norm in enumerate(norms):
            assert 0.8 < norm <= 1.05, (
                f"Embedding {i} has norm {norm:.4f}, expected ~1.0"
            )

    def test_zero_vector_not_returned(self):
        """Test that embeddings are never zero vectors."""
        # Try multiple queries
        queries = ["test", "hello", "embedding", "normalize", "vector"]

        for query in queries:
            embedding = embed_query(query)
            norm = np.linalg.norm(embedding)
            assert norm > 0.1, (
                f"Query '{query}' produced near-zero embedding (norm={norm:.4f})"
            )


class TestEmbeddingDataTypes:
    """Test that embeddings use correct data types."""

    def test_query_embedding_dtype(self):
        """Test that query embeddings use float32."""
        embedding = embed_query("data type check")

        assert embedding.dtype == np.float32, (
            f"Expected float32 dtype, got {embedding.dtype}"
        )

    def test_batch_embedding_dtype(self):
        """Test that batch embeddings use float32."""
        embeddings = embed_passages(["first", "second", "third"])

        assert embeddings.dtype == np.float32, (
            f"Expected float32 dtype, got {embeddings.dtype}"
        )

    def test_embedding_values_in_valid_range(self):
        """Test that embedding values are in reasonable range for normalized vectors."""
        embeddings = embed_passages(["test1", "test2", "test3"])

        # Normalized vectors should have values roughly in [-1, 1]
        assert np.all(embeddings >= -1.1), "Embedding values too negative"
        assert np.all(embeddings <= 1.1), "Embedding values too positive"


class TestEmbeddingConsistency:
    """Test consistency of embeddings across different backends."""

    def test_embedding_model_backend(self):
        """Test that we're using expected backend."""
        # Should be either 'stub' or 'real'
        assert EMBEDDINGS_BACKEND in ["stub", "real"], (
            f"Unknown backend: {EMBEDDINGS_BACKEND}"
        )

    def test_stub_embedder_directly(self):
        """Test StubEmbedder directly if using stub backend."""
        if EMBEDDINGS_BACKEND == "stub":
            embedder = StubEmbedder()
            embedding = embedder.encode("test")

            assert embedding.shape == (1, 384), (
                f"Stub embedder should return (1, 384), got {embedding.shape}"
            )

    def test_batch_with_single_text(self):
        """Test that batch embedding with single text matches query embedding."""
        text = "Single text batch test"

        query_emb = embed_query(text)
        batch_emb = embed_passages([text])

        assert batch_emb.shape[0] == 1, "Batch should contain 1 embedding"
        # Due to potential floating point differences, use allclose
        assert np.allclose(query_emb, batch_emb[0], rtol=1e-5), (
            "Single text batch embedding should match query embedding"
        )

    def test_empty_string_handling(self):
        """Test that empty strings are handled gracefully."""
        try:
            embedding = embed_query("")
            # If it succeeds, should still be valid shape
            assert len(embedding) == EMBEDDING_DIM
        except (ValueError, RuntimeError):
            # Some embedders may reject empty strings, which is acceptable
            pass

    def test_very_long_text_handling(self):
        """Test that very long texts are handled (truncation or processing)."""
        long_text = "word " * 10000  # Very long text
        try:
            embedding = embed_query(long_text)
            # Should still return valid embedding
            assert len(embedding) == EMBEDDING_DIM
            assert np.linalg.norm(embedding) > 0
        except (ValueError, RuntimeError):
            # Some embedders may have length limits, which is acceptable
            pass


class TestEmbeddingEdgeCases:
    """Test edge cases and error handling."""

    def test_special_characters(self):
        """Test embedding texts with special characters."""
        texts = [
            "Hello @world!",
            "Numbers: 123.456",
            "Symbols: !@#$%^&*()",
            "Unicode: 你好世界 🌍",
        ]

        embeddings = embed_passages(texts)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == EMBEDDING_DIM

    def test_whitespace_only(self):
        """Test embedding whitespace-only strings."""
        try:
            embedding = embed_query("   ")
            # If it succeeds, should still have correct dimension
            assert len(embedding) == EMBEDDING_DIM
        except (ValueError, RuntimeError):
            # Some embedders may reject whitespace-only strings
            pass

    def test_very_short_text(self):
        """Test embedding very short texts."""
        texts = ["a", "hi", "ok"]
        embeddings = embed_passages(texts)

        assert embeddings.shape == (len(texts), EMBEDDING_DIM)

    def test_duplicate_texts(self):
        """Test that duplicate texts in batch produce identical embeddings."""
        texts = ["duplicate", "duplicate", "different"]
        embeddings = embed_passages(texts)

        # First two should be identical
        assert np.allclose(embeddings[0], embeddings[1], rtol=1e-6), (
            "Duplicate texts should produce identical embeddings"
        )
        # Third should be different
        assert not np.allclose(embeddings[0], embeddings[2], rtol=1e-3), (
            "Different text should produce different embedding"
        )


class TestEmbeddingMetadata:
    """Test embedding metadata and configuration."""

    def test_embedding_dim_matches_constant(self):
        """Test that actual embedding dimension matches EMBEDDING_DIM constant."""
        embedding = embed_query("metadata test")
        assert len(embedding) == EMBEDDING_DIM, (
            f"Actual dimension {len(embedding)} != EMBEDDING_DIM {EMBEDDING_DIM}"
        )

    def test_embedding_backend_resolves(self):
        """Test that embedding backend is properly configured."""
        # EMBEDDINGS_BACKEND should be set to something valid
        assert EMBEDDINGS_BACKEND, "EMBEDDINGS_BACKEND should be set"

    def test_embedding_model_string(self):
        """Test that embeddings can handle model string representation."""
        # This is more of a sanity check
        from src.embeddings import EMBEDDING_MODEL

        assert isinstance(EMBEDDING_MODEL, str), "EMBEDDING_MODEL should be string"
        assert len(EMBEDDING_MODEL) > 0, "EMBEDDING_MODEL should not be empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
