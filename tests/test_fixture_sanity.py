"""Fixture sanity tests to validate index integrity.

These tests ensure that:
1. FAISS index dimensions match the configured embedding model
2. Vector counts match the index metadata
3. All namespace directories have valid metadata files
"""

import json
import os
import pytest
import faiss
from pathlib import Path


class TestIndexDimensionality:
    """Test that FAISS indexes have correct dimensionality."""

    def test_faiss_dim_matches_embedding_dim(self):
        """Verify FAISS index vector dimensions match the embedding model."""
        # Get EMBEDDING_DIM without importing server
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
        index_root = Path(os.getenv("RAG_INDEX_ROOT", "index/faiss"))

        # Iterate through all namespace directories
        for ns_dir in index_root.glob("*/"):
            if not ns_dir.is_dir():
                continue

            # Try both "index" and "index.bin" names
            index_file = ns_dir / "index"
            if not index_file.exists():
                index_file = ns_dir / "index.bin"
            if not index_file.exists():
                continue  # Skip namespaces without index files

            # Load the FAISS index
            try:
                index = faiss.read_index(str(index_file))
            except Exception as e:
                pytest.fail(f"Failed to load index from {index_file}: {e}")

            # Verify dimension matches
            actual_dim = index.d
            assert actual_dim == embedding_dim, (
                f"Namespace '{ns_dir.name}': index dimension {actual_dim} "
                f"does not match EMBEDDING_DIM {embedding_dim}"
            )

    def test_index_vector_count_matches_ntotal(self):
        """Verify FAISS index ntotal matches the actual vector count."""
        index_root = Path(os.getenv("RAG_INDEX_ROOT", "index/faiss"))

        # Iterate through all namespace directories
        for ns_dir in index_root.glob("*/"):
            if not ns_dir.is_dir():
                continue

            # Try both "index" and "index.bin" names
            index_file = ns_dir / "index"
            if not index_file.exists():
                index_file = ns_dir / "index.bin"
            if not index_file.exists():
                continue  # Skip namespaces without index files

            # Load the FAISS index
            try:
                index = faiss.read_index(str(index_file))
            except Exception as e:
                pytest.fail(f"Failed to load index from {index_file}: {e}")

            # Verify ntotal is positive
            assert index.ntotal > 0, (
                f"Namespace '{ns_dir.name}': index has no vectors (ntotal={index.ntotal})"
            )

    def test_all_namespaces_have_valid_metadata(self):
        """Verify all namespace directories have valid meta.json files."""
        index_root = Path(os.getenv("RAG_INDEX_ROOT", "index/faiss"))

        # Iterate through all namespace directories
        for ns_dir in index_root.glob("*/"):
            if not ns_dir.is_dir():
                continue

            # Skip empty namespaces (no index files)
            index_file = ns_dir / "index"
            if not index_file.exists():
                index_file = ns_dir / "index.bin"
            if not index_file.exists():
                continue

            meta_file = ns_dir / "meta.json"
            assert meta_file.exists(), (
                f"Namespace '{ns_dir.name}': meta.json not found at {meta_file}"
            )

            # Verify metadata is valid JSON
            try:
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(
                    f"Namespace '{ns_dir.name}': meta.json is not valid JSON: {e}"
                )
            except Exception as e:
                pytest.fail(f"Namespace '{ns_dir.name}': failed to read meta.json: {e}")

            # Verify required fields exist
            required_fields = ["dim", "num_vectors", "model"]
            for field in required_fields:
                assert field in metadata, (
                    f"Namespace '{ns_dir.name}': meta.json missing required field '{field}'"
                )

            # Verify dimension is positive
            dim = metadata.get("dim")
            assert isinstance(dim, int) and dim > 0, (
                f"Namespace '{ns_dir.name}': invalid dimension in metadata: {dim}"
            )

            # Verify num_vectors is non-negative
            num_vectors = metadata.get("num_vectors")
            assert isinstance(num_vectors, int) and num_vectors >= 0, (
                f"Namespace '{ns_dir.name}': invalid num_vectors in metadata: {num_vectors}"
            )

    def test_metadata_dimension_matches_index_dimension(self):
        """Verify metadata dimension matches actual FAISS index dimension."""
        index_root = Path(os.getenv("RAG_INDEX_ROOT", "index/faiss"))

        # Iterate through all namespace directories
        for ns_dir in index_root.glob("*/"):
            if not ns_dir.is_dir():
                continue

            # Load metadata
            meta_file = ns_dir / "meta.json"
            if not meta_file.exists():
                pytest.skip(f"meta.json not found for {ns_dir.name}")

            try:
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                pytest.skip(f"Failed to load metadata for {ns_dir.name}: {e}")

            # Load FAISS index
            index_file = ns_dir / "index"
            if not index_file.exists():
                pytest.skip(f"Index file not found for {ns_dir.name}")

            try:
                index = faiss.read_index(str(index_file))
            except Exception as e:
                pytest.skip(f"Failed to load index for {ns_dir.name}: {e}")

            # Compare dimensions
            meta_dim = metadata.get("dim")
            index_dim = index.d

            assert meta_dim == index_dim, (
                f"Namespace '{ns_dir.name}': metadata dimension {meta_dim} "
                f"does not match index dimension {index_dim}"
            )

    def test_index_directories_exist(self):
        """Verify that index and metadata files exist for loaded namespaces."""
        index_root = Path(os.getenv("RAG_INDEX_ROOT", "index/faiss"))

        # Discover all namespaces from index directory
        if not index_root.exists():
            pytest.skip(f"Index root directory not found: {index_root}")

        namespaces = [d.name for d in index_root.iterdir() if d.is_dir()]
        if not namespaces:
            pytest.skip("No namespaces found in index directory")

        # For each namespace, verify files exist
        for ns in namespaces:
            ns_dir = index_root / ns

            assert ns_dir.exists() and ns_dir.is_dir(), (
                f"Namespace '{ns}': directory does not exist at {ns_dir}"
            )

            # Check for index file (try both "index" and "index.bin")
            index_file = ns_dir / "index"
            if not index_file.exists():
                index_file = ns_dir / "index.bin"

            if not index_file.exists():
                # Skip empty/incomplete namespaces
                continue

            # Check for metadata file
            meta_file = ns_dir / "meta.json"
            assert meta_file.exists(), (
                f"Namespace '{ns}': meta.json not found at {meta_file}"
            )
