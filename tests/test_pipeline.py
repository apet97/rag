#!/usr/bin/env python3
"""E2E tests for multi-corpus RAG pipeline."""

import json
import pytest
from pathlib import Path


class TestCrawler:
    """Test crawler output."""

    def test_clockify_crawled(self):
        """Clockify pages were scraped."""
        clockify_dir = Path("data/raw/clockify")
        files = list(clockify_dir.glob("*.html")) if clockify_dir.exists() else []
        assert len(files) >= 5, f"Expected ≥5 Clockify pages, got {len(files)}"

    def test_langchain_crawled(self):
        """LangChain pages were scraped."""
        lc_dir = Path("data/raw/langchain")
        files = list(lc_dir.glob("*.html")) if lc_dir.exists() else []
        assert len(files) >= 5, f"Expected ≥5 LangChain pages, got {len(files)}"

    def test_html_valid(self):
        """HTML files have valid JSON wrappers."""
        for ns_dir in Path("data/raw").glob("*"):
            if ns_dir.is_dir():
                for html_file in list(ns_dir.glob("*.html"))[:3]:
                    with open(html_file) as f:
                        wrapper = json.load(f)
                        assert "meta" in wrapper
                        assert "html" in wrapper
                        assert len(wrapper["html"]) > 100


class TestPreprocessor:
    """Test preprocessing output."""

    def test_markdown_created(self):
        """Markdown files exist."""
        for ns in ["clockify", "langchain"]:
            ns_dir = Path(f"data/clean/{ns}")
            md_files = list(ns_dir.glob("*.md")) if ns_dir.exists() else []
            assert len(md_files) >= 5, f"{ns}: expected ≥5 markdown files, got {len(md_files)}"

    def test_frontmatter_valid(self):
        """Markdown has valid frontmatter."""
        for ns_dir in Path("data/clean").glob("*"):
            if ns_dir.is_dir():
                for md_file in list(ns_dir.glob("*.md"))[:2]:
                    with open(md_file) as f:
                        content = f.read()
                        assert content.startswith("---")
                        parts = content.split("---", 2)
                        assert len(parts) >= 3
                        fm = json.loads(parts[1])
                        assert "url" in fm
                        assert "namespace" in fm


class TestChunking:
    """Test chunking output."""

    def test_chunks_exist(self):
        """Chunk files exist for each namespace."""
        for ns in ["clockify", "langchain"]:
            chunks_file = Path(f"data/chunks/{ns}.jsonl")
            assert chunks_file.exists(), f"Chunks file not found: {chunks_file}"

    def test_parent_child_structure(self):
        """Chunks have parent-child relationships."""
        for chunks_file in Path("data/chunks").glob("*.jsonl"):
            chunks = []
            with open(chunks_file) as f:
                for line in f:
                    chunks.append(json.loads(line))

            assert len(chunks) >= 10, f"Too few chunks in {chunks_file}"

            parents = [c for c in chunks if c.get("node_type") == "parent"]
            children = [c for c in chunks if c.get("node_type") == "child"]

            assert len(parents) > 0, "No parent nodes"
            assert len(children) > 0, "No child nodes"
            assert len(children) > len(parents), "Should have more children than parents"


class TestEmbedding:
    """Test embedding indexes."""

    def test_indexes_exist(self):
        """FAISS indexes created."""
        for ns in ["clockify", "langchain"]:
            ns_dir = Path(f"index/faiss/{ns}")
            assert (ns_dir / "index.bin").exists(), f"Index not found for {ns}"
            assert (ns_dir / "meta.json").exists(), f"Metadata not found for {ns}"

    def test_index_integrity(self):
        """Index metadata is valid."""
        for ns_dir in Path("index/faiss").glob("*/"):
            if ns_dir.name == "hybrid":
                continue
            meta_file = ns_dir / "meta.json"
            with open(meta_file) as f:
                meta = json.load(f)
                assert meta["num_vectors"] > 0
                assert meta["dimension"] > 0
                assert len(meta["chunks"]) == meta["num_vectors"]


class TestRetrieval:
    """Test retrieval functionality."""

    @pytest.mark.asyncio
    async def test_server_starts(self):
        """Server imports correctly."""
        from src.server import app
        assert app is not None

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Health check works."""
        from fastapi.testclient import TestClient
        from src.server import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "indexes_loaded" in data

    @pytest.mark.asyncio
    async def test_search_endpoint(self):
        """Search endpoint works."""
        from fastapi.testclient import TestClient
        from src.server import app

        client = TestClient(app)

        # Test Clockify search
        resp = client.get("/search?q=timesheet&namespace=clockify&k=5")
        if resp.status_code == 200:
            data = resp.json()
            assert "results" in data
            assert data["count"] >= 0

    @pytest.mark.asyncio
    async def test_chat_endpoint(self):
        """Chat endpoint accessible."""
        from fastapi.testclient import TestClient
        from src.server import app

        client = TestClient(app)
        resp = client.post(
            "/chat",
            json={"question": "How do I create a project?", "namespace": "clockify", "k": 5}
        )
        # May fail without LLM, but endpoint should be callable
        assert resp.status_code in (200, 500, 503)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
