#!/usr/bin/env python3
"""
Direct RAG test script - bypasses HTTP middleware issues.

This script tests the RAG system directly without using the HTTP server,
which is useful when middleware has compatibility issues.

Usage:
    python test_rag.py
    python test_rag.py "your custom question"
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.index_manager import IndexManager
from src.embeddings import embed_query, get_embedder
from src.config import NAMESPACE, INDEX_ROOT, NAMESPACES
import numpy as np
import faiss


def test_search(query: str, k: int = 5):
    """Test search functionality directly."""
    print("=" * 80)
    print("RAG DIRECT TEST (Bypassing HTTP Server)")
    print("=" * 80)
    print()

    print(f"📝 Query: {query}")
    print()

    # Initialize
    print("🔧 Initializing...")
    try:
        # Load embedder
        embedder = get_embedder()
        print(f"✓ Embedding model loaded")

        # Load index manager
        mgr = IndexManager(INDEX_ROOT, NAMESPACES)
        mgr.ensure_loaded()
        print(f"✓ Index manager loaded")
        print()

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Embed query
    print("🔍 Embedding query...")
    try:
        query_vec = embed_query(query)
        # Convert to numpy array if needed
        if not isinstance(query_vec, np.ndarray):
            query_vec = np.array(query_vec, dtype=np.float32)
        print(f"✓ Query embedded (dimension={query_vec.shape})")
        print()
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return

    # Search
    print(f"🔎 Searching index (namespace={NAMESPACES[0]})...")
    try:
        namespace = NAMESPACES[0]

        # Get FAISS index and metadata
        idx_data = mgr.get_index(namespace)
        faiss_index = idx_data["index"]
        metadata = idx_data["metas"]

        # Prepare query vector for FAISS (needs 2D array: shape (1, dim))
        if query_vec.ndim == 1:
            query_vec_2d = query_vec.reshape(1, -1).astype(np.float32)
        else:
            query_vec_2d = query_vec.astype(np.float32)

        # Normalize if index is normalized
        if mgr.is_normalized(namespace):
            faiss.normalize_L2(query_vec_2d)

        # Search FAISS index
        scores, indices = faiss_index.search(query_vec_2d, k)

        # Build results with metadata
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(metadata):
                results.append((int(idx), float(score)))

        print(f"✓ Search complete: Found {len(results)} results")
        print()
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Display results
    print("📊 SEARCH RESULTS")
    print("=" * 80)
    print()

    if not results:
        print("No results found.")
        return

    for i, (doc_id, score) in enumerate(results, 1):
        try:
            meta = metadata[doc_id]

            print(f"Result #{i}")
            print(f"  Score: {score:.4f}")
            print(f"  URL: {meta.get('url', 'N/A')}")

            title = meta.get('title', meta.get('h1', 'N/A'))
            if len(title) > 100:
                title = title[:97] + "..."
            print(f"  Title: {title}")

            content = meta.get('content', meta.get('text', ''))
            if content:
                preview = content[:200].replace('\n', ' ')
                if len(content) > 200:
                    preview += "..."
                print(f"  Preview: {preview}")

            print()

        except Exception as e:
            print(f"  ❌ Error retrieving metadata: {e}")
            print()

    print("=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)


def test_chat(query: str):
    """Test full RAG chat with LLM (if available)."""
    print("=" * 80)
    print("RAG CHAT TEST (With LLM)")
    print("=" * 80)
    print()

    print(f"📝 Query: {query}")
    print()

    # Import components
    from src.llm_client import LLMClient
    from src.prompt import RAGPrompt

    # Search first
    print("🔍 Step 1: Retrieving relevant documents...")
    try:
        embedder = get_embedder()
        mgr = IndexManager(INDEX_ROOT, NAMESPACES)
        mgr.ensure_loaded()

        query_vec = embed_query(query)
        # Convert to numpy array if needed
        if not isinstance(query_vec, np.ndarray):
            query_vec = np.array(query_vec, dtype=np.float32)

        namespace = NAMESPACES[0]

        # Get FAISS index and metadata
        idx_data = mgr.get_index(namespace)
        faiss_index = idx_data["index"]
        metadata = idx_data["metas"]

        # Prepare query vector for FAISS (needs 2D array: shape (1, dim))
        if query_vec.ndim == 1:
            query_vec_2d = query_vec.reshape(1, -1).astype(np.float32)
        else:
            query_vec_2d = query_vec.astype(np.float32)

        # Normalize if index is normalized
        if mgr.is_normalized(namespace):
            faiss.normalize_L2(query_vec_2d)

        # Search FAISS index
        scores, indices = faiss_index.search(query_vec_2d, 5)

        # Build results with metadata
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(metadata):
                results.append((int(idx), float(score)))

        print(f"✓ Found {len(results)} relevant documents")
        print()

    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Build context chunks for RAG prompt
    print("📚 Step 2: Building context...")
    chunks = []

    for i, (doc_id, score) in enumerate(results, 1):
        try:
            meta = metadata[doc_id]
            content = meta.get('content', meta.get('text', ''))

            if content:
                # Build chunk dict with all metadata
                chunk = {
                    "text": content,
                    "score": score,
                    "rank": i,
                    **meta  # Include all original metadata (url, title, etc.)
                }
                chunks.append(chunk)
        except Exception:
            pass

    print(f"✓ Context built ({len(chunks)} chunks)")
    print()

    # Generate answer
    print("🤖 Step 3: Generating answer with LLM...")
    try:
        llm = LLMClient()

        # Build prompt (returns tuple: messages, sources, developer_instructions)
        prompt = RAGPrompt()
        messages, sources_list, dev_instructions = prompt.build_messages(
            question=query,
            chunks=chunks,
            namespace="clockify"
        )

        # Call LLM
        answer = llm.chat(messages)

        print("✓ Answer generated")
        print()

    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        print()
        print("💡 Tip: Set MOCK_LLM=true in .env for offline testing")
        return

    # Display answer
    print("💬 ANSWER")
    print("=" * 80)
    print(answer)
    print()
    print("=" * 80)

    print()
    print("📖 SOURCES")
    print("=" * 80)
    for i, source in enumerate(sources_list, 1):
        title = source.get('title', source.get('h1', 'N/A'))
        url = source.get('url', 'N/A')
        print(f"[{i}] {title}")
        print(f"    {url}")
    print("=" * 80)


def main():
    """Main entry point."""
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "How do I create a project in Clockify?"

    # Run search test
    test_search(query, k=5)

    # Ask if user wants to test LLM
    print()
    try:
        response = input("Run full RAG test with LLM? (y/N): ").strip().lower()
        if response in ('y', 'yes'):
            print()
            test_chat(query)
    except (KeyboardInterrupt, EOFError):
        print()
        print("Skipping LLM test.")


if __name__ == "__main__":
    main()
