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
        print(f"✓ Query embedded (dimension={len(query_vec)})")
        print()
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return

    # Search
    print(f"🔎 Searching index (namespace={NAMESPACE})...")
    try:
        # Try with namespace suffix
        namespace = f"{NAMESPACE}_url" if not NAMESPACE.endswith("_url") else NAMESPACE
        results = mgr.search(namespace, query_vec, k=k)
        print(f"✓ Search complete: Found {len(results)} results")
        print()
    except Exception as e:
        print(f"❌ Search failed: {e}")
        # Try without suffix
        try:
            results = mgr.search(NAMESPACE, query_vec, k=k)
            print(f"✓ Search complete: Found {len(results)} results")
            print()
            namespace = NAMESPACE
        except Exception as e2:
            print(f"❌ Search failed (both attempts): {e2}")
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
            meta = mgr.get_metadata(namespace, doc_id)

            print(f"Result #{i}")
            print(f"  Score: {score:.4f}")
            print(f"  URL: {meta.get('url', 'N/A')}")

            title = meta.get('title', meta.get('h1', 'N/A'))
            if len(title) > 100:
                title = title[:97] + "..."
            print(f"  Title: {title}")

            content = meta.get('content', '')
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
        namespace = f"{NAMESPACE}_url" if not NAMESPACE.endswith("_url") else NAMESPACE
        results = mgr.search(namespace, query_vec, k=5)

        print(f"✓ Found {len(results)} relevant documents")
        print()

    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return

    # Build context
    print("📚 Step 2: Building context...")
    context_parts = []
    sources = []

    for i, (doc_id, score) in enumerate(results, 1):
        try:
            meta = mgr.get_metadata(namespace, doc_id)
            content = meta.get('content', '')
            url = meta.get('url', '')
            title = meta.get('title', meta.get('h1', ''))

            if content:
                context_parts.append(f"[{i}] {content[:500]}")
                sources.append(f"[{i}] {title} - {url}")
        except Exception:
            pass

    context = "\n\n".join(context_parts)
    print(f"✓ Context built ({len(context)} characters)")
    print()

    # Generate answer
    print("🤖 Step 3: Generating answer with LLM...")
    try:
        llm = LLMClient()

        # Build prompt
        prompt = RAGPrompt()
        messages = prompt.build_messages(query, context)

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
    for source in sources:
        print(source)
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
