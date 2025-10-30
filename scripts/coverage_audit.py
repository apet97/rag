#!/usr/bin/env python3
"""
Coverage audit script: Compute category distribution, breadcrumb health, and chunk statistics.

Generates a comprehensive analysis of documentation coverage to track quality over time.
Output is JSON for easy parsing and historical tracking.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import sys

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_chunks(namespace: str = "clockify") -> List[Dict]:
    """Load all chunks for a namespace."""
    chunks_file = Path(f"data/chunks/{namespace}.jsonl")
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return chunks


def analyze_breadcrumbs(chunks: List[Dict]) -> Dict:
    """Analyze breadcrumb distribution and health."""
    breadcrumb_lengths = Counter()
    category_chunks = defaultdict(int)
    subcategory_chunks = defaultdict(int)
    fallback_count = 0

    for chunk in chunks:
        bc = chunk.get("breadcrumb", [])
        breadcrumb_lengths[len(bc)] += 1

        # Check for fallback breadcrumbs (just ["Clockify Help Center", title])
        if len(bc) == 2 and bc[0] == "Clockify Help Center":
            fallback_count += 1

        # Count by category (second element if exists)
        if len(bc) > 1:
            category_chunks[bc[1]] += 1

        # Count by subcategory (third element if exists)
        if len(bc) > 2:
            subcategory_chunks[f"{bc[1]} > {bc[2]}"] += 1

    return {
        "total_chunks": len(chunks),
        "breadcrumb_lengths": dict(breadcrumb_lengths),
        "fallback_breadcrumbs": {
            "count": fallback_count,
            "percentage": round(100 * fallback_count / len(chunks), 1) if chunks else 0
        },
        "categories": dict(sorted(category_chunks.items(), key=lambda x: x[1], reverse=True)),
        "subcategories_sample": dict(sorted(subcategory_chunks.items(), key=lambda x: x[1], reverse=True)[:10]),
        "unique_categories": len(category_chunks),
        "unique_subcategories": len(subcategory_chunks),
    }


def analyze_sections(chunks: List[Dict]) -> Dict:
    """Analyze section hierarchy and content distribution."""
    section_stats = defaultdict(lambda: {"chunks": 0, "total_tokens": 0})

    for chunk in chunks:
        section = chunk.get("section", "Unknown")
        tokens = chunk.get("tokens", 0)
        section_stats[section]["chunks"] += 1
        section_stats[section]["total_tokens"] += tokens

    # Sort by chunk count
    sorted_sections = sorted(section_stats.items(), key=lambda x: x[1]["chunks"], reverse=True)

    return {
        "total_sections": len(section_stats),
        "top_sections": [
            {
                "section": sec,
                "chunks": stats["chunks"],
                "total_tokens": stats["total_tokens"],
                "avg_tokens": round(stats["total_tokens"] / stats["chunks"], 1) if stats["chunks"] > 0 else 0
            }
            for sec, stats in sorted_sections[:15]
        ]
    }


def analyze_tokens(chunks: List[Dict]) -> Dict:
    """Analyze token distribution and efficiency."""
    tokens = [chunk.get("tokens", 0) for chunk in chunks]
    tokens = [t for t in tokens if t > 0]  # Filter out zeros

    if not tokens:
        return {
            "total_tokens": 0,
            "avg_tokens": 0,
            "median_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0
        }

    tokens_sorted = sorted(tokens)
    total = sum(tokens)

    return {
        "total_tokens": total,
        "chunks_analyzed": len(tokens),
        "avg_tokens": round(total / len(tokens), 1),
        "median_tokens": round(tokens_sorted[len(tokens) // 2], 1),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "p95_tokens": round(tokens_sorted[int(0.95 * len(tokens))], 1),
    }


def analyze_urls(chunks: List[Dict]) -> Dict:
    """Analyze URL distribution and unique sources."""
    urls = Counter()
    unique_urls = set()

    for chunk in chunks:
        url = chunk.get("url", "")
        if url:
            urls[url] += 1
            unique_urls.add(url)

    return {
        "total_chunks": len(chunks),
        "unique_urls": len(unique_urls),
        "avg_chunks_per_url": round(len(chunks) / len(unique_urls), 2) if unique_urls else 0,
        "top_urls_by_chunk_count": [
            {"url": url, "chunks": count}
            for url, count in urls.most_common(10)
        ]
    }


def analyze_chunk_types(chunks: List[Dict]) -> Dict:
    """Analyze chunk node types (parent vs child)."""
    node_types = Counter()

    for chunk in chunks:
        node_type = chunk.get("node_type", "unknown")
        node_types[node_type] += 1

    return dict(node_types)


def main():
    """Run comprehensive coverage audit."""
    import argparse

    parser = argparse.ArgumentParser(description="Coverage audit for RAG chunks")
    parser.add_argument("--namespace", default="clockify", help="Namespace to audit")
    parser.add_argument("--output", help="Output JSON file (default: print to stdout)")
    parser.add_argument("--summary", action="store_true", help="Print summary to stdout")
    args = parser.parse_args()

    try:
        print(f"Loading chunks for namespace: {args.namespace}...", file=sys.stderr)
        chunks = load_chunks(args.namespace)
        print(f"✓ Loaded {len(chunks)} chunks", file=sys.stderr)

        # Run all analyses
        print("Running analyses...", file=sys.stderr)
        audit = {
            "namespace": args.namespace,
            "total_chunks": len(chunks),
            "breadcrumbs": analyze_breadcrumbs(chunks),
            "sections": analyze_sections(chunks),
            "tokens": analyze_tokens(chunks),
            "urls": analyze_urls(chunks),
            "chunk_types": analyze_chunk_types(chunks),
        }

        # Output
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(audit, f, indent=2)
            print(f"✓ Audit saved to {args.output}", file=sys.stderr)

        if args.summary or not args.output:
            print("\n" + "="*60, file=sys.stderr)
            print(f"COVERAGE AUDIT - {args.namespace.upper()}", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print(f"Total chunks: {audit['total_chunks']}", file=sys.stderr)
            print(f"Unique URLs: {audit['urls']['unique_urls']}", file=sys.stderr)
            print(f"Unique categories: {audit['breadcrumbs']['unique_categories']}", file=sys.stderr)
            print(f"Unique subcategories: {audit['breadcrumbs']['unique_subcategories']}", file=sys.stderr)
            print(f"Total tokens: {audit['tokens']['total_tokens']}", file=sys.stderr)
            print(f"Avg tokens/chunk: {audit['tokens']['avg_tokens']}", file=sys.stderr)
            print(f"\nBreadcrumb health:", file=sys.stderr)
            print(f"  - Full breadcrumbs (3-element): {audit['breadcrumbs']['breadcrumb_lengths'].get(3, 0)} chunks", file=sys.stderr)
            print(f"  - Partial breadcrumbs (2-element): {audit['breadcrumbs']['breadcrumb_lengths'].get(2, 0)} chunks", file=sys.stderr)
            print(f"  - Fallbacks: {audit['breadcrumbs']['fallback_breadcrumbs']['percentage']}%", file=sys.stderr)
            print(f"\nTop categories by chunk count:", file=sys.stderr)
            for i, (cat, count) in enumerate(list(audit['breadcrumbs']['categories'].items())[:5], 1):
                print(f"  {i}. {cat}: {count} chunks", file=sys.stderr)
            print(f"\nTop sections by chunk count:", file=sys.stderr)
            for sec in audit['sections']['top_sections'][:3]:
                print(f"  - {sec['section']}: {sec['chunks']} chunks ({sec['avg_tokens']} tokens avg)", file=sys.stderr)
            print(f"\nChunk types:", file=sys.stderr)
            for node_type, count in audit['chunk_types'].items():
                print(f"  - {node_type}: {count}", file=sys.stderr)

        # Output JSON to stdout if no file specified and not summary-only
        if not args.output and not args.summary:
            json.dump(audit, sys.stdout, indent=2)

        print("✓ Coverage audit complete", file=sys.stderr)

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
