#!/usr/bin/env python3
"""Diagnose missed evaluation cases to identify failure patterns.

Analyzes evaluation results to categorize misses by:
- Intent type (how_to, definition, troubleshooting, etc.)
- Retrieved document titles vs ground truth
- Glossary term coverage
- Decomposition strategy used

Usage:
    python3 eval/diagnose_misses.py logs/evals/baseline_latest.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def load_eval_results(filepath: str) -> dict:
    """Load evaluation JSON results."""
    with open(filepath) as f:
        return json.load(f)


def categorize_misses(eval_data: dict) -> Dict[str, List[dict]]:
    """Group missed cases by failure pattern."""
    misses = defaultdict(list)

    for detail in eval_data.get("details", []):
        if not detail.get("recall@5", False):  # Missed case
            case_id = detail.get("id")
            query = detail.get("query", "")
            decomp_strat = detail.get("decomposition_strategy", "none")
            retrieved_urls = detail.get("retrieved_urls", [])

            miss_info = {
                "id": case_id,
                "query": query,
                "strategy": decomp_strat,
                "retrieved": retrieved_urls[:3],  # Top 3 retrieved
                "answer_hit": detail.get("answer_hit", False),
            }

            # Categorize by decomposition strategy
            misses[f"strategy_{decomp_strat}"].append(miss_info)

            # Categorize by query type (heuristic based on query text)
            if any(word in query.lower() for word in ["what", "how", "which", "why"]):
                if any(word in query.lower() for word in ["difference", "vs", "versus", "compare"]):
                    misses["intent_comparison"].append(miss_info)
                else:
                    misses["intent_question"].append(miss_info)
            elif any(word in query.lower() for word in ["export", "create", "delete", "set", "enable", "disable"]):
                misses["intent_howto"].append(miss_info)
            else:
                misses["intent_other"].append(miss_info)

    return misses


def print_diagnosis(eval_data: dict, misses: Dict[str, List[dict]]):
    """Print detailed miss diagnosis."""
    total_cases = eval_data.get("cases", 0)
    total_misses = sum(
        1 for detail in eval_data.get("details", [])
        if not detail.get("recall@5", False)
    )
    recall_rate = eval_data.get("recall_at_5", 0)

    print(f"\n{'='*80}")
    print("EVALUATION MISS DIAGNOSIS")
    print(f"{'='*80}")
    print(f"Total cases: {total_cases}")
    print(f"Missed cases: {total_misses} ({total_misses/total_cases*100:.1f}%)")
    print(f"Recall@5: {recall_rate:.3f}")

    # Group by strategy
    print(f"\n{'By Decomposition Strategy':^80}")
    print("-" * 80)
    strategy_keys = [k for k in misses.keys() if k.startswith("strategy_")]
    if strategy_keys:
        print(f"{'Strategy':<20} {'Miss Count':<15} {'Pct of Total':<15}")
        print("-" * 80)
        for key in sorted(strategy_keys):
            strategy_name = key.replace("strategy_", "")
            count = len(misses[key])
            pct = count / total_misses * 100 if total_misses > 0 else 0
            print(f"{strategy_name:<20} {count:<15} {pct:<14.1f}%")
    else:
        print("No decomposition data available")

    # Group by intent
    print(f"\n{'By Query Intent':^80}")
    print("-" * 80)
    intent_keys = [k for k in misses.keys() if k.startswith("intent_")]
    if intent_keys:
        print(f"{'Intent':<20} {'Miss Count':<15} {'Pct of Total':<15}")
        print("-" * 80)
        for key in sorted(intent_keys):
            intent_name = key.replace("intent_", "").title()
            count = len(misses[key])
            pct = count / total_misses * 100 if total_misses > 0 else 0
            print(f"{intent_name:<20} {count:<15} {pct:<14.1f}%")
    else:
        print("No intent categorization available")

    # Detailed miss examples
    print(f"\n{'='*80}")
    print("SAMPLE MISSED CASES (First 5)")
    print(f"{'='*80}")

    miss_list = []
    for detail in eval_data.get("details", []):
        if not detail.get("recall@5", False):
            miss_list.append(detail)

    for i, miss in enumerate(miss_list[:5], 1):
        print(f"\n[{i}] {miss.get('id')}: {miss.get('query', 'Unknown query')}")
        print(f"    Decomposition: {miss.get('decomposition_strategy', 'none')}")
        print(f"    Answer match: {'✓' if miss.get('answer_hit') else '✗'}")
        print(f"    Top 3 retrieved:")
        for j, url in enumerate(miss.get("retrieved_urls", [])[:3], 1):
            print(f"      {j}. {url}")

    print(f"\n{'='*80}\n")


def extract_failure_patterns(eval_data: dict) -> Dict[str, Any]:
    """Extract and summarize failure patterns."""
    patterns = {
        "low_coverage_queries": [],  # Queries with few glossary matches
        "generic_title_misses": [],  # Misses where retrieved docs have generic titles
        "api_vocabulary_gaps": [],  # API-related queries that missed
        "multi_intent_failures": [],  # Multi-part queries that failed to decompose
    }

    for detail in eval_data.get("details", []):
        if not detail.get("recall@5", False):
            query = detail.get("query", "").lower()
            retrieved = detail.get("retrieved_urls", [])

            # API vocabulary gap
            if any(word in query for word in ["api", "integration", "webhook", "curl"]):
                patterns["api_vocabulary_gaps"].append({
                    "id": detail.get("id"),
                    "query": query,
                })

            # Multi-intent failure
            if any(word in query for word in ["and", "vs", "versus", "compare"]):
                if detail.get("decomposition_strategy") in ["none", "heuristic"]:
                    patterns["multi_intent_failures"].append({
                        "id": detail.get("id"),
                        "query": query,
                        "strategy": detail.get("decomposition_strategy"),
                    })

            # Generic title detection
            if retrieved:
                generic_titles = [
                    url for url in retrieved
                    if any(generic in url for generic in [
                        "whats-new", "troubleshooting", "help", "getting-started"
                    ])
                ]
                if len(generic_titles) >= 2:  # Mostly generic results
                    patterns["generic_title_misses"].append({
                        "id": detail.get("id"),
                        "query": query,
                    })

    return patterns


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 eval/diagnose_misses.py <eval_results.json>")
        print("Example: python3 eval/diagnose_misses.py logs/evals/baseline_latest.json")
        sys.exit(1)

    filepath = sys.argv[1]
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    eval_data = load_eval_results(filepath)
    misses = categorize_misses(eval_data)
    print_diagnosis(eval_data, misses)

    # Extract patterns
    patterns = extract_failure_patterns(eval_data)
    print(f"{'='*80}")
    print("KEY FAILURE PATTERNS")
    print(f"{'='*80}")
    print(f"\nAPI Vocabulary Gaps ({len(patterns['api_vocabulary_gaps'])} cases):")
    for case in patterns["api_vocabulary_gaps"][:3]:
        print(f"  - {case['query']}")

    print(f"\nMulti-Intent Failures ({len(patterns['multi_intent_failures'])} cases):")
    for case in patterns["multi_intent_failures"][:3]:
        print(f"  - {case['query']} (strategy: {case['strategy']})")

    print(f"\nGeneric Title Misses ({len(patterns['generic_title_misses'])} cases):")
    for case in patterns["generic_title_misses"][:3]:
        print(f"  - {case['query']}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
