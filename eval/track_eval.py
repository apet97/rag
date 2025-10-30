#!/usr/bin/env python3
"""Evaluation tracking and versioning utility.

Runs evaluations and saves results with automatic versioning to logs/evals/.
Provides baseline + with-decomposition comparisons.

Usage:
    python3 eval/track_eval.py --label "baseline" --decomposition-off
    python3 eval/track_eval.py --label "with_decomposition"
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import argparse


def run_eval(args_list, label: str) -> dict:
    """Run evaluation script and capture JSON output."""
    cmd = ["python3", "eval/run_eval.py", "--json"] + args_list

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running eval: {result.stderr}")
        return None

    try:
        data = json.loads(result.stdout)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output: {e}")
        print(f"stdout: {result.stdout[:500]}")
        return None


def save_results(eval_data: dict, label: str, output_dir: Path = None):
    """Save evaluation results with timestamp versioning."""
    if output_dir is None:
        output_dir = Path("logs/evals")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp and version
    now = datetime.now().isoformat()[:19]  # YYYY-MM-DDTHH:MM:SS
    filename = f"{label}_{now.replace(':', '-')}.json"
    filepath = output_dir / filename

    # Save full JSON
    with open(filepath, "w") as f:
        json.dump(eval_data, f, indent=2)

    print(f"✓ Saved to {filepath}")

    # Also save to a latest.json symlink for each label
    latest_link = output_dir / f"{label}_latest.json"
    try:
        latest_link.unlink()  # Remove old symlink if exists
    except FileNotFoundError:
        pass

    try:
        latest_link.symlink_to(filepath.name)
        print(f"✓ Updated {latest_link.name}")
    except Exception as e:
        print(f"Note: Could not create symlink: {e}")

    return filepath


def print_summary(eval_data: dict, label: str):
    """Pretty print evaluation summary."""
    if not eval_data:
        return

    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY: {label}")
    print(f"{'='*80}")
    print(f"Cases: {eval_data.get('cases', 0)}")
    print(f"Recall@5: {eval_data.get('recall_at_5', 0):.3f}")
    print(f"MRR@5: {eval_data.get('mrr_at_5', 0):.3f}")
    print(f"Answer accuracy: {eval_data.get('answer_accuracy', 0):.3f}")

    latency = eval_data.get('retrieval_latency_ms', {})
    print(f"Retrieval latency p50/p95 (ms): {latency.get('p50', 0)} / {latency.get('p95', 0)}")

    full_latency = eval_data.get('full_latency_ms', {})
    print(f"Full pipeline latency p50/p95 (ms): {full_latency.get('p50', 0)} / {full_latency.get('p95', 0)}")
    print(f"{'='*80}\n")


def compare_results(baseline_data: dict, comparison_data: dict):
    """Compare two evaluation runs."""
    print(f"\n{'='*80}")
    print("A/B COMPARISON: Baseline vs With Decomposition")
    print(f"{'='*80}")

    baseline_recall = baseline_data.get('recall_at_5', 0)
    comparison_recall = comparison_data.get('recall_at_5', 0)
    recall_delta = comparison_recall - baseline_recall

    baseline_accuracy = baseline_data.get('answer_accuracy', 0)
    comparison_accuracy = comparison_data.get('answer_accuracy', 0)
    accuracy_delta = comparison_accuracy - baseline_accuracy

    baseline_latency = baseline_data.get('retrieval_latency_ms', {}).get('p50', 0)
    comparison_latency = comparison_data.get('retrieval_latency_ms', {}).get('p50', 0)
    latency_delta_pct = ((comparison_latency - baseline_latency) / baseline_latency * 100) if baseline_latency > 0 else 0

    print(f"Recall@5:")
    print(f"  Baseline: {baseline_recall:.3f}")
    print(f"  Comparison: {comparison_recall:.3f}")
    print(f"  Delta: {recall_delta:+.3f} ({recall_delta/baseline_recall*100:+.1f}%)")

    print(f"\nAnswer Accuracy:")
    print(f"  Baseline: {baseline_accuracy:.3f}")
    print(f"  Comparison: {comparison_accuracy:.3f}")
    print(f"  Delta: {accuracy_delta:+.3f} ({accuracy_delta/baseline_accuracy*100:+.1f}%)" if baseline_accuracy > 0 else f"  Delta: {accuracy_delta:+.3f}")

    print(f"\nRetrieval Latency p50 (ms):")
    print(f"  Baseline: {baseline_latency}")
    print(f"  Comparison: {comparison_latency}")
    print(f"  Delta: {latency_delta_pct:+.1f}%")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run and track RAG evaluations with automatic versioning"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline evaluation (decomposition disabled)"
    )
    parser.add_argument(
        "--with-decomposition",
        action="store_true",
        help="Run evaluation with decomposition enabled"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both baseline and with-decomposition, then compare"
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Custom label for results (e.g., 'session5c', 'post_embedding_fix')"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/evals"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k results to evaluate"
    )

    args = parser.parse_args()

    # Determine what to run
    if args.both:
        baseline_label = args.label + "_baseline" if args.label else "baseline"
        comparison_label = args.label + "_with_decomp" if args.label else "with_decomposition"

        # Run baseline
        print("\n" + "="*80)
        print("RUNNING BASELINE EVALUATION (Decomposition Disabled)")
        print("="*80)
        baseline_data = run_eval(
            ["--decomposition-off", "--k", str(args.k), "--log-decomposition"],
            baseline_label
        )
        if baseline_data:
            save_results(baseline_data, baseline_label, args.output_dir)
            print_summary(baseline_data, baseline_label)

        # Run with decomposition
        print("\n" + "="*80)
        print("RUNNING EVALUATION WITH DECOMPOSITION")
        print("="*80)
        comparison_data = run_eval(
            ["--k", str(args.k), "--log-decomposition"],
            comparison_label
        )
        if comparison_data:
            save_results(comparison_data, comparison_label, args.output_dir)
            print_summary(comparison_data, comparison_label)

        # Compare
        if baseline_data and comparison_data:
            compare_results(baseline_data, comparison_data)

    elif args.baseline:
        label = args.label or "baseline"
        eval_data = run_eval(["--decomposition-off", "--k", str(args.k)], label)
        if eval_data:
            save_results(eval_data, label, args.output_dir)
            print_summary(eval_data, label)

    elif args.with_decomposition:
        label = args.label or "with_decomposition"
        eval_data = run_eval(["--k", str(args.k)], label)
        if eval_data:
            save_results(eval_data, label, args.output_dir)
            print_summary(eval_data, label)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
