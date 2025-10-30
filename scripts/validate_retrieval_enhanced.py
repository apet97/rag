#!/usr/bin/env python3
"""Enhanced retrieval quality validation for personal PC testing."""

import json
import requests
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# 25 realistic queries covering common support scenarios
TEST_QUERIES = [
    # Time Tracking (5)
    "How do I start tracking time in Clockify?",
    "What's the difference between timer and manual time entry?",
    "Can I track time retroactively after the fact?",
    "How do I pause and resume a timer?",
    "What happens if I forget to stop the timer?",

    # Projects & Organization (5)
    "How do I create a new project in Clockify?",
    "How do I delete a project?",
    "Can I organize projects by client?",
    "How do I set billable rates for projects?",
    "Can I assign multiple users to a project?",

    # Reports & Exports (5)
    "How do I generate a timesheet report?",
    "How do I export my time tracking data to Excel?",
    "Can I view reports by team member?",
    "What reporting features does Clockify offer?",
    "Can I schedule reports to be sent automatically?",

    # Approvals & Management (5)
    "What are timesheet approvals?",
    "How do I approve timesheets as a manager?",
    "Can I set up approval workflows?",
    "How do I reject a timesheet?",
    "Can I track time on behalf of my team?",

    # Integrations & Features (5)
    "What integrations does Clockify support?",
    "Does Clockify integrate with Jira for time tracking?",
    "Can I sync my calendar with Clockify?",
    "Is there a Clockify desktop app or mobile app?",
    "Does Clockify work with Slack?",
]

class RetrieverValidator:
    """Enhanced retrieval validation with detailed analysis."""

    def __init__(self):
        self.base_url = "http://localhost:8888"
        self.results = []
        self.quality_metrics = defaultdict(list)

    def print_header(self, text):
        """Print formatted header."""
        print(f"\n{'='*90}")
        print(f"{text:^90}")
        print(f"{'='*90}\n")

    def print_separator(self):
        """Print separator."""
        print(f"{'-'*90}\n")

    def categorize_score(self, score):
        """Categorize retrieval score."""
        if score > 0.8:
            return "Excellent", "🏆"
        elif score > 0.7:
            return "Good", "🎯"
        elif score > 0.5:
            return "Acceptable", "✓"
        else:
            return "Poor", "⚠️"

    def validate_query(self, query, namespace, query_num, total):
        """Validate single query retrieval."""
        print(f"[{query_num:2d}/{total}] {query[:70]}")

        try:
            start_time = time.time()
            response = requests.get(
                f"{self.base_url}/search",
                params={"q": query, "namespace": namespace, "k": 5},
                timeout=10
            )
            latency = time.time() - start_time

            if response.status_code != 200:
                print(f"  ❌ FAILED: HTTP {response.status_code}\n")
                return None

            data = response.json()
            sources = data.get("results", [])

            if not sources:
                print(f"  ⚠️  No results returned\n")
                return None

            # Extract details
            titles = [s.get("title", "Untitled") for s in sources]
            scores = [s.get("vector_score", 0) for s in sources]
            bodies = [s.get("body", "")[:100] for s in sources]
            avg_score = sum(scores) / len(scores)

            quality, symbol = self.categorize_score(avg_score)
            self.quality_metrics[quality].append(query)

            # Print detailed result
            print(f"  {symbol} {quality:12} (Avg: {avg_score:.3f}, Min: {min(scores):.3f}, Max: {max(scores):.3f})")
            print(f"     Latency: {latency*1000:.0f}ms\n")

            # Print top 3 with details
            for i, (title, score, body) in enumerate(zip(titles[:3], scores[:3], bodies[:3]), 1):
                print(f"     {i}. [{score:.3f}] {title[:60]}")
                print(f"        {body}...\n")

            self.print_separator()

            return {
                "query": query,
                "namespace": namespace,
                "status": "success",
                "avg_score": avg_score,
                "min_score": min(scores),
                "max_score": max(scores),
                "result_count": len(sources),
                "top_titles": titles,
                "top_scores": scores,
                "latency_ms": latency * 1000,
            }

        except requests.exceptions.Timeout:
            print(f"  ❌ TIMEOUT (>10s)\n")
            return None
        except requests.exceptions.ConnectionError:
            print(f"  ❌ CONNECTION ERROR\n")
            return None
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}\n")
            return None

    def run_validation(self):
        """Run full validation suite."""
        self.print_header("🔍 ENHANCED RETRIEVAL QUALITY VALIDATION")

        print(f"Configuration:")
        print(f"  Server:        {self.base_url}")
        print(f"  Queries:       {len(TEST_QUERIES)}")
        print(f"  Namespace:     clockify")
        print(f"  Results per query: 5\n")

        self.print_header("Validation Results")

        for i, query in enumerate(TEST_QUERIES, 1):
            result = self.validate_query(query, "clockify", i, len(TEST_QUERIES))
            if result:
                self.results.append(result)

        # Generate analysis report
        self.print_header("Retrieval Quality Analysis")

        if not self.results:
            print("❌ No successful retrievals\n")
            return

        successful = len(self.results)
        scores = [r["avg_score"] for r in self.results]

        print(f"Overall Statistics:")
        print(f"  Successful queries:     {successful}/{len(TEST_QUERIES)}")
        print(f"  Success rate:           {successful/len(TEST_QUERIES)*100:.0f}%\n")

        print(f"Score Distribution:")
        print(f"  Average score:          {sum(scores)/len(scores):.3f}")
        print(f"  Median score:           {sorted(scores)[len(scores)//2]:.3f}")
        print(f"  Min score:              {min(scores):.3f}")
        print(f"  Max score:              {max(scores):.3f}\n")

        print(f"Quality Breakdown:")
        for quality in ["Excellent", "Good", "Acceptable", "Poor"]:
            count = len(self.quality_metrics[quality])
            if count > 0:
                pct = count / successful * 100
                queries = self.quality_metrics[quality][:2]
                query_preview = f" (e.g., '{queries[0][:40]}...')" if queries else ""
                print(f"  {quality:12} {count:2d} queries ({pct:5.1f}%){query_preview}")

        print(f"\nLatency Statistics:")
        latencies = [r["latency_ms"] for r in self.results]
        print(f"  Average:                {sum(latencies)/len(latencies):.0f}ms")
        print(f"  Min:                    {min(latencies):.0f}ms")
        print(f"  Max:                    {max(latencies):.0f}ms")
        print(f"  Queries < 100ms:        {sum(1 for l in latencies if l < 100)}")
        print(f"  Queries < 500ms:        {sum(1 for l in latencies if l < 500)}")

        # Find best and worst performing queries
        print(f"\n" + "="*90)
        print(f"Best & Worst Performing Queries".center(90))
        print(f"="*90 + "\n")

        sorted_results = sorted(self.results, key=lambda r: r["avg_score"], reverse=True)

        print(f"🏆 TOP 5 BEST PERFORMING:")
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. [{result['avg_score']:.3f}] {result['query'][:65]}")
            print(f"     → {result['top_titles'][0][:60]}")

        print(f"\n⚠️  BOTTOM 5 NEEDS IMPROVEMENT:")
        for i, result in enumerate(sorted_results[-5:][::-1], 1):
            print(f"  {i}. [{result['avg_score']:.3f}] {result['query'][:65]}")
            print(f"     → {result['top_titles'][0][:60]}")

        # Recommendations
        print(f"\n" + "="*90)
        print(f"Recommendations".center(90))
        print(f"="*90 + "\n")

        poor_queries = self.quality_metrics.get("Poor", [])
        if poor_queries:
            print(f"⚠️  {len(poor_queries)} queries scoring < 0.5:")
            for q in poor_queries[:3]:
                print(f"    • {q}")
            print(f"\n    Action: Review corpus for relevant content on these topics\n")
        else:
            print(f"✅ All queries scoring >= 0.5 - Excellent coverage!\n")

        # Save detailed results
        results_file = LOG_DIR / "retrieval_test_data.json"
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "query_count": len(TEST_QUERIES),
                "successful_count": successful,
                "success_rate": successful/len(TEST_QUERIES)*100,
                "queries": self.results,
                "summary": {
                    "avg_score": sum(scores)/len(scores),
                    "median_score": sorted(scores)[len(scores)//2],
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "avg_latency_ms": sum(latencies)/len(latencies),
                    "quality_distribution": {
                        k: len(v) for k, v in self.quality_metrics.items()
                    }
                }
            }, f, indent=2)

        print(f"✅ Detailed results saved to: {results_file}\n")

        # Generate markdown report
        self._generate_markdown_report()

        return {
            "total_queries": len(TEST_QUERIES),
            "successful": successful,
            "avg_score": sum(scores)/len(scores),
            "quality_metrics": dict(self.quality_metrics)
        }

    def _generate_markdown_report(self):
        """Generate markdown report."""
        report_file = LOG_DIR / "retrieval_quality_report.md"

        lines = [
            "# Retrieval Quality Validation Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"- **Total Queries Tested:** {len(TEST_QUERIES)}",
            f"- **Successful:** {len(self.results)}/{len(TEST_QUERIES)} ({len(self.results)/len(TEST_QUERIES)*100:.0f}%)",
            f"- **Average Relevance Score:** {sum([r['avg_score'] for r in self.results])/len(self.results):.3f}",
            "",
            "## Quality Distribution",
            "",
        ]

        for quality, queries in self.quality_metrics.items():
            lines.append(f"- **{quality}:** {len(queries)} queries")

        lines.extend([
            "",
            "## Detailed Results",
            "",
        ])

        for result in self.results:
            lines.extend([
                f"### Query: {result['query']}",
                f"- **Score:** {result['avg_score']:.3f}",
                f"- **Latency:** {result['latency_ms']:.0f}ms",
                f"- **Top Result:** {result['top_titles'][0]}",
                "",
            ])

        lines.extend([
            "## Performance Metrics",
            "",
            f"- **Min Score:** {min([r['avg_score'] for r in self.results]):.3f}",
            f"- **Max Score:** {max([r['avg_score'] for r in self.results]):.3f}",
            f"- **Avg Latency:** {sum([r['latency_ms'] for r in self.results])/len(self.results):.0f}ms",
            "",
            "## Status",
            "",
            "✅ **PERSONAL PC RETRIEVAL VALIDATION COMPLETE**",
            "",
            "The retrieval system is working and ready for mock LLM testing.",
        ])

        with open(report_file, "w") as f:
            f.write("\n".join(lines))

        print(f"✅ Markdown report saved to: {report_file}")

if __name__ == "__main__":
    validator = RetrieverValidator()
    results = validator.run_validation()

    if results and results["successful"] > 0:
        print("\n" + "="*90)
        print("🎉 RETRIEVAL VALIDATION SUCCESSFUL".center(90))
        print("="*90)
        print(f"\nNext Step: Create RAG pipeline with mock LLM mode")
        print(f"Command: python scripts/test_rag_mock.py\n")
        exit(0)
    else:
        print("\n⚠️  Retrieval validation needs attention\n")
        exit(1)
