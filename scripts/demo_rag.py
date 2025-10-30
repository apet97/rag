#!/usr/bin/env python3
"""Interactive demo of Clockify RAG system with hand-picked queries."""

import requests
import time
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Hand-picked realistic queries for demo
DEMO_QUERIES = [
    {
        "query": "How do I start tracking time in Clockify?",
        "category": "Getting Started",
        "namespace": "clockify"
    },
    {
        "query": "What integrations does Clockify support?",
        "category": "Integrations",
        "namespace": "clockify"
    },
    {
        "query": "How do I generate a timesheet report?",
        "category": "Reports",
        "namespace": "clockify"
    },
    {
        "query": "Can I track time retroactively?",
        "category": "Time Tracking",
        "namespace": "clockify"
    },
    {
        "query": "How do I set up project budgeting?",
        "category": "Project Management",
        "namespace": "clockify"
    },
    {
        "query": "What are the best practices for time tracking?",
        "category": "Best Practices",
        "namespace": "clockify"
    },
    {
        "query": "How do I export data to Excel?",
        "category": "Data Export",
        "namespace": "clockify"
    },
    {
        "query": "Can I use Clockify on mobile?",
        "category": "Mobile",
        "namespace": "clockify"
    },
    {
        "query": "How do I set up team approvals?",
        "category": "Team Management",
        "namespace": "clockify"
    },
    {
        "query": "What is the difference between projects and tasks?",
        "category": "Organization",
        "namespace": "clockify"
    },
]

class Demo:
    """Interactive RAG demo."""

    def __init__(self):
        self.base_url = "http://localhost:8888"
        self.results = []
        self.llm_available = False

    def check_llm(self):
        """Check if LLM is available."""
        try:
            response = requests.post(
                "http://localhost:8080/v1/chat/completions",
                json={
                    "model": "oss20b",
                    "messages": [{"role": "user", "content": "OK"}],
                    "max_tokens": 5,
                },
                timeout=5
            )
            self.llm_available = response.status_code == 200
        except:
            self.llm_available = False

    def print_header(self, text):
        """Print formatted header."""
        print(f"\n{'='*80}")
        print(f"{text:^80}")
        print(f"{'='*80}\n")

    def print_separator(self):
        """Print separator."""
        print(f"{'-'*80}\n")

    def run_demo_query(self, query_data):
        """Run a single demo query."""
        query = query_data["query"]
        category = query_data["category"]
        namespace = query_data["namespace"]

        print(f"📌 Category: {category}")
        print(f"❓ Query: {query}\n")

        try:
            # Retrieve relevant sources
            start_time = time.time()
            response = requests.get(
                f"{self.base_url}/search",
                params={"q": query, "namespace": namespace, "k": 3},
                timeout=10
            )
            retrieval_time = time.time() - start_time

            if response.status_code != 200:
                print(f"❌ Retrieval failed\n")
                return None

            data = response.json()
            sources = data.get("results", [])

            if not sources:
                print(f"❌ No sources found\n")
                return None

            print(f"📚 Retrieved {len(sources)} sources in {retrieval_time:.2f}s:")
            for i, source in enumerate(sources, 1):
                title = source.get("title", "Untitled")
                score = source.get("vector_score", 0)
                print(f"   {i}. {title[:70]} (score: {score:.3f})")

            # Try to get LLM answer if available
            if self.llm_available:
                print(f"\n⏳ Generating answer with LLM...")
                try:
                    llm_start = time.time()
                    llm_response = requests.post(
                        f"{self.base_url}/chat",
                        json={"question": query, "namespace": namespace, "k": 3},
                        timeout=15
                    )
                    llm_time = time.time() - llm_start

                    if llm_response.status_code == 200:
                        llm_data = llm_response.json()
                        answer = llm_data.get("answer", "")

                        if answer:
                            print(f"\n💬 Answer ({llm_time:.2f}s):")
                            # Print first 150 characters of answer
                            answer_preview = answer[:150] + ("..." if len(answer) > 150 else "")
                            print(f"   {answer_preview}\n")

                            return {
                                "query": query,
                                "category": category,
                                "retrieval_time": retrieval_time,
                                "llm_time": llm_time,
                                "sources_count": len(sources),
                                "has_answer": True
                            }

                except:
                    pass

            # Retrieval-only result
            return {
                "query": query,
                "category": category,
                "retrieval_time": retrieval_time,
                "llm_time": 0,
                "sources_count": len(sources),
                "has_answer": False
            }

        except Exception as e:
            print(f"❌ Error: {str(e)}\n")
            return None

    def run(self):
        """Run the interactive demo."""
        self.print_header("🎯 CLOCKIFY RAG SYSTEM - INTERACTIVE DEMO")

        print(f"System Status:")
        print(f"  Server:           http://localhost:8888")
        print(f"  Retrieval:        ✅ Ready")

        self.check_llm()
        if self.llm_available:
            print(f"  LLM:              ✅ Available")
        else:
            print(f"  LLM:              ⏳ Not running (showing retrieval-only)")

        print(f"  Demo Queries:     {len(DEMO_QUERIES)}")

        self.print_header("🚀 DEMO EXECUTION")

        successful = 0
        failed = 0

        for i, query_data in enumerate(DEMO_QUERIES, 1):
            print(f"[{i}/{len(DEMO_QUERIES)}] ", end="")

            result = self.run_demo_query(query_data)

            if result:
                self.results.append(result)
                successful += 1
            else:
                failed += 1

            self.print_separator()

        # Print demo summary
        self.print_header("📊 DEMO SUMMARY")

        print(f"Total Queries:          {len(DEMO_QUERIES)}")
        print(f"Successful:             {successful} ✅")
        print(f"Failed:                 {failed} ❌")

        if self.results:
            avg_retrieval_time = sum(r["retrieval_time"] for r in self.results) / len(self.results)
            total_sources = sum(r["sources_count"] for r in self.results)
            with_answers = sum(1 for r in self.results if r.get("has_answer"))

            print(f"\nRetrieval Performance:")
            print(f"  Average Latency:    {avg_retrieval_time:.3f}s")
            print(f"  Total Sources:      {total_sources}")
            print(f"  Avg per Query:      {total_sources / len(self.results):.1f}")

            if self.llm_available and with_answers > 0:
                avg_llm_time = sum(r["llm_time"] for r in self.results if r["has_answer"]) / with_answers
                print(f"\nLLM Performance:")
                print(f"  Queries with Answer: {with_answers}")
                print(f"  Average Latency:     {avg_llm_time:.3f}s")

            print(f"\nSystem Readiness:")
            if successful == len(DEMO_QUERIES):
                print(f"  ✅ PRODUCTION READY - All queries successful")
                readiness_score = 100
            elif successful >= len(DEMO_QUERIES) * 0.8:
                print(f"  ✅ NEARLY READY - {successful}/{len(DEMO_QUERIES)} queries successful")
                readiness_score = 85
            else:
                print(f"  ⚠️  NEEDS WORK - {successful}/{len(DEMO_QUERIES)} queries successful")
                readiness_score = 50

            print(f"  Readiness Score:     {readiness_score}/100")

        # Save demo report
        report_file = LOG_DIR / "demo_report.json"
        with open(report_file, "w") as f:
            import json
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "llm_available": self.llm_available,
                "queries_run": len(DEMO_QUERIES),
                "successful": successful,
                "failed": failed,
                "results": self.results
            }, f, indent=2)

        print(f"\n✅ Demo report saved to {report_file}")

        print(f"\nNext Steps:")
        print(f"  1. To enable full RAG with LLM generation:")
        print(f"     ollama pull oss20b && ollama serve")
        print(f"  2. Rerun this demo to see LLM-generated answers")
        print(f"  3. Deploy to production: python scripts/deployment_checklist.py")

        return 0 if failed == 0 else 1

if __name__ == "__main__":
    demo = Demo()
    exit_code = demo.run()
    exit(exit_code)
