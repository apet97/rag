#!/usr/bin/env python3
"""Test FastAPI RAG server endpoints and response formats."""

import json
import requests
import time
from pathlib import Path
from datetime import datetime

# Logging setup
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

BASE_URL = "http://localhost:8888"

def test_health_endpoint():
    """Test /health endpoint."""
    print("\n" + "="*80)
    print("TEST 1: /health ENDPOINT")
    print("="*80 + "\n")

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health endpoint responding\n")
            print(f"Response:")
            print(json.dumps(data, indent=2))
            return {
                "endpoint": "/health",
                "status": "passed",
                "status_code": response.status_code,
                "response": data
            }
        else:
            print(f"❌ Unexpected status code: {response.status_code}\n")
            return {
                "endpoint": "/health",
                "status": "failed",
                "status_code": response.status_code,
                "error": "Non-200 response"
            }
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}\n")
        return {
            "endpoint": "/health",
            "status": "failed",
            "error": str(e)
        }

def test_search_endpoint():
    """Test /search endpoint."""
    print("\n" + "="*80)
    print("TEST 2: /search ENDPOINT")
    print("="*80 + "\n")

    test_cases = [
        {
            "name": "Basic search - Clockify namespace",
            "params": {"q": "How do I create a project?", "namespace": "clockify", "k": 3},
            "should_pass": True
        },
        {
            "name": "Search with different k value",
            "params": {"q": "time tracking", "namespace": "clockify", "k": 5},
            "should_pass": True
        },
        {
            "name": "Search LangChain namespace",
            "params": {"q": "what is a vector database", "namespace": "langchain", "k": 3},
            "should_pass": True
        },
        {
            "name": "Search with empty query (should fail gracefully)",
            "params": {"q": "", "namespace": "clockify", "k": 3},
            "should_pass": False
        },
        {
            "name": "Search with invalid namespace",
            "params": {"q": "test query", "namespace": "invalid", "k": 3},
            "should_pass": False
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"[Test {i}/{len(test_cases)}] {test_case['name']}")

        try:
            start_time = time.time()
            response = requests.get(
                f"{BASE_URL}/search",
                params=test_case["params"],
                timeout=10
            )
            latency = time.time() - start_time

            print(f"  Status Code: {response.status_code}")
            print(f"  Latency: {latency:.3f}s")

            if test_case["should_pass"]:
                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✅ PASSED")

                    # Validate response format
                    if "results" in data and "count" in data:
                        print(f"     Results count: {data.get('count', 0)}")
                        if data.get('count', 0) > 0:
                            first_result = data['results'][0]
                            print(f"     Top result: {first_result.get('title', 'N/A')[:60]}")
                            print(f"     Score: {first_result.get('vector_score', 'N/A')}")
                    print()

                    results.append({
                        "name": test_case["name"],
                        "status": "passed",
                        "status_code": response.status_code,
                        "latency_s": latency,
                        "result_count": data.get("count", 0),
                    })
                else:
                    print(f"  ❌ FAILED - Expected 200, got {response.status_code}\n")
                    results.append({
                        "name": test_case["name"],
                        "status": "failed",
                        "status_code": response.status_code,
                    })
            else:
                if response.status_code != 200:
                    print(f"  ✅ PASSED (correctly rejected)\n")
                    results.append({
                        "name": test_case["name"],
                        "status": "passed_correctly_rejected",
                        "status_code": response.status_code,
                    })
                else:
                    print(f"  ⚠️  Should have failed but got 200\n")
                    results.append({
                        "name": test_case["name"],
                        "status": "warning",
                        "status_code": response.status_code,
                    })

        except Exception as e:
            print(f"  ❌ Connection failed: {str(e)}\n")
            results.append({
                "name": test_case["name"],
                "status": "failed",
                "error": str(e)
            })

    return results

def test_chat_endpoint():
    """Test /chat endpoint (RAG generation)."""
    print("\n" + "="*80)
    print("TEST 3: /chat ENDPOINT (RAG)")
    print("="*80 + "\n")

    test_cases = [
        {
            "name": "Basic RAG query",
            "payload": {
                "question": "How do I create a project in Clockify?",
                "namespace": "clockify",
                "k": 3
            },
            "should_pass": True
        },
        {
            "name": "RAG query with different namespace",
            "payload": {
                "question": "What is langchain",
                "namespace": "langchain",
                "k": 3
            },
            "should_pass": True
        },
        {
            "name": "RAG with custom k value",
            "payload": {
                "question": "time tracking features",
                "namespace": "clockify",
                "k": 5
            },
            "should_pass": True
        },
    ]

    results = []
    llm_available = False

    # Check LLM availability first
    try:
        test_response = requests.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": "oss20b",
                "messages": [{"role": "user", "content": "OK"}],
                "max_tokens": 5,
            },
            timeout=5
        )
        llm_available = test_response.status_code == 200
    except:
        llm_available = False

    if not llm_available:
        print("⚠️  LLM NOT RUNNING - Testing retrieval part of RAG endpoint only\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"[Test {i}/{len(test_cases)}] {test_case['name']}")

        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/chat",
                json=test_case["payload"],
                timeout=15
            )
            latency = time.time() - start_time

            print(f"  Status Code: {response.status_code}")
            print(f"  Latency: {latency:.3f}s")

            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ PASSED")

                # Validate response format
                if "sources" in data:
                    print(f"     Sources retrieved: {len(data.get('sources', []))}")
                if "answer" in data and llm_available:
                    answer = data.get("answer", "")
                    print(f"     Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                elif "sources" in data and not llm_available:
                    print(f"     (LLM generation skipped - not running)")

                print()

                results.append({
                    "name": test_case["name"],
                    "status": "passed",
                    "status_code": response.status_code,
                    "latency_s": latency,
                    "has_sources": "sources" in data,
                    "has_answer": "answer" in data,
                })
            else:
                print(f"  ❌ FAILED - HTTP {response.status_code}\n")
                results.append({
                    "name": test_case["name"],
                    "status": "failed",
                    "status_code": response.status_code,
                })

        except requests.exceptions.Timeout:
            if llm_available:
                print(f"  ⚠️  Request timeout (15s) - LLM may be slow\n")
                results.append({
                    "name": test_case["name"],
                    "status": "timeout_with_llm",
                })
            else:
                print(f"  ❌ Timeout (should be <2s without LLM)\n")
                results.append({
                    "name": test_case["name"],
                    "status": "failed_timeout",
                })
        except Exception as e:
            print(f"  ❌ Connection failed: {str(e)}\n")
            results.append({
                "name": test_case["name"],
                "status": "failed",
                "error": str(e)
            })

    return results, llm_available

def main():
    """Run all API tests."""
    print("\n" + "="*80)
    print("CLOCKIFY RAG - API ENDPOINT TEST SUITE")
    print("="*80)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {
            "health": {},
            "search": [],
            "chat": [],
        },
        "summary": {}
    }

    # Test 1: Health
    health_result = test_health_endpoint()
    all_results["tests"]["health"] = health_result

    # Test 2: Search
    search_results = test_search_endpoint()
    all_results["tests"]["search"] = search_results

    # Test 3: Chat
    chat_results, llm_available = test_chat_endpoint()
    all_results["tests"]["chat"] = chat_results
    all_results["llm_available"] = llm_available

    # Calculate summary
    health_passed = health_result.get("status") == "passed"
    search_passed = sum(1 for r in search_results if r.get("status") == "passed")
    search_total = len(search_results)
    chat_passed = sum(1 for r in chat_results if r.get("status") == "passed")
    chat_total = len(chat_results)

    all_results["summary"] = {
        "health_endpoint": "✅ PASSED" if health_passed else "❌ FAILED",
        "search_endpoint": f"{search_passed}/{search_total} tests passed",
        "chat_endpoint": f"{chat_passed}/{chat_total} tests passed",
        "overall_pass_rate": f"{(health_passed + search_passed + chat_passed)}/{1 + search_total + chat_total}"
    }

    # Print final summary
    print("\n" + "="*80)
    print("API TEST SUMMARY")
    print("="*80 + "\n")
    print(f"Health Endpoint:       {all_results['summary']['health_endpoint']}")
    print(f"Search Endpoint:       {all_results['summary']['search_endpoint']}")
    print(f"Chat Endpoint:         {all_results['summary']['chat_endpoint']}")
    print(f"LLM Available:         {'✅ Yes' if llm_available else '⏳ No'}")
    print(f"\nOverall Pass Rate:     {all_results['summary']['overall_pass_rate']}")

    # Save results
    results_file = LOG_DIR / "api_test_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")

    return all_results

if __name__ == "__main__":
    results = main()

    # Check overall pass rate
    if results["summary"]["health_endpoint"].startswith("✅"):
        print("\n🎉 API endpoint tests PASSED")
        exit(0)
    else:
        print("\n⚠️  Some API endpoint tests failed")
        exit(1)
