#!/usr/bin/env python3
"""Master test suite orchestration for Clockify RAG system."""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print colored header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_test(name, status, details=""):
    """Print test result with color."""
    if status == "PASSED":
        symbol = f"{Colors.OKGREEN}✅{Colors.ENDC}"
        status_text = f"{Colors.OKGREEN}{status}{Colors.ENDC}"
    elif status == "FAILED":
        symbol = f"{Colors.FAIL}❌{Colors.ENDC}"
        status_text = f"{Colors.FAIL}{status}{Colors.ENDC}"
    elif status == "PARTIAL":
        symbol = f"{Colors.WARNING}⚠️{Colors.ENDC}"
        status_text = f"{Colors.WARNING}{status}{Colors.ENDC}"
    elif status == "RUNNING":
        symbol = f"{Colors.OKBLUE}⏳{Colors.ENDC}"
        status_text = f"{Colors.OKBLUE}{status}{Colors.ENDC}"
    else:
        symbol = "•"
        status_text = status

    print(f"{symbol} {name:<50} {status_text:<15} {details}")

def run_test(script_name, description):
    """Run a single test script and return results."""
    print_test(description, "RUNNING")

    script_path = Path("scripts") / f"{script_name}.py"

    if not script_path.exists():
        print_test(description, "FAILED", "Script not found")
        return {
            "name": script_name,
            "status": "failed",
            "error": "Script not found"
        }

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Check if test passed
        if result.returncode == 0:
            status = "PASSED"
            result_status = "passed"
        else:
            status = "FAILED"
            result_status = "failed"

        # Try to load results JSON if it exists
        result_file = LOG_DIR / f"{script_name.replace('test_', '').replace('validate_', '')}_test_results.json"
        if script_name == "validate_retrieval":
            result_file = LOG_DIR / "retrieval_test_results.json"
        elif script_name == "test_llm_connection":
            result_file = LOG_DIR / "llm_connection_test.json"
        elif script_name == "test_rag_pipeline":
            result_file = LOG_DIR / "rag_pipeline_test_results.json"
        elif script_name == "test_api":
            result_file = LOG_DIR / "api_test_results.json"

        details = ""
        result_data = {}

        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    result_data = json.load(f)
                    if "summary" in result_data:
                        summary = result_data["summary"]
                        if "success_rate" in summary:
                            details = f"({summary['success_rate']:.0f}% success)"
                        elif "overall_pass_rate" in summary:
                            details = f"({summary['overall_pass_rate']})"
            except:
                pass

        print_test(description, status, details)

        return {
            "name": script_name,
            "status": result_status,
            "return_code": result.returncode,
            "result_data": result_data,
            "stdout_lines": len(result.stdout.split('\n')),
            "stderr_lines": len(result.stderr.split('\n')),
        }

    except subprocess.TimeoutExpired:
        print_test(description, "FAILED", "Timeout (>120s)")
        return {
            "name": script_name,
            "status": "failed",
            "error": "Timeout"
        }
    except Exception as e:
        print_test(description, "FAILED", f"Error: {str(e)[:30]}")
        return {
            "name": script_name,
            "status": "failed",
            "error": str(e)
        }

def main():
    """Run all tests in sequence."""
    print_header("CLOCKIFY RAG - COMPREHENSIVE TEST SUITE")

    print(f"{Colors.BOLD}Starting test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")

    # Define test suite
    tests = [
        ("validate_retrieval", "Step 1: Retrieval Validation (20 queries)"),
        ("test_llm_connection", "Step 2: LLM Connection Test"),
        ("test_rag_pipeline", "Step 3: RAG Pipeline Test (15 queries)"),
        ("test_api", "Step 4: API Endpoint Tests"),
    ]

    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "summary": {}
    }

    print(f"{Colors.OKCYAN}Test Execution:{Colors.ENDC}\n")

    test_results = []
    for script_name, description in tests:
        start_time = time.time()
        result = run_test(script_name, description)
        result["duration_s"] = time.time() - start_time
        test_results.append(result)
        results["tests"].append(result)
        time.sleep(0.5)  # Brief pause between tests

    # Calculate summary
    passed = sum(1 for r in test_results if r.get("status") == "passed")
    failed = sum(1 for r in test_results if r.get("status") == "failed")
    total = len(test_results)

    # Print summary
    print_header("TEST SUITE SUMMARY")

    print(f"{Colors.BOLD}Results:{Colors.ENDC}\n")
    print(f"  Total Tests:     {total}")
    print(f"  {Colors.OKGREEN}Passed:{Colors.ENDC}     {passed}")
    print(f"  {Colors.FAIL}Failed:{Colors.ENDC}     {failed}")

    if total > 0:
        pass_rate = (passed / total) * 100
        if pass_rate >= 90:
            color = Colors.OKGREEN
            symbol = "✅"
        elif pass_rate >= 70:
            color = Colors.WARNING
            symbol = "⚠️"
        else:
            color = Colors.FAIL
            symbol = "❌"

        print(f"  Pass Rate:       {color}{symbol} {pass_rate:.0f}%{Colors.ENDC}")

    print(f"\n{Colors.BOLD}Test Details:{Colors.ENDC}\n")

    for result in test_results:
        name = result.get("name", "Unknown")
        status = result.get("status", "unknown")
        duration = result.get("duration_s", 0)

        if status == "passed":
            status_symbol = f"{Colors.OKGREEN}✅ PASSED{Colors.ENDC}"
        elif status == "failed":
            status_symbol = f"{Colors.FAIL}❌ FAILED{Colors.ENDC}"
        else:
            status_symbol = "❓ UNKNOWN"

        print(f"  {name:30} {status_symbol:30} ({duration:.1f}s)")

    # Print recommendations
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}\n")

    if failed > 0:
        print(f"  {Colors.WARNING}⚠️  {failed} test(s) failed.{Colors.ENDC}")
        print(f"     Review the detailed output above and fix issues.")
    else:
        print(f"  {Colors.OKGREEN}✅ All core tests passed!{Colors.ENDC}")

    print(f"\n  To complete validation:")
    print(f"    1. {Colors.OKCYAN}Start LLM:{Colors.ENDC} ollama pull oss20b && ollama serve")
    print(f"    2. {Colors.OKCYAN}Rerun tests:{Colors.ENDC} python scripts/run_all_tests.py")
    print(f"    3. {Colors.OKCYAN}Deploy:{Colors.ENDC} python scripts/deployment_checklist.py")

    # Save results
    results["summary"] = {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total * 100) if total > 0 else 0,
    }

    results_file = LOG_DIR / "test_suite_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {results_file}\n")

    # Exit with appropriate code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
