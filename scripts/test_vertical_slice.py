"""
Verification gate for Phase 2: runs the full vertical slice against a
real PR and confirms it catches both planted vulnerabilities (hardcoded
API key + SQL injection via f-string), while staying under the token
budget the plan specifies (<15K tokens for 120b).
"""

import sys

from agents.graph import run_review
from core.gateway import LLMGateway

TEST_PR_URL = "https://github.com/Kushagra-Mishra1008/review-swarm-testbed/pull/1"
MAX_TOKENS_BUDGET = 15_000


def check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main() -> int:
    all_passed = True

    gateway = LLMGateway()

    print(f"Running review on: {TEST_PR_URL}\n")
    final_state = run_review(TEST_PR_URL, gateway=gateway)

    findings = final_state.get("final_findings", [])
    errors = final_state.get("errors", [])

    # --- Diagnostics: show what actually happened inside the graph ---
    print("--- Diagnostics ---")
    print(f"Files detected in diff: {[f['file_path'] for f in final_state.get('files', [])]}")
    print(f"Active specialists (lead's decision): {final_state.get('active_specialists', [])}")
    print()

    print(f"Total findings: {len(findings)}")
    print(f"Errors during run: {len(errors)}")
    for e in errors:
        print(f"       ERROR: {e}")

    print("\n--- All findings ---")
    for f in findings:
        print(f"[{f['severity'].upper()}] {f['file']}:{f['line']} — {f['message']}")

    # --- Check 1: found the hardcoded secret ---
    found_secret = any(
        "secret" in f["message"].lower()
        or "api key" in f["message"].lower()
        or "hardcoded" in f["message"].lower()
        or "credential" in f["message"].lower()
        for f in findings
    )
    all_passed &= check("Found the hardcoded API key", found_secret)

    # --- Check 2: found the SQL injection ---
    found_sql_injection = any(
        "sql injection" in f["message"].lower()
        or ("sql" in f["message"].lower() and "inject" in f["message"].lower())
        or ("f-string" in f["message"].lower() and "sql" in f["message"].lower())
        for f in findings
    )
    all_passed &= check("Found the SQL injection", found_sql_injection)

    # --- Check 3: token budget ---
    ledger_spend = gateway._ledger.spent_today("openai/gpt-oss-120b")
    all_passed &= check(
        f"120b token spend under budget ({ledger_spend} / {MAX_TOKENS_BUDGET})",
        ledger_spend < MAX_TOKENS_BUDGET,
    )

    print()
    if all_passed:
        print("Phase 2 vertical slice verification: ALL CHECKS PASSED")
        return 0
    else:
        print("Phase 2 vertical slice verification: SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())