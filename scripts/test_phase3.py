"""
Verification gate for Phase 3: runs the full concurrent-specialist
pipeline against a real 10-file PR and confirms:
  - zero 429s (rate limit errors) in the run
  - all activated specialists produced findings
  - total 120b spend under 20K tokens
  - wall clock under 4 minutes
  - static analysis evidence actually reached at least one file
"""

import asyncio
import sys
import time

from agents.graph import run_review
from core.gateway import GatewayError, LLMGateway

TEST_PR_URL = "https://github.com/Kushagra-Mishra1008/review-swarm-testbed/pull/2"
MAX_TOKENS_BUDGET_120B = 20_000
MAX_WALL_CLOCK_SECONDS = 240


def check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


async def main_async() -> int:
    all_passed = True

    gateway = LLMGateway()

    print(f"Running review on: {TEST_PR_URL}\n")
    start = time.monotonic()

    try:
        final_state = await run_review(TEST_PR_URL, gateway=gateway)
    except GatewayError as e:
        print(f"[FAIL] Run raised GatewayError (likely a 429 that exhausted retries): {e}")
        return 1

    elapsed = time.monotonic() - start

    # --- Diagnostics ---
    print("--- Diagnostics ---")
    print(f"Files detected in diff: {[f['file_path'] for f in final_state.get('files', [])]}")
    print(f"Active specialists (lead's decision): {final_state.get('active_specialists', [])}")

    static_by_file = final_state.get("static_findings_by_file", {})
    static_file_count = sum(1 for v in static_by_file.values() if v)
    print(f"Files with static analysis evidence: {static_file_count}")
    print(f"Wall clock: {elapsed:.1f}s\n")

    findings = final_state.get("final_findings", [])
    errors = final_state.get("errors", [])

    print(f"Total findings: {len(findings)}")
    print(f"Errors during run: {len(errors)}")
    for e in errors:
        print(f"       ERROR: {e}")

    print("\n--- All findings ---")
    for f in findings:
        print(f"[{f['severity'].upper()}] {f['category']} — {f['file']}:{f['line']} — {f['message']}  (via {f['source_agent']})")

    # --- Check 1: no rate limit errors ---
    no_429_errors = not any("429" in e for e in errors)
    all_passed &= check("Zero 429 errors during run", no_429_errors)

    # --- Check 2: all activated specialists produced at least one finding ---
    active = final_state.get("active_specialists", [])
    agents_with_findings = {f["source_agent"] for f in findings}
    for specialist in active:
        expected_agent = f"{specialist}_specialist"
        produced = expected_agent in agents_with_findings
        all_passed &= check(f"'{specialist}' specialist produced findings", produced)

    # --- Check 3: token budget ---
    ledger_spend = gateway._ledger.spent_today("openai/gpt-oss-120b")
    all_passed &= check(
        f"120b token spend under budget ({ledger_spend} / {MAX_TOKENS_BUDGET_120B})",
        ledger_spend < MAX_TOKENS_BUDGET_120B,
    )

    # --- Check 4: wall clock ---
    all_passed &= check(
        f"Wall clock under {MAX_WALL_CLOCK_SECONDS}s ({elapsed:.1f}s)",
        elapsed < MAX_WALL_CLOCK_SECONDS,
    )

    # --- Check 5: static analysis evidence reached at least one file ---
    all_passed &= check("Static analysis produced evidence for at least one file", static_file_count > 0)

    print()
    if all_passed:
        print("Phase 3 verification: ALL CHECKS PASSED")
        return 0
    else:
        print("Phase 3 verification: SOME CHECKS FAILED")
        return 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())