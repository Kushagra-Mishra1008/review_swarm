"""
Verification gate for Phase 4: runs the full pipeline through all three
MCP servers (GitHub, Filesystem, Repo Index) end-to-end against a real
PR, and confirms:
  - GitHub MCP successfully fetched PR data (replacing REST entirely)
  - Repo Index MCP connected successfully mid-run and advertised its tools
  - the review still produces real findings, same as Phase 3
  - no direct REST calls remain in tools/ (static source check)
"""

import asyncio
import os
import sys
import time

from agents.graph import run_review
from core.gateway import GatewayError, LLMGateway

TEST_PR_URL = "https://github.com/Kushagra-Mishra1008/review-swarm-testbed/pull/2"


def check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def check_no_rest_calls_remain() -> bool:
    """
    Static source check: tools/github.py should no longer exist, and no
    file under tools/ should import 'requests' anymore, since GitHub MCP
    replaced the last REST usage. tools/github_mcp.py is exempt by name.
    """
    tools_dir = "tools"
    if os.path.exists(os.path.join(tools_dir, "github.py")):
        print("       tools/github.py still exists")
        return False

    for filename in os.listdir(tools_dir):
        if not filename.endswith(".py") or filename == "github_mcp.py":
            continue
        path = os.path.join(tools_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if "import requests" in content:
            print(f"       {path} still imports requests")
            return False

    return True


async def main_async() -> int:
    all_passed = True

    gateway = LLMGateway()

    print(f"Running review on: {TEST_PR_URL}\n")
    start = time.monotonic()

    try:
        final_state = await run_review(TEST_PR_URL, gateway=gateway)
    except GatewayError as e:
        print(f"[FAIL] Run raised GatewayError: {e}")
        return 1
    except Exception as e:
        print(f"[FAIL] Run raised an unexpected exception: {type(e).__name__}: {e}")
        return 1

    elapsed = time.monotonic() - start

    errors = final_state.get("errors", [])
    findings = final_state.get("final_findings", [])

    print("--- Diagnostics ---")
    for e in errors:
        print(f"       {e}")
    print(f"\nWall clock: {elapsed:.1f}s")
    print(f"Total findings: {len(findings)}\n")

    # --- Check 1: PR was actually fetched (files populated means GitHub MCP worked) ---
    all_passed &= check(
        "GitHub MCP fetched PR data (files list populated)",
        len(final_state.get("files", [])) > 0,
    )

    # --- Check 2: Repo Index MCP connected successfully mid-run ---
    repo_index_connected = any("repo_index MCP connected" in e for e in errors)
    all_passed &= check("Repo Index MCP connected mid-run", repo_index_connected)

    # --- Check 3: review still produces real findings end-to-end ---
    all_passed &= check("Review produced findings", len(findings) > 0)

    # --- Check 4: no direct REST calls remain ---
    all_passed &= check("No direct REST calls remain in tools/", check_no_rest_calls_remain())

    print("\n--- Findings ---")
    for f in findings:
        print(f"[{f['severity'].upper()}] {f['category']} — {f['file']}:{f['line']} — {f['message']}")

    print()
    if all_passed:
        print("Phase 4 verification: ALL CHECKS PASSED")
        return 0
    else:
        print("Phase 4 verification: SOME CHECKS FAILED")
        return 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())