"""
Verification gate for core/gateway.py.

Run this after setting GROQ_API_KEY. It exercises the gateway end-to-end:
a real call, a cache hit on the same call, and a check that usage got
recorded in the daily ledger. If this script passes, Phase 0 is done.
"""

import sys

from core.budget import BudgetExceededError
from core.gateway import GatewayError, LLMGateway
from core.models import TaskType


def check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main() -> int:
    all_passed = True

    try:
        gateway = LLMGateway()
    except Exception as e:
        print(f"[FAIL] Could not construct LLMGateway: {e}")
        print("Is GROQ_API_KEY set?")
        return 1

    system_prompt = "You are a terse assistant. Reply in under 10 words."
    schema_prompt = "Respond with plain text only, no formatting."
    few_shot = "Example: Q: What is 2+2? A: Four."
    variable_content = "What is the capital of France?"

    # --- Call 1: should be a real API call, not cached ---
    try:
        response1 = gateway.call(
            task_type=TaskType.WORKER,
            system_prompt=system_prompt,
            schema_prompt=schema_prompt,
            few_shot=few_shot,
            variable_content=variable_content,
            max_tokens=150,
        )
    except (GatewayError, BudgetExceededError) as e:
        print(f"[FAIL] First call raised: {e}")
        return 1

    all_passed &= check("First call returns content", bool(response1.get("content")))
    all_passed &= check("First call is not cached", response1.get("cached") is False)
    all_passed &= check(
        "First call has usage data",
        response1.get("usage", {}).get("total_tokens", 0) > 0,
    )

    print(f"       Response: {response1['content']!r}")
    print(f"       Tokens used: {response1['usage']['total_tokens']}")

    # --- Call 2: identical request, should hit cache ---
    response2 = gateway.call(
        task_type=TaskType.WORKER,
        system_prompt=system_prompt,
        schema_prompt=schema_prompt,
        few_shot=few_shot,
        variable_content=variable_content,
        max_tokens=150,
    )

    all_passed &= check("Second identical call is cached", response2.get("cached") is True)
    all_passed &= check(
        "Cached response content matches original",
        response2.get("content") == response1.get("content"),
    )

    # --- Ledger check: daily spend should reflect call 1's usage ---
    model_name = resolve_model_for_check()
    spent = gateway._ledger.spent_today(model_name)
    all_passed &= check("Daily ledger recorded nonzero spend", spent > 0)
    print(f"       Ledger spend today for worker model: {spent} tokens")

    print()
    if all_passed:
        print("Phase 0 gateway verification: ALL CHECKS PASSED")
        return 0
    else:
        print("Phase 0 gateway verification: SOME CHECKS FAILED")
        return 1


def resolve_model_for_check() -> str:
    from core.models import resolve_model
    return resolve_model(TaskType.WORKER)


if __name__ == "__main__":
    sys.exit(main())