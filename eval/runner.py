"""
Runs both the swarm and a single-agent baseline over every collected PR,
caching results to disk per PR so this is safely resumable across
multiple days — the plan estimates ~4 days of token budget for 50 PRs,
so this WILL be interrupted and re-run repeatedly, not run in one sitting.

Stops gracefully (not with an error) if the daily token budget is close
to exhausted, so a scheduled/manual re-run tomorrow just picks up where
today left off.
"""

import asyncio
import json
import os

from agents.graph import run_review
from core.config import DAILY_LEDGER_SOFT_LIMIT_FRACTION, MODEL_120B, MODEL_LIMITS
from core.gateway import GatewayError, LLMGateway
from core.models import TaskType
from eval.collect import load_cached_prs
from tools.diff import build_file_preview

EVAL_RESULTS_DIR = "eval/data/results"

# Cap on how much diff text the single-call baseline sees per file,
# same principle as build_file_preview elsewhere — keeps one massive
# PR from blowing the baseline call's token budget.
BASELINE_MAX_CHARS_PER_FILE = 400
BASELINE_MAX_TOKENS = 1500


BASELINE_SYSTEM_PROMPT = (
    "You are a code reviewer. You are given an entire pull request's "
    "changes at once and must review it in a single pass — no "
    "specialists, no file-by-file breakdown. Find real, concrete issues: "
    "security problems, bugs, performance issues, missing test coverage, "
    "and maintainability concerns."
)

BASELINE_SCHEMA_PROMPT = (
    "Respond with ONLY a JSON object matching this exact shape, no other text:\n"
    '{"findings": [{"file": "path.py", "line": 12, "severity": "blocker", '
    '"category": "security", "message": "..."}]}\n'
    "severity must be one of: blocker, major, minor, nit."
)

BASELINE_FEW_SHOT = (
    "Example input:\n"
    "File: api.py\n"
    '10: API_KEY = "abc123"\n\n'
    'Example output: {"findings": [{"file": "api.py", "line": 10, '
    '"severity": "blocker", "category": "security", "message": '
    '"Hardcoded API key exposed in source code."}]}'
)


def _load_result(pr_number: int) -> dict | None:
    path = os.path.join(EVAL_RESULTS_DIR, f"{pr_number}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_result(pr_number: int, result: dict) -> None:
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    path = os.path.join(EVAL_RESULTS_DIR, f"{pr_number}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def _budget_headroom_remaining(gateway: LLMGateway) -> bool:
    """
    True if there's meaningful room left in today's 120b budget. Stops
    the batch well before the gateway's own hard 90% cutoff would start
    rejecting calls mid-PR, so we always finish whatever PR we started.
    """
    spent = gateway._ledger.spent_today(MODEL_120B)
    limit = MODEL_LIMITS[MODEL_120B].tpd
    safety_margin = 0.75  # stop earlier than the gateway's own 90% cutoff
    return spent < limit * safety_margin


async def run_baseline(pr_record: dict, gateway: LLMGateway) -> dict:
    """
    Single-agent baseline: one LLM call, whole diff, no hierarchy, no
    retrieval, no static analysis. This is the plan's explicit "does the
    architecture earn itself" comparison point.
    """
    files_meta = pr_record.get("files_meta", [])
    parts = []
    for f in files_meta:
        filename = f.get("filename", "")
        patch = f.get("patch", "")
        if not patch:
            continue
        if len(patch) > BASELINE_MAX_CHARS_PER_FILE:
            patch = patch[:BASELINE_MAX_CHARS_PER_FILE] + "\n... (truncated)"
        parts.append(f"File: {filename}\n{patch}")
    variable_content = "\n\n".join(parts) if parts else "(no file changes available)"

    response = gateway.call(
        task_type=TaskType.ORCHESTRATION,
        system_prompt=BASELINE_SYSTEM_PROMPT,
        schema_prompt=BASELINE_SCHEMA_PROMPT,
        few_shot=BASELINE_FEW_SHOT,
        variable_content=variable_content,
        max_tokens=BASELINE_MAX_TOKENS,
    )

    try:
        raw = json.loads(response["content"])
        findings = raw.get("findings", [])
    except (json.JSONDecodeError, TypeError):
        findings = []

    return {
        "findings": findings,
        "tokens": response["usage"]["total_tokens"],
    }


async def run_eval_batch(gateway: LLMGateway | None = None) -> dict:
    """
    Iterates every cached PR (from eval/collect.py), runs the swarm +
    baseline on any not already scored, saves results incrementally.
    Stops early (not an error) if the daily token budget gets tight.

    Returns a summary: {"scored": [...], "skipped_budget": [...],
    "already_done": [...]}
    """
    gateway = gateway or LLMGateway()
    prs = load_cached_prs()

    scored, skipped_budget, already_done = [], [], []

    for pr_record in prs:
        pr_number = pr_record["pr_number"]

        if _load_result(pr_number) is not None:
            already_done.append(pr_number)
            continue

        if not _budget_headroom_remaining(gateway):
            skipped_budget.append(pr_number)
            continue

        try:
            swarm_state = await run_review(pr_record["pr_url"], gateway=gateway)
        except GatewayError as e:
            print(f"PR #{pr_number}: swarm run failed ({e}), skipping")
            continue

        baseline_result = await run_baseline(pr_record, gateway)

        result = {
            "pr_number": pr_number,
            "pr_url": pr_record["pr_url"],
            "swarm_findings": swarm_state.get("final_findings", []),
            "swarm_tokens": gateway._ledger.spent_today(MODEL_120B),  # cumulative, refined in score.py
            "baseline_findings": baseline_result["findings"],
            "baseline_tokens": baseline_result["tokens"],
            "human_review_comments": pr_record.get("review_comments", []),
        }
        _save_result(pr_number, result)
        scored.append(pr_number)
        print(f"PR #{pr_number}: scored ({len(result['swarm_findings'])} swarm findings, "
              f"{len(result['baseline_findings'])} baseline findings)")

    return {"scored": scored, "skipped_budget": skipped_budget, "already_done": already_done}


async def main():
    gateway = LLMGateway()
    summary = await run_eval_batch(gateway)
    print(f"\nScored this run: {len(summary['scored'])}")
    print(f"Skipped (budget): {len(summary['skipped_budget'])}")
    print(f"Already done: {len(summary['already_done'])}")
    if summary["skipped_budget"]:
        print("Re-run this script tomorrow (or after budget resets) to continue.")


if __name__ == "__main__":
    asyncio.run(main())