"""
Pulls merged PRs and their human review comments from a target GitHub
repo, caches everything to disk as JSON so the real collection run
(which the plan estimates at several days of token budget downstream)
never re-fetches what it already has.

Uses the GitHub MCP server (same one wired into agents/graph.py in
Phase 4) rather than a separate REST client — one GitHub access path
for the whole project.
"""

import asyncio
import json
import os

from core.mcp_client import build_static_mcp_client
from tools.github_mcp import _call_pr_read, _find_tool, _normalize_result

EVAL_DATA_DIR = "eval/data/prs"


class CollectedPR:
    """Plain dict shape, saved as {pr_number}.json under EVAL_DATA_DIR:
    {"pr_number": int, "pr_url": str, "metadata": {...},
     "files_meta": [...], "review_comments": [...]}"""
    pass


async def collect_prs(owner: str, repo: str, count: int, mcp_client=None) -> list[dict]:
    """
    Fetches `count` recently-merged PRs from owner/repo, each with its
    file changes and human review comments, and saves each to disk
    individually. Skips any PR already cached locally — safe to re-run
    this across multiple sessions without re-fetching.

    Returns the list of newly collected PRs (already-cached ones are
    not re-returned, since the caller — runner.py — reads from disk
    anyway).
    """
    os.makedirs(EVAL_DATA_DIR, exist_ok=True)
    mcp_client = mcp_client or build_static_mcp_client()

    tools = await mcp_client.get_tools(server_name="github")
    pull_request_read = _find_tool(tools, "pull_request_read")

    # list_pull_requests fetches candidates; substantial review comment
    # count is the filter for "real code review happened here", not
    # just merged status.
    list_tool = _find_tool(tools, "list_pull_requests")
    raw_list = await list_tool.ainvoke(
        {"owner": owner, "repo": repo, "state": "closed", "sort": "updated", "perPage": count * 3}
    )
    candidates = _normalize_result(raw_list)
    if isinstance(candidates, dict):
        candidates = candidates.get("pull_requests", candidates.get("items", []))

    collected = []
    for pr_summary in candidates:
        if len(collected) >= count:
            break

        pr_number = pr_summary.get("number")
        if pr_number is None:
            continue

        cache_path = os.path.join(EVAL_DATA_DIR, f"{pr_number}.json")
        if os.path.exists(cache_path):
            continue  # already collected in a prior run

        if not pr_summary.get("merged") and pr_summary.get("state") != "closed":
            continue

        metadata = _normalize_result(
            await _call_pr_read(pull_request_read, owner, repo, pr_number, method="get")
        )
        if not metadata.get("merged"):
            continue  # only want merged PRs — closed-without-merge isn't useful signal

        files_result = _normalize_result(
            await _call_pr_read(pull_request_read, owner, repo, pr_number, method="get_files")
        )
        files_meta = files_result if isinstance(files_result, list) else files_result.get("files", [])

        comments_result = _normalize_result(
            await _call_pr_read(pull_request_read, owner, repo, pr_number, method="get_review_comments")
        )
        review_comments = comments_result if isinstance(comments_result, list) else comments_result.get("comments", [])

        # Skip PRs with little to no real review discussion — the eval
        # needs substantive human comments to score against, per the plan.
        if len(review_comments) < 2:
            continue

        record = {
            "pr_number": pr_number,
            "pr_url": f"https://github.com/{owner}/{repo}/pull/{pr_number}",
            "metadata": metadata,
            "files_meta": files_meta,
            "review_comments": review_comments,
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        collected.append(record)

    return collected


def load_cached_prs() -> list[dict]:
    """Loads every PR already collected to disk — this is what
    runner.py actually iterates over."""
    if not os.path.exists(EVAL_DATA_DIR):
        return []

    records = []
    for filename in sorted(os.listdir(EVAL_DATA_DIR)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(EVAL_DATA_DIR, filename), "r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect merged PRs + review comments for eval")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    collected = await collect_prs(args.owner, args.repo, args.count)
    print(f"Collected {len(collected)} new PRs (cached at {EVAL_DATA_DIR}/)")
    print(f"Total cached: {len(load_cached_prs())}")


if __name__ == "__main__":
    asyncio.run(main())