"""
Replaces the old REST-based tools/github.py. Fetches PR data through the
official GitHub MCP server instead of hand-rolled requests calls.

GitHub's MCP server consolidates PR operations into a single
pull_request_read tool with a `method` parameter. get_files returns a
list of content blocks wrapping a JSON string of file entries; each
entry's `patch` field holds the actual unified-diff hunk for that file,
which we reconstruct into standard diff text tools/diff.py already
knows how to parse.
"""

import json
import re

from langchain_mcp_adapters.client import MultiServerMCPClient

PR_URL_RE = re.compile(
    r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
)


class GitHubMCPError(Exception):
    pass


def parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    """Extracts (owner, repo, pr_number) from a GitHub PR URL."""
    match = PR_URL_RE.search(pr_url)
    if not match:
        raise GitHubMCPError(f"Could not parse PR URL: {pr_url}")
    return match.group("owner"), match.group("repo"), int(match.group("number"))


async def fetch_pr_via_mcp(pr_url: str, client: MultiServerMCPClient) -> dict:
    """
    Async entry point: given a PR URL and an already-connected MCP
    client, returns:
        {"owner": str, "repo": str, "pr_number": int,
         "metadata": {...}, "diff": "...", "files_meta": [...]}
    """
    owner, repo, pr_number = parse_pr_url(pr_url)

    tools = await client.get_tools(server_name="github")
    pull_request_read = _find_tool(tools, "pull_request_read")

    raw_metadata = await _call_pr_read(pull_request_read, owner, repo, pr_number, method="get")
    raw_files = await _call_pr_read(pull_request_read, owner, repo, pr_number, method="get_files")

    metadata = _normalize_result(raw_metadata)
    files_result = _normalize_result(raw_files)
    files_meta = files_result if isinstance(files_result, list) else files_result.get("files", [])
    diff = _build_diff_text_from_files(files_meta)

    return {
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number,
        "metadata": metadata,
        "diff": diff,
        "files_meta": files_meta,
    }


def _find_tool(tools: list, name: str):
    for t in tools:
        if t.name == name:
            return t
    raise GitHubMCPError(f"GitHub MCP server did not expose a '{name}' tool — check GITHUB_TOOLSETS.")


async def _call_pr_read(tool, owner: str, repo: str, pr_number: int, method: str):
    try:
        return await tool.ainvoke(
            {"owner": owner, "repo": repo, "pullNumber": pr_number, "method": method}
        )
    except Exception:
        return await tool.ainvoke(
            {"owner": owner, "repo": repo, "pull_number": pr_number, "method": method}
        )


def _normalize_result(raw):
    """
    Confirmed shape: results come back as a list of content blocks,
    e.g. [{"type": "text", "text": "{...json...}"}]. Extracts and
    parses the JSON text from the first block. Falls back gracefully
    (empty dict) for anything unexpected rather than raising.
    """
    if isinstance(raw, list):
        for item in raw:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _build_diff_text_from_files(files_meta: list[dict]) -> str:
    """
    Reconstructs standard unified diff text from per-file patch data.
    Files with no 'patch' (binary files, or very large diffs where
    GitHub omits patches) are skipped rather than crashing.
    """
    parts = []
    for f in files_meta:
        filename = f.get("filename") or f.get("path") or f.get("file")
        patch = f.get("patch")
        if not filename or not patch:
            continue
        parts.append(f"diff --git a/{filename} b/{filename}\n{patch}")
    return "\n".join(parts)