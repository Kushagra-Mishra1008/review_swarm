"""
Fetches PR data from GitHub's REST API — the diff, file list, and PR
metadata. This is the Phase 2 stand-in; Phase 4 replaces this with the
official GitHub MCP server. Kept deliberately thin so that swap is easy.
"""

import os
import re

import requests

GITHUB_API_BASE = "https://api.github.com"

PR_URL_RE = re.compile(
    r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
)


class GitHubError(Exception):
    pass


def parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    """Extracts (owner, repo, pr_number) from a GitHub PR URL."""
    match = PR_URL_RE.search(pr_url)
    if not match:
        raise GitHubError(f"Could not parse PR URL: {pr_url}")
    return match.group("owner"), match.group("repo"), int(match.group("number"))


class GitHubClient:
    def __init__(self, token: str | None = None):
        self._token = token or os.environ.get("GITHUB_TOKEN")
        if not self._token:
            raise GitHubError(
                "No GitHub token found. Set the GITHUB_TOKEN environment variable."
            )
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def get_pr_metadata(self, owner: str, repo: str, pr_number: int) -> dict:
        """Title, description, base/head branches, author, etc."""
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"
        response = self._session.get(url, timeout=15)
        self._raise_for_status(response, context=f"fetching PR metadata for {url}")
        return response.json()

    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """
        Raw unified diff text for the whole PR — this is what
        tools/diff.py parses into FileHunk objects.
        """
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"
        headers = {**self._session.headers, "Accept": "application/vnd.github.v3.diff"}
        response = requests.get(url, headers=headers, timeout=15)
        self._raise_for_status(response, context=f"fetching PR diff for {url}")
        return response.text

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[dict]:
        """
        Per-file metadata (status: added/modified/removed, additions,
        deletions count) — useful for the lead's cheap triage summary
        without needing the full diff text.
        """
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/files"
        response = self._session.get(url, timeout=15)
        self._raise_for_status(response, context=f"fetching PR files for {url}")
        return response.json()

    def get_file_content(self, owner: str, repo: str, path: str, ref: str) -> str | None:
        """
        Full content of a file at a given ref (branch/commit). Used when
        a specialist needs more context than just the diff hunk — e.g. to
        see the whole function a change lives in, not just the changed
        lines.

        Returns None for binary files or files GitHub can't decode as
        text, rather than raising — callers should treat that as
        "no extra context available," not a fatal error.
        """
        import base64

        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
        response = self._session.get(url, params={"ref": ref}, timeout=15)
        self._raise_for_status(response, context=f"fetching file content for {path}")

        data = response.json()
        if data.get("encoding") != "base64":
            return None
        try:
            return base64.b64decode(data["content"]).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return None

    @staticmethod
    def _raise_for_status(response: "requests.Response", context: str) -> None:
        if response.status_code == 404:
            raise GitHubError(f"Not found (404) — check token permissions. Context: {context}")
        if response.status_code == 403:
            raise GitHubError(f"Forbidden (403) — likely rate-limited or missing scope. Context: {context}")
        if not response.ok:
            raise GitHubError(f"GitHub API error {response.status_code}: {response.text[:300]}. Context: {context}")


def fetch_pr(pr_url: str, token: str | None = None) -> dict:
    """
    Convenience entry point: given a PR URL, returns everything
    downstream code needs in one dict:
        {"metadata": {...}, "diff": "...", "files_meta": [...],
         "owner": str, "repo": str, "pr_number": int}
    """
    owner, repo, pr_number = parse_pr_url(pr_url)
    client = GitHubClient(token=token)

    return {
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number,
        "metadata": client.get_pr_metadata(owner, repo, pr_number),
        "diff": client.get_pr_diff(owner, repo, pr_number),
        "files_meta": client.get_pr_files(owner, repo, pr_number),
    }