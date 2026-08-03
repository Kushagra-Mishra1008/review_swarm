"""
ReviewState — the single shared state object that flows through the
entire LangGraph. Every node (lead, specialists, workers) reads from
and writes to this same structure.

Findings from parallel branches (multiple specialists, multiple file
workers) all merge into the same `findings` list via LangGraph's
Annotated + operator.add reducer — so concurrent branches don't clobber
each other's results, they accumulate.
"""

import operator
from typing import Annotated, TypedDict


class Finding(TypedDict):
    file: str
    line: int
    severity: str        # "blocker" | "major" | "minor" | "nit"
    category: str          # e.g. "security", "performance"
    message: str
    suggested_fix: str | None
    source_agent: str        # which agent produced this finding


class FileHunk(TypedDict):
    """One file's changes within the PR diff."""
    file_path: str
    additions: list[tuple[int, str]]   # (line_number, added_line_text)
    deletions: list[tuple[int, str]]
    full_content: str | None             # full file content, if fetched


class ReviewState(TypedDict):
    # --- Input ---
    pr_url: str

    # --- Populated by fetch_pr / parse_diff (plain Python) ---
    files: list[FileHunk]

    # --- Populated by a local clone step, needed so static analysis
    #     tools (ruff/semgrep) have real files on disk to scan ---
    repo_local_path: str

    # --- Populated by run_static_analysis (plain Python, no LLM) —
    #     {file_path: [StaticFinding, ...]}. Written once, no reducer. ---
    static_findings_by_file: dict

    # --- Populated by lead_triage ---
    active_specialists: list[str]

    # --- Accumulates across all branches (lead + every specialist + every worker) ---
    findings: Annotated[list[Finding], operator.add]

    # --- Populated ONCE by the final report node — deduped + ranked
    #     version of `findings`. Separate key (not `findings` itself)
    #     because `findings` uses an operator.add reducer: writing to it
    #     again would append instead of replace, doubling every finding. ---
    final_findings: list[Finding]
    report_text: str

    # --- Populated by budget tracking, read from the gateway after each call ---
    token_spent: Annotated[int, operator.add]

    # --- Accumulates any recoverable errors (e.g. a worker failed to
    #     parse structured output) without killing the whole run ---
    errors: Annotated[list[str], operator.add]