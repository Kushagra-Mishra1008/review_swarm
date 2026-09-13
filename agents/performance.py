"""
Performance specialist. Runs on gpt-oss-120b, selects which changed
files deserve a performance-focused scan, spawns a worker per selected
file with a performance-specific focus hint.

File selection sees actual code content (via build_file_preview), not
just file paths — selecting by filename alone misses real issues in
plainly-named files and over-selects files that just sound relevant.

An empty selection is treated the same as a failed one: scan everything.
A specialist that declines to pick any file produces a silent
zero-finding review, which is indistinguishable from a clean PR.

Both sync (performance_specialist) and async (aperformance_specialist)
versions exist — async is used in Phase 3+ when all four specialists
fan out concurrently via asyncio.gather.
"""

import asyncio
import json

from pydantic import BaseModel, Field, ValidationError

from agents.state import Finding, ReviewState
from agents.worker import ascan_file, scan_file
from core.gateway import LLMGateway
from core.models import TaskType
from tools.diff import build_file_preview

PERFORMANCE_FOCUS_HINT = (
    "N+1 query patterns, queries or expensive calls inside loops, missing "
    "pagination on unbounded queries, unnecessary repeated computation, "
    "blocking I/O in a hot path, and inefficient data structure choices"
)

# --- Structured output schema for file selection -------------------------

class FileSelection(BaseModel):
    files_to_scan: list[str] = Field(
        description="File paths from the PR worth a performance-focused scan"
    )
    reasoning: str = Field(description="One sentence on the selection")


# --- Static prompt pieces -------------------------------------------------

SELECT_SYSTEM_PROMPT = (
    "You are the Performance Analyst in an automated code review system. "
    "You are shown each changed file's actual added code, not just its "
    "name — base your selection on what the code does, never on the "
    "file name alone. Select every file with a real performance-relevant "
    "change; skip files with no runtime-impacting logic, like pure "
    "documentation or static config. Prefer selecting too many files "
    "over selecting none."
)

SELECT_SCHEMA_PROMPT = (
    "Respond with ONLY a JSON object matching this exact shape, no other text:\n"
    '{"files_to_scan": ["path/to/file.py"], "reasoning": "one sentence"}\n'
    "files_to_scan must be a subset of the file paths shown to you. Include "
    "a file if it contains queries/expensive calls inside loops, blocking "
    "sleeps or I/O in a hot path, unbounded queries, or unnecessary repeated "
    "computation. "
    "Only return an empty list if every file shown is pure documentation, "
    "configuration, or data with no executable code at all."
)

SELECT_FEW_SHOT = (
    "Example input:\n"
    "File: inventory.py\n"
    '5: for product_id in product_ids:\n'
    '6:     time.sleep(0.5)\n'
    '7:     cursor.execute("SELECT stock FROM inventory WHERE product_id = ?", (product_id,))\n\n'
    "File: models.py\n"
    '1: @dataclass\n'
    '2: class OrderItem:\n'
    '3:     product_id: int\n\n'
    'Example output: {"files_to_scan": ["inventory.py"], "reasoning": '
    '"inventory.py has a blocking sleep and a query inside a loop; models.py '
    'is a plain dataclass with no runtime logic."}'
)


def performance_specialist(state: ReviewState, gateway: LLMGateway) -> dict:
    """Synchronous version — used by Phase 2's single-specialist flow."""
    if not state["files"]:
        return {"findings": []}

    selection = _call_file_selection(gateway, state["files"])
    selected_paths = _resolve_selected_paths(selection, [f["file_path"] for f in state["files"]])

    findings: list[Finding] = []
    file_lookup = {f["file_path"]: f for f in state["files"]}
    static_by_file = state.get("static_findings_by_file", {})

    for path in selected_paths:
        file_hunk = file_lookup[path]
        static_findings = static_by_file.get(path)
        worker_findings = scan_file(
            file_hunk, gateway, focus_hint=PERFORMANCE_FOCUS_HINT,
            static_findings=static_findings, specialist_name="performance",
        )
        findings.extend(_tag_findings(worker_findings))

    return {"findings": findings}


async def aperformance_specialist(state: ReviewState, gateway: LLMGateway) -> dict:
    """
    Async version — selects files (one LLM call, now with real code
    content), then scans every selected file CONCURRENTLY via
    asyncio.gather, queued behind the gateway's semaphore.
    """
    if not state["files"]:
        return {"findings": []}

    all_paths = [f["file_path"] for f in state["files"]]
    selection = await _acall_file_selection(gateway, state["files"])
    selected_paths = _resolve_selected_paths(selection, all_paths)

    fell_back = selection is None or not [
        p for p in selection.files_to_scan if p in all_paths
    ]

    file_lookup = {f["file_path"]: f for f in state["files"]}
    static_by_file = state.get("static_findings_by_file", {})

    tasks = [
        ascan_file(
            file_lookup[path],
            gateway,
            focus_hint=PERFORMANCE_FOCUS_HINT,
            static_findings=static_by_file.get(path),
            specialist_name="performance",
        )
        for path in selected_paths
    ]
    results = await asyncio.gather(*tasks) if tasks else []

    findings: list[Finding] = []
    for worker_findings in results:
        findings.extend(_tag_findings(worker_findings))

    diag = f"[diag] performance selected files: {selected_paths}"
    if fell_back:
        diag += " (fell back to all files — selection was empty or failed)"

    return {
        "findings": findings,
        "errors": [diag],
    }


def _resolve_selected_paths(selection: FileSelection | None, file_paths: list[str]) -> list[str]:
    """
    Falls back to every file when selection failed (None) OR came back
    empty. An empty selection means no workers spawn, which reports as
    "no issues found" — a false clean bill of health.
    """
    if selection is None:
        return file_paths
    resolved = [p for p in selection.files_to_scan if p in file_paths]
    return resolved or file_paths


def _tag_findings(findings: list[Finding]) -> list[Finding]:
    for f in findings:
        f["source_agent"] = "performance_specialist"
        f["category"] = f.get("category") or "performance"
    return findings


def _call_file_selection(gateway: LLMGateway, files: list, retry_note: str = "") -> FileSelection | None:
    variable_content = build_file_preview(files) + retry_note

    response = gateway.call(
        task_type=TaskType.SPECIALIST,
        system_prompt=SELECT_SYSTEM_PROMPT,
        schema_prompt=SELECT_SCHEMA_PROMPT,
        few_shot=SELECT_FEW_SHOT,
        variable_content=variable_content,
        max_tokens=600,
    )

    try:
        raw = json.loads(response["content"])
        return FileSelection(**raw)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        if retry_note:
            return None
        return _call_file_selection(
            gateway,
            files,
            retry_note=f"\n\nYour previous response was invalid: {e}. Respond with valid JSON only.",
        )


async def _acall_file_selection(gateway: LLMGateway, files: list, retry_note: str = "") -> FileSelection | None:
    variable_content = build_file_preview(files) + retry_note

    response = await gateway.acall(
        task_type=TaskType.SPECIALIST,
        system_prompt=SELECT_SYSTEM_PROMPT,
        schema_prompt=SELECT_SCHEMA_PROMPT,
        few_shot=SELECT_FEW_SHOT,
        variable_content=variable_content,
        max_tokens=600,
    )

    try:
        raw = json.loads(response["content"])
        return FileSelection(**raw)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        if retry_note:
            return None
        return await _acall_file_selection(
            gateway,
            files,
            retry_note=f"\n\nYour previous response was invalid: {e}. Respond with valid JSON only.",
        )