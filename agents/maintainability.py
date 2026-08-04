"""
Maintainability specialist. Runs on gpt-oss-120b, selects which changed
files deserve a maintainability-focused scan, spawns a worker per
selected file with a maintainability-specific focus hint.

File selection sees actual code content (via build_file_preview), not
just file paths — needed to judge things like parameter counts or
duplicated logic, which are invisible from a filename.

Both sync (maintainability_specialist) and async
(amaintainability_specialist) versions exist — async is used in Phase 3+
when all four specialists fan out concurrently via asyncio.gather.
"""

import asyncio
import json

from pydantic import BaseModel, Field, ValidationError

from agents.state import Finding, ReviewState
from agents.worker import ascan_file, scan_file
from core.gateway import LLMGateway
from core.models import TaskType
from tools.diff import build_file_preview

MAINTAINABILITY_FOCUS_HINT = (
    "unclear naming, duplicated logic that should be extracted, functions "
    "doing too many unrelated things, missing or misleading docstrings on "
    "non-obvious logic, and magic numbers/strings that should be named "
    "constants"
)

# --- Structured output schema for file selection -------------------------

class FileSelection(BaseModel):
    files_to_scan: list[str] = Field(
        description="File paths from the PR worth a maintainability-focused scan"
    )
    reasoning: str = Field(description="One sentence on the selection")


# --- Static prompt pieces -------------------------------------------------

SELECT_SYSTEM_PROMPT = (
    "You are the Maintainability Analyst in an automated code review "
    "system. You are shown each changed file's actual added code, not "
    "just its name — base your selection on what the code does, never "
    "on the file name alone. Select files with real logic worth "
    "assessing; skip files with no real logic, like pure documentation "
    "or generated files."
)

SELECT_SCHEMA_PROMPT = (
    "Respond with ONLY a JSON object matching this exact shape, no other text:\n"
    '{"files_to_scan": ["path/to/file.py"], "reasoning": "one sentence"}\n'
    "files_to_scan must be a subset of the file paths shown to you. Include "
    "a file if it has functions with many parameters or unrelated "
    "responsibilities, duplicated logic that should be extracted, or magic "
    "numbers/strings without named constants."
)

SELECT_FEW_SHOT = (
    "Example input:\n"
    "File: discounts.py\n"
    '1: def apply_discount(price, tier):\n'
    '2:     if tier == "bronze": return price * 0.95\n'
    '3:     elif tier == "silver": return price * 0.90\n\n'
    "File: README.md\n"
    '1: # review-swarm-testbed\n\n'
    'Example output: {"files_to_scan": ["discounts.py"], "reasoning": '
    '"discounts.py has an if/elif chain with unnamed magic numbers that '
    'should be a lookup table; README.md has no logic to assess."}'
)


def maintainability_specialist(state: ReviewState, gateway: LLMGateway) -> dict:
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
            file_hunk, gateway, focus_hint=MAINTAINABILITY_FOCUS_HINT,
            static_findings=static_findings, specialist_name="maintainability",
        )
        findings.extend(_tag_findings(worker_findings))

    return {"findings": findings}


async def amaintainability_specialist(state: ReviewState, gateway: LLMGateway) -> dict:
    """
    Async version — selects files (one LLM call, now with real code
    content), then scans every selected file CONCURRENTLY via
    asyncio.gather, queued behind the gateway's semaphore.
    """
    if not state["files"]:
        return {"findings": []}

    selection = await _acall_file_selection(gateway, state["files"])
    selected_paths = _resolve_selected_paths(selection, [f["file_path"] for f in state["files"]])

    file_lookup = {f["file_path"]: f for f in state["files"]}
    static_by_file = state.get("static_findings_by_file", {})

    tasks = [
        ascan_file(
            file_lookup[path],
            gateway,
            focus_hint=MAINTAINABILITY_FOCUS_HINT,
            static_findings=static_by_file.get(path),
            specialist_name="maintainability",
        )
        for path in selected_paths
    ]
    results = await asyncio.gather(*tasks) if tasks else []

    findings: list[Finding] = []
    for worker_findings in results:
        findings.extend(_tag_findings(worker_findings))

    return {
        "findings": findings,
        "errors": [f"[diag] maintainability selected files: {selected_paths}"],
    }


def _resolve_selected_paths(selection: FileSelection | None, file_paths: list[str]) -> list[str]:
    if selection is None:
        return file_paths
    return [p for p in selection.files_to_scan if p in file_paths]


def _tag_findings(findings: list[Finding]) -> list[Finding]:
    for f in findings:
        f["source_agent"] = "maintainability_specialist"
        f["category"] = f.get("category") or "maintainability"
    return findings


def _call_file_selection(gateway: LLMGateway, files: list, retry_note: str = "") -> FileSelection | None:
    variable_content = build_file_preview(files) + retry_note

    response = gateway.call(
        task_type=TaskType.SPECIALIST,
        system_prompt=SELECT_SYSTEM_PROMPT,
        schema_prompt=SELECT_SCHEMA_PROMPT,
        few_shot=SELECT_FEW_SHOT,
        variable_content=variable_content,
        max_tokens=300,
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
        max_tokens=300,
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