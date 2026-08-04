"""
Security specialist. Runs on gpt-oss-120b, selects which changed files
deserve a security-focused scan, spawns a worker per selected file with
a security-specific focus hint.

File selection now sees actual code content (via build_file_preview),
not just file paths — selecting by filename alone was missing real
issues in files with unassuming names (e.g. a hardcoded key in
"notifications.py") while over-selecting files that just sound
security-adjacent by name.

Both sync (security_specialist) and async (asecurity_specialist)
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

SECURITY_FOCUS_HINT = (
    "hardcoded secrets/API keys/credentials, SQL injection, command "
    "injection, unsafe deserialization, missing authentication/authorization "
    "checks, and insecure use of user input"
)

# --- Structured output schema for file selection -------------------------

class FileSelection(BaseModel):
    files_to_scan: list[str] = Field(
        description="File paths from the PR worth a security-focused scan"
    )
    reasoning: str = Field(description="One sentence on the selection")


# --- Static prompt pieces -------------------------------------------------

SELECT_SYSTEM_PROMPT = (
    "You are the Security Analyst in an automated code review system. "
    "You are shown each changed file's actual added code, not just its "
    "name — base your selection on what the code does, never on the "
    "file name alone. A file named 'utils.py' can hold a hardcoded "
    "secret; a file named 'validators.py' can be entirely clean. Select "
    "every file with a real security-relevant change; skip files with "
    "no security-relevant logic, like pure documentation or config "
    "formatting."
)

SELECT_SCHEMA_PROMPT = (
    "Respond with ONLY a JSON object matching this exact shape, no other text:\n"
    '{"files_to_scan": ["path/to/file.py"], "reasoning": "one sentence"}\n'
    "files_to_scan must be a subset of the file paths shown to you. Include "
    "a file if it contains hardcoded secrets/keys, raw SQL built with string "
    "formatting, unsafe deserialization, or missing auth/authz checks."
)

SELECT_FEW_SHOT = (
    "Example input:\n"
    "File: notifications.py\n"
    '2: SENDGRID_API_KEY = "SG.abc123..."\n\n'
    "File: validators.py\n"
    '1: def validate_order_payload(data): return "items" in data\n\n'
    'Example output: {"files_to_scan": ["notifications.py"], "reasoning": '
    '"notifications.py has a hardcoded API key; validators.py is a plain '
    'boolean check with no security-relevant logic."}'
)


def security_specialist(state: ReviewState, gateway: LLMGateway) -> dict:
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
            file_hunk, gateway, focus_hint=SECURITY_FOCUS_HINT,
            static_findings=static_findings, specialist_name="security",
        )
        findings.extend(_tag_findings(worker_findings))

    return {"findings": findings}


async def asecurity_specialist(state: ReviewState, gateway: LLMGateway) -> dict:
    """
    Async version — selects files (one LLM call, now with real code
    content), then scans every selected file CONCURRENTLY via
    asyncio.gather. Each ascan_file call queues behind the gateway's
    semaphore, so real network concurrency stays capped.
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
            focus_hint=SECURITY_FOCUS_HINT,
            static_findings=static_by_file.get(path),
            specialist_name="security",
        )
        for path in selected_paths
    ]
    results = await asyncio.gather(*tasks) if tasks else []

    findings: list[Finding] = []
    for worker_findings in results:
        findings.extend(_tag_findings(worker_findings))

    return {
        "findings": findings,
        "errors": [f"[diag] security selected files: {selected_paths}"],
    }


async def asecurity_specialist(state: ReviewState, gateway: LLMGateway) -> dict:
    """
    Async version — selects files (one LLM call, now with real code
    content), then scans every selected file CONCURRENTLY via
    asyncio.gather. Each ascan_file call queues behind the gateway's
    semaphore, so real network concurrency stays capped.
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
            focus_hint=SECURITY_FOCUS_HINT,
            static_findings=static_by_file.get(path),
        )
        for path in selected_paths
    ]
    results = await asyncio.gather(*tasks) if tasks else []

    findings: list[Finding] = []
    for worker_findings in results:
        findings.extend(_tag_findings(worker_findings))

    return {
        "findings": findings,
        "errors": [f"[diag] security selected files: {selected_paths}"],
    }


def _resolve_selected_paths(selection: FileSelection | None, file_paths: list[str]) -> list[str]:
    if selection is None:
        return file_paths  # fall back to scanning everything on parse failure
    return [p for p in selection.files_to_scan if p in file_paths]


def _tag_findings(findings: list[Finding]) -> list[Finding]:
    for f in findings:
        f["source_agent"] = "security_specialist"
        f["category"] = f.get("category") or "security"
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