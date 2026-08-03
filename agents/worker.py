"""
File Scanner worker — the narrow, cheap tier. One instance gets spawned
per file a specialist wants inspected, running on gpt-oss-20b in a
separate rate-limit pool from the 120b specialists.

A worker does one thing: given a single file's diff hunk (plus optional
focus hint and static-analysis evidence), look for concrete issues and
return structured findings. No orchestration, no judgment about which
specialists to run — that's the lead's job.

Both sync (scan_file) and async (ascan_file) versions exist: sync is
used by Phase 2's single-specialist flow, async is used in Phase 3+
when multiple specialists fan out concurrently via asyncio.gather and
workers need to run under the gateway's semaphore rather than blocking
each other.
"""

import json

from pydantic import BaseModel, Field, ValidationError

from agents.state import Finding, FileHunk
from core.gateway import LLMGateway
from core.models import TaskType
from tools.static_analysis import StaticFinding, format_evidence_for_prompt

# --- Structured output schema -------------------------------------------

class WorkerFinding(BaseModel):
    line: int = Field(description="Line number in the new file where the issue is")
    severity: str = Field(description="blocker | major | minor | nit")
    category: str = Field(description="e.g. security, performance, style")
    message: str = Field(description="What's wrong, in one sentence")
    suggested_fix: str | None = Field(default=None, description="Optional one-line fix suggestion")


class WorkerOutput(BaseModel):
    findings: list[WorkerFinding] = Field(default_factory=list)


VALID_SEVERITIES = {"blocker", "major", "minor", "nit"}

# --- Static prompt pieces -------------------------------------------------

WORKER_SYSTEM_PROMPT = (
    "You are a file-level code scanner in an automated review pipeline. "
    "You inspect one file's changes and report concrete, specific issues. "
    "You do not comment on style preferences or make judgment calls about "
    "architecture — only real, actionable problems: security issues, bugs, "
    "and clear defects introduced by this diff. If you are given machine-"
    "detected issues from static analysis tools, verify which are real "
    "problems in this diff and include only those — plus anything the "
    "tools missed."
)

WORKER_SCHEMA_PROMPT = (
    "Respond with ONLY a JSON object matching this exact shape, no other text:\n"
    '{"findings": [{"line": 12, "severity": "blocker", "category": "security", '
    '"message": "...", "suggested_fix": "..."}]}\n'
    "severity must be one of: blocker, major, minor, nit. "
    "If there are no issues, respond with {\"findings\": []}. "
    "Only report issues on lines that were actually added or changed."
)

WORKER_FEW_SHOT = (
    "Example input: an added line `cursor.execute(f\"SELECT * FROM t WHERE id = "
    "{user_input}\")` at line 40.\n"
    'Example output: {"findings": [{"line": 40, "severity": "blocker", '
    '"category": "security", "message": "SQL query built with an f-string '
    'using unsanitized user input — SQL injection risk.", "suggested_fix": '
    '"Use a parameterized query with a placeholder instead of an f-string."}]}'
)


def scan_file(
    file_hunk: FileHunk,
    gateway: LLMGateway,
    focus_hint: str | None = None,
    static_findings: list[StaticFinding] | None = None,
) -> list[Finding]:
    """
    Synchronous version — used by Phase 2's single-specialist flow.
    Returns an empty list (not an exception) on repeated validation
    failure — one bad worker call should never crash the whole review.
    """
    variable_content = _build_variable_content(file_hunk, focus_hint, static_findings)

    output = _call_worker(gateway, variable_content)
    if output is None:
        return []

    return _output_to_findings(output, file_hunk["file_path"])


async def ascan_file(
    file_hunk: FileHunk,
    gateway: LLMGateway,
    focus_hint: str | None = None,
    static_findings: list[StaticFinding] | None = None,
) -> list[Finding]:
    """
    Async version — used when multiple specialists fan out concurrently
    (Phase 3+). Calls gateway.acall() instead of gateway.call(), so this
    worker's request queues behind the gateway's semaphore alongside
    every other concurrent call, rather than blocking the event loop.
    """
    variable_content = _build_variable_content(file_hunk, focus_hint, static_findings)

    output = await _acall_worker(gateway, variable_content)
    if output is None:
        return []

    return _output_to_findings(output, file_hunk["file_path"])


def _output_to_findings(output: WorkerOutput, file_path: str) -> list[Finding]:
    findings: list[Finding] = []
    for wf in output.findings:
        if wf.severity not in VALID_SEVERITIES:
            continue
        findings.append(
            Finding(
                file=file_path,
                line=wf.line,
                severity=wf.severity,
                category=wf.category,
                message=wf.message,
                suggested_fix=wf.suggested_fix,
                source_agent="worker",
            )
        )
    return findings


def _build_variable_content(
    file_hunk: FileHunk,
    focus_hint: str | None,
    static_findings: list[StaticFinding] | None,
) -> str:
    added_lines = "\n".join(f"{ln}: {text}" for ln, text in file_hunk["additions"])
    parts = [f"File: {file_hunk['file_path']}", "", "Added/changed lines:", added_lines]

    if static_findings:
        evidence = format_evidence_for_prompt(static_findings)
        if evidence:
            parts.append(f"\n{evidence}")

    if focus_hint:
        parts.append(f"\nFocus especially on: {focus_hint}")

    return "\n".join(parts)


def _call_worker(gateway: LLMGateway, variable_content: str, retry_note: str = "") -> WorkerOutput | None:
    response = gateway.call(
        task_type=TaskType.WORKER,
        system_prompt=WORKER_SYSTEM_PROMPT,
        schema_prompt=WORKER_SCHEMA_PROMPT,
        few_shot=WORKER_FEW_SHOT,
        variable_content=variable_content + retry_note,
        max_tokens=500,
    )

    try:
        raw = json.loads(response["content"])
        return WorkerOutput(**raw)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        if retry_note:
            return None
        return _call_worker(
            gateway,
            variable_content,
            retry_note=f"\n\nYour previous response was invalid: {e}. Respond with valid JSON only.",
        )


async def _acall_worker(gateway: LLMGateway, variable_content: str, retry_note: str = "") -> WorkerOutput | None:
    response = await gateway.acall(
        task_type=TaskType.WORKER,
        system_prompt=WORKER_SYSTEM_PROMPT,
        schema_prompt=WORKER_SCHEMA_PROMPT,
        few_shot=WORKER_FEW_SHOT,
        variable_content=variable_content + retry_note,
        max_tokens=500,
    )

    try:
        raw = json.loads(response["content"])
        return WorkerOutput(**raw)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        if retry_note:
            return None
        return await _acall_worker(
            gateway,
            variable_content,
            retry_note=f"\n\nYour previous response was invalid: {e}. Respond with valid JSON only.",
        )