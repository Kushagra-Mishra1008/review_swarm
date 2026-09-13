"""
File Scanner worker — the narrow, cheap tier. One instance gets spawned
per file a specialist wants inspected, running on gpt-oss-20b in a
separate rate-limit pool from the 120b specialists.

Publishes worker_spawn/worker_complete events (via backend/events.py,
through core/run_context.py's contextvar) so the frontend can show
dynamic worker activity — these aren't fixed graph nodes, so they only
show up as event-stream lines and a live per-specialist counter, not as
static boxes in the agent graph.

Both sync (scan_file) and async (ascan_file) versions exist: sync is
used by Phase 2's single-specialist flow, async is used in Phase 3+
when multiple specialists fan out concurrently via asyncio.gather.

Parse and severity failures are recorded to WORKER_ERRORS rather than
being swallowed — a worker that silently returns nothing is
indistinguishable from a worker that found nothing, which made the
Phase 5 eval unreadable.
"""

import json

from pydantic import BaseModel, Field, ValidationError

from agents.state import Finding, FileHunk
from core.gateway import LLMGateway
from core.models import TaskType
from core.run_context import current_run_id
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

# Small models routinely emit off-spec severity words despite the schema
# prompt. Mapping them is strictly better than discarding the finding.
SEVERITY_ALIASES = {
    "critical": "blocker",
    "high": "blocker",
    "severe": "blocker",
    "error": "blocker",
    "fatal": "blocker",
    "warning": "major",
    "warn": "major",
    "medium": "major",
    "moderate": "major",
    "low": "minor",
    "info": "nit",
    "informational": "nit",
    "nitpick": "nit",
    "style": "nit",
    "suggestion": "nit",
    "trivial": "nit",
}

# Module-level diagnostic sink. Appended to on every dropped output so
# a run can be inspected afterwards without re-spending tokens.
WORKER_ERRORS: list[dict] = []


def _record_error(kind: str, detail: str, file_path: str | None = None) -> None:
    WORKER_ERRORS.append({"kind": kind, "detail": detail, "file": file_path})
    _publish_event("worker_error", {"kind": kind, "detail": detail, "file": file_path})


def _publish_event(event_type: str, data: dict) -> None:
    """Same lazy-import pattern as core/gateway.py and agents/graph.py."""
    run_id = current_run_id.get()
    if run_id is None:
        return
    from backend.events import event_bus
    event_bus.publish(run_id, event_type, data)


def _normalize_severity(raw) -> str | None:
    """Lowercase, strip, and map known synonyms. Returns None only if the
    value is genuinely unrecognizable."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower().rstrip(".!")
    s = SEVERITY_ALIASES.get(s, s)
    return s if s in VALID_SEVERITIES else None


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
    "Do not use any other severity word. "
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
    specialist_name: str | None = None,
) -> list[Finding]:
    """Synchronous version — used by Phase 2's single-specialist flow."""
    file_path = file_hunk["file_path"]
    _publish_event("worker_spawn", {"file": file_path, "specialist": specialist_name})

    variable_content = _build_variable_content(file_hunk, focus_hint, static_findings)
    output = _call_worker(gateway, variable_content, file_path=file_path)
    findings = _output_to_findings(output, file_path) if output else []

    _publish_event("worker_complete", {"file": file_path, "specialist": specialist_name, "finding_count": len(findings)})
    return findings


async def ascan_file(
    file_hunk: FileHunk,
    gateway: LLMGateway,
    focus_hint: str | None = None,
    static_findings: list[StaticFinding] | None = None,
    specialist_name: str | None = None,
) -> list[Finding]:
    """Async version — used when multiple specialists fan out concurrently
    (Phase 3+). Queues behind the gateway's semaphore via gateway.acall()."""
    file_path = file_hunk["file_path"]
    _publish_event("worker_spawn", {"file": file_path, "specialist": specialist_name})

    variable_content = _build_variable_content(file_hunk, focus_hint, static_findings)
    output = await _acall_worker(gateway, variable_content, file_path=file_path)
    findings = _output_to_findings(output, file_path) if output else []

    _publish_event("worker_complete", {"file": file_path, "specialist": specialist_name, "finding_count": len(findings)})
    return findings


def _output_to_findings(output: WorkerOutput, file_path: str) -> list[Finding]:
    findings: list[Finding] = []
    for wf in output.findings:
        severity = _normalize_severity(wf.severity)
        if severity is None:
            _record_error(
                "unrecognized_severity",
                f"dropped finding with severity={wf.severity!r}: {wf.message[:80]}",
                file_path,
            )
            continue
        findings.append(
            Finding(
                file=file_path,
                line=wf.line,
                severity=severity,
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


def _extract_json(text: str) -> str:
    """Small models often wrap JSON in prose or markdown fences. Strip the
    fences, then fall back to the outermost brace pair."""
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
        cleaned = cleaned.strip()
    if cleaned.startswith("{"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]
    return cleaned


def _call_worker(
    gateway: LLMGateway,
    variable_content: str,
    retry_note: str = "",
    file_path: str | None = None,
) -> WorkerOutput | None:
    response = gateway.call(
        task_type=TaskType.WORKER,
        system_prompt=WORKER_SYSTEM_PROMPT,
        schema_prompt=WORKER_SCHEMA_PROMPT,
        few_shot=WORKER_FEW_SHOT,
        variable_content=variable_content + retry_note,
        max_tokens=1200,
    )

    try:
        raw = json.loads(_extract_json(response["content"]))
        return WorkerOutput(**raw)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        if retry_note:
            _record_error(
                "parse_failed_twice",
                f"{type(e).__name__}: {e} | raw={str(response.get('content'))[:200]}",
                file_path,
            )
            return None
        return _call_worker(
            gateway,
            variable_content,
            retry_note=f"\n\nYour previous response was invalid: {e}. Respond with valid JSON only.",
            file_path=file_path,
        )


async def _acall_worker(
    gateway: LLMGateway,
    variable_content: str,
    retry_note: str = "",
    file_path: str | None = None,
) -> WorkerOutput | None:
    response = await gateway.acall(
        task_type=TaskType.WORKER,
        system_prompt=WORKER_SYSTEM_PROMPT,
        schema_prompt=WORKER_SCHEMA_PROMPT,
        few_shot=WORKER_FEW_SHOT,
        variable_content=variable_content + retry_note,
        max_tokens=1200,
    )

    try:
        raw = json.loads(_extract_json(response["content"]))
        return WorkerOutput(**raw)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        if retry_note:
            _record_error(
                "parse_failed_twice",
                f"{type(e).__name__}: {e} | raw={str(response.get('content'))[:200]}",
                file_path,
            )
            return None
        return await _acall_worker(
            gateway,
            variable_content,
            retry_note=f"\n\nYour previous response was invalid: {e}. Respond with valid JSON only.",
            file_path=file_path,
        )