"""
Review Lead — the orchestrator.

Owns three kinds of work, kept separate on purpose:
  1. apply_activation_rules — plain Python, runs BEFORE any LLM call.
     Cheap deterministic rules that decide specialists without spending
     a single token when the answer is obvious.
  2. lead_triage — the ONE LLM call this file makes, and ONLY when the
     rules don't already produce a confident decision. Structured JSON,
     max_tokens=300.
  3. dedupe_findings / rank_findings / generate_report — plain Python.
     No judgment needed to sort a list or drop exact duplicates.
"""

import json

from pydantic import BaseModel, Field, ValidationError

from agents.state import Finding, FileHunk, ReviewState
from core.gateway import LLMGateway
from core.models import TaskType
from tools.diff import build_file_preview

# --- Structured output schema for triage ------------------------------

class TriageDecision(BaseModel):
    active_specialists: list[str] = Field(
        description="Which specialists to activate: subset of "
                    "['security', 'performance', 'testing', 'maintainability']"
    )
    reasoning: str = Field(description="One sentence on why these specialists")


VALID_SPECIALISTS = {"security", "performance", "testing", "maintainability"}

# Small-PR threshold for the "maintainability only" conditional rule.
SMALL_DIFF_LINE_THRESHOLD = 20

# Filenames that indicate a dependency change — triggers security, always.
DEPENDENCY_FILENAMES = {
    "requirements.txt", "pyproject.toml", "poetry.lock", "pipfile",
    "pipfile.lock", "package.json", "package-lock.json", "yarn.lock",
    "go.mod", "go.sum", "cargo.toml", "cargo.lock", "gemfile", "gemfile.lock",
}


# --- Step 1: conditional rules (plain Python, no LLM) -------------------

def apply_activation_rules(files: list[FileHunk]) -> list[str] | None:
    """
    Applies the plan's deterministic pre-LLM rules, in order:
      - diff under SMALL_DIFF_LINE_THRESHOLD total changed lines ->
        maintainability only (checked first — a tiny diff shouldn't
        trigger a full multi-specialist review)
      - dependency files changed -> security, always
      - no test files in the diff -> testing, always (combined with
        whatever else applies)

    Returns a list of specialists if a rule fired with full confidence,
    or None if the decision should fall through to the LLM.
    """
    total_changed_lines = sum(
        len(f["additions"]) + len(f["deletions"]) for f in files
    )

    if total_changed_lines < SMALL_DIFF_LINE_THRESHOLD:
        return ["maintainability"]

    specialists: set[str] = set()

    if _touches_dependency_file(files):
        specialists.add("security")

    if not _has_test_files(files):
        specialists.add("testing")

    return sorted(specialists) if specialists else None


def _touches_dependency_file(files: list[FileHunk]) -> bool:
    for f in files:
        filename = f["file_path"].rsplit("/", 1)[-1].lower()
        if filename in DEPENDENCY_FILENAMES:
            return True
    return False


def _has_test_files(files: list[FileHunk]) -> bool:
    for f in files:
        path = f["file_path"].lower()
        filename = path.rsplit("/", 1)[-1]
        if filename.startswith("test_") or filename.endswith("_test.py") or "/tests/" in path or path.startswith("tests/"):
            return True
    return False


# --- Static prompt pieces (order matters for cache hits) ---------------

TRIAGE_SYSTEM_PROMPT = (
    "You are the lead reviewer on an automated code review system. "
    "Your only job right now is to decide which specialist reviewers "
    "should look at this PR. You do not review code yourself, but you "
    "must actually read the added code shown to you to make this decision — "
    "do not judge based on line counts or file names alone. Multiple "
    "specialists are often needed on the same PR — do not assume only "
    "one category of issue can be present."
)

TRIAGE_SCHEMA_PROMPT = (
    "Respond with ONLY a JSON object matching this exact shape, no other text:\n"
    '{"active_specialists": ["security"], "reasoning": "one sentence"}\n'
    "Valid values for active_specialists: security, performance, testing, "
    "maintainability. Include EVERY specialist genuinely relevant to this diff — "
    "it is common and expected for 2-4 to apply at once. Include security if you "
    "see: hardcoded secrets/keys/credentials, raw SQL built with string "
    "formatting/f-strings/concatenation, unsafe deserialization, or missing "
    "auth checks. Include performance if you see: queries or expensive calls "
    "inside loops, blocking sleeps/I/O in a hot path, or unbounded queries. "
    "Include testing if you see: weak assertions, missing edge case coverage, "
    "or logic changes without corresponding test changes. Include "
    "maintainability if you see: functions with many parameters or unrelated "
    "responsibilities, duplicated logic, or magic numbers."
)

TRIAGE_FEW_SHOT = (
    "Example input:\n"
    "File: api.py\n"
    '10: API_KEY = "abc123"\n'
    '11: cursor.execute(f"SELECT * FROM t WHERE id = {x}")\n\n'
    "File: worker.py\n"
    '20: for item in items:\n'
    '21:     time.sleep(0.5)\n'
    '22:     db.query(item.id)\n\n'
    'Example output: {"active_specialists": ["security", "performance"], '
    '"reasoning": "Hardcoded API key and f-string SQL need a security check; '
    'the sleep+query inside a loop needs a performance check."}'
)


def lead_triage(state: ReviewState, gateway: LLMGateway) -> dict:
    """
    LangGraph node. First tries the deterministic activation rules; only
    falls through to an LLM call when the rules don't produce a
    confident answer.

    On a validation failure, retries once with the error appended to the
    prompt; if that also fails, falls back to activating every specialist
    rather than silently reviewing nothing.
    """
    rule_decision = apply_activation_rules(state["files"])
    if rule_decision is not None:
        return {"active_specialists": rule_decision}

    diff_preview = build_file_preview(state["files"])
    variable_content = f"Changed files and their added lines:\n{diff_preview}"

    decision = _call_triage(gateway, variable_content)

    if decision is None:
        return {
            "active_specialists": sorted(VALID_SPECIALISTS),
            "errors": ["Triage failed validation twice — defaulting to all specialists."],
        }

    return {"active_specialists": decision.active_specialists}


def _call_triage(gateway: LLMGateway, variable_content: str, retry_note: str = "") -> TriageDecision | None:
    response = gateway.call(
        task_type=TaskType.ORCHESTRATION,
        system_prompt=TRIAGE_SYSTEM_PROMPT,
        schema_prompt=TRIAGE_SCHEMA_PROMPT,
        few_shot=TRIAGE_FEW_SHOT,
        variable_content=variable_content + retry_note,
        max_tokens=300,
    )

    try:
        raw = json.loads(response["content"])
        decision = TriageDecision(**raw)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        if retry_note:
            return None  # already retried once, give up
        return _call_triage(
            gateway,
            variable_content,
            retry_note=f"\n\nYour previous response was invalid: {e}. Respond with valid JSON only.",
        )

    # Filter out any hallucinated specialist names defensively.
    decision.active_specialists = [
        s for s in decision.active_specialists if s in VALID_SPECIALISTS
    ]
    return decision


# --- Post-processing: plain Python, no LLM -----------------------------

def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """
    Drops exact duplicates — same file, same line, same category. Two
    specialists flagging the identical issue independently is common and
    shouldn't show up twice in the report.
    """
    seen = set()
    deduped = []
    for f in findings:
        key = (f["file"], f["line"], f["category"], f["message"])
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    return deduped


_SEVERITY_ORDER = {"blocker": 0, "major": 1, "minor": 2, "nit": 3}


def rank_findings(findings: list[Finding]) -> list[Finding]:
    """Sorts by severity (blocker first), then file, then line."""
    return sorted(
        findings,
        key=lambda f: (_SEVERITY_ORDER.get(f["severity"], 99), f["file"], f["line"]),
    )


def generate_report(state: ReviewState) -> dict:
    """
    Final LangGraph node. Dedupes, ranks, and produces a plain-text
    summary report. Writes to final_findings (NOT findings) — findings
    uses an operator.add reducer, so writing the cleaned list back to it
    would append rather than replace, doubling every entry.
    """
    deduped = dedupe_findings(state["findings"])
    ranked = rank_findings(deduped)

    counts = {"blocker": 0, "major": 0, "minor": 0, "nit": 0}
    for f in ranked:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1

    lines = [
        f"Review complete: {len(ranked)} findings "
        f"({counts['blocker']} blocker, {counts['major']} major, "
        f"{counts['minor']} minor, {counts['nit']} nit)",
        f"Tokens spent: {state.get('token_spent', 0)}",
        "",
    ]
    for f in ranked:
        lines.append(f"[{f['severity'].upper()}] {f['file']}:{f['line']} — {f['message']}")

    return {
        "final_findings": ranked,
        "report_text": "\n".join(lines),
    }