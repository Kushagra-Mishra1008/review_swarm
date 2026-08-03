"""
Static analysis tools: runs ruff and semgrep over changed files and
returns their findings as plain data. Per the plan, this is fed into
specialist/worker prompts as EVIDENCE ("here are N machine-detected
issues, judge which are real and add anything they missed") rather than
reported directly — the tool detects cheaply, the LLM judges.

Both are external CLI tools, not just pip packages with importable APIs
we call directly — we shell out to them, same pattern as ripgrep in
retrieval/search.py.
"""

import json
import subprocess


class StaticFinding(dict):
    """
    Plain dict shape (not a dataclass — this is a lightweight, tool-
    agnostic structure since ruff and semgrep have different native
    formats we normalize into this one):
        {"file": str, "line": int, "tool": str, "rule": str, "message": str}
    """
    pass


def run_ruff(file_paths: list[str], repo_root: str) -> list[StaticFinding]:
    """
    Runs ruff (Python linter) over the given files, returns findings as
    normalized dicts. Ruff is fast enough to run on every relevant file
    without meaningfully slowing the review down.

    Returns an empty list (not an exception) if ruff isn't installed or
    errors — static analysis is a nice-to-have signal, not a hard
    dependency; specialists still work without it, just with less
    pre-computed evidence.
    """
    if not file_paths:
        return []

    try:
        proc = subprocess.run(
            ["ruff", "check", "--output-format=json", *file_paths],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if not proc.stdout:
        return []

    try:
        raw_results = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []

    findings = []
    for item in raw_results:
        findings.append(
            StaticFinding(
                file=item.get("filename", ""),
                line=item.get("location", {}).get("row", 0),
                tool="ruff",
                rule=item.get("code", "unknown"),
                message=item.get("message", ""),
            )
        )
    return findings


def run_semgrep(file_paths: list[str], repo_root: str) -> list[StaticFinding]:
    """
    Runs semgrep with its default auto-detected ruleset over the given
    files. Semgrep is significantly slower than ruff (it's a much
    heavier static analysis pass), so this matters more for the "10-file
    PR under 4 minutes wall clock" gate — worth keeping an eye on timing.

    Same graceful-degradation behavior as run_ruff: missing install or
    any error just means zero semgrep evidence, not a crash.
    """
    if not file_paths:
        return []

    try:
        proc = subprocess.run(
            ["semgrep", "--config=auto", "--json", "--quiet", *file_paths],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if not proc.stdout:
        return []

    try:
        raw_output = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []

    findings = []
    for item in raw_output.get("results", []):
        findings.append(
            StaticFinding(
                file=item.get("path", ""),
                line=item.get("start", {}).get("line", 0),
                tool="semgrep",
                rule=item.get("check_id", "unknown"),
                message=item.get("extra", {}).get("message", ""),
            )
        )
    return findings


def run_static_analysis(file_paths: list[str], repo_root: str) -> dict[str, list[StaticFinding]]:
    """
    Runs both tools and groups results by file path — this is the shape
    specialists/workers actually want: "here's everything the tools
    found in file X."
    """
    ruff_findings = run_ruff(file_paths, repo_root)
    semgrep_findings = run_semgrep(file_paths, repo_root)

    by_file: dict[str, list[StaticFinding]] = {}
    for finding in ruff_findings + semgrep_findings:
        by_file.setdefault(finding["file"], []).append(finding)

    return by_file


def format_evidence_for_prompt(findings: list[StaticFinding]) -> str:
    """
    Turns a file's static findings into a short text block to prepend to
    a worker's prompt. Empty string if there's nothing to show — a
    worker with no static evidence just reasons from the diff alone,
    same as before this feature existed.
    """
    if not findings:
        return ""

    lines = ["Machine-detected issues in this file (verify which are real, add anything missed):"]
    for f in findings:
        lines.append(f"  - line {f['line']} [{f['tool']}/{f['rule']}]: {f['message']}")
    return "\n".join(lines)