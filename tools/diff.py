"""
Parses a PR's raw unified diff into per-file hunks (FileHunk objects).

Pure Python, no LLM involved — this is exactly the kind of deterministic
work the plan says should never be an agent call. Takes the raw diff text
GitHub gives us and turns it into structured data agents.py can reason
about.
"""

import re

from agents.state import FileHunk

# Matches a diff file header, e.g.:
#   diff --git a/main.py b/main.py
FILE_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")

# Matches a hunk header, e.g.:
#   @@ -12,7 +12,9 @@ def get_user(user_id: int):
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

# Per-file cap on how many added-line characters get shown to an LLM in
# a preview. Keeps a huge PR from blowing a call's token budget while
# still giving real signal instead of just a file name or line count.
MAX_CHARS_PER_FILE_PREVIEW = 500


def parse_unified_diff(diff_text: str) -> list[FileHunk]:
    """
    Parses raw unified diff text (as returned by GitHub's PR diff API)
    into a list of FileHunk dicts — one per file touched by the PR.

    Each FileHunk's additions/deletions are (line_number, line_text)
    tuples using the *new* file's line numbers for additions and the
    *old* file's line numbers for deletions, which is what a reviewer
    actually wants to point at.
    """
    files: list[FileHunk] = []
    current_file: FileHunk | None = None

    new_line_num = 0
    old_line_num = 0

    for line in diff_text.splitlines():
        file_match = FILE_HEADER_RE.match(line)
        if file_match:
            if current_file is not None:
                files.append(current_file)
            current_file = FileHunk(
                file_path=file_match.group(2),
                additions=[],
                deletions=[],
                full_content=None,
            )
            continue

        if current_file is None:
            continue  # skip anything before the first file header

        hunk_match = HUNK_HEADER_RE.match(line)
        if hunk_match:
            old_line_num = int(hunk_match.group(1))
            new_line_num = int(hunk_match.group(2))
            continue

        if line.startswith("+") and not line.startswith("+++"):
            current_file["additions"].append((new_line_num, line[1:]))
            new_line_num += 1
        elif line.startswith("-") and not line.startswith("---"):
            current_file["deletions"].append((old_line_num, line[1:]))
            old_line_num += 1
        elif line.startswith(" "):
            # Context line — present in both old and new, advance both counters.
            new_line_num += 1
            old_line_num += 1
        # Lines like "\ No newline at end of file" are ignored entirely.

    if current_file is not None:
        files.append(current_file)

    return files


def summarize_diff(files: list[FileHunk]) -> str:
    """
    Human/LLM-readable one-line-per-file summary (file name + line
    counts only, no code). Used where a cheap overview is genuinely
    enough — NOT for any prompt that decides which files matter, since
    that decision needs real content (see build_file_preview).
    """
    lines = []
    for f in files:
        add_count = len(f["additions"])
        del_count = len(f["deletions"])
        lines.append(f"{f['file_path']}: +{add_count} -{del_count}")
    return "\n".join(lines)


def build_file_preview(files: list[FileHunk], max_chars_per_file: int = MAX_CHARS_PER_FILE_PREVIEW) -> str:
    """
    Builds a per-file preview of actual added lines (not just names or
    counts), truncated per file so a huge PR doesn't blow a call's token
    budget. This is the shared building block for any prompt that needs
    to make a real judgment about file content — lead triage, and every
    specialist's file-selection call. A filename alone is not enough
    signal for either of those decisions.
    """
    parts = []
    for f in files:
        added_text = "\n".join(f"{ln}: {text}" for ln, text in f["additions"])
        if len(added_text) > max_chars_per_file:
            added_text = added_text[:max_chars_per_file] + "\n... (truncated)"
        parts.append(f"File: {f['file_path']}\n{added_text}" if added_text else f"File: {f['file_path']} (no additions)")
    return "\n\n".join(parts)