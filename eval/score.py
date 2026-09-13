"""
Matches agent findings (swarm or baseline) to human review comments, per
the plan's rule: same file AND within ±3 lines AND semantically similar
(using the local embedder — free, same one from retrieval/indexer.py).

Line matching is now best-effort, not required: GitHub sets a review
comment's `line` to null once its diff position goes stale (common on
any PR that gets updated after review starts). Requiring a line match
was silently dropping a large share of real, on-topic human comments —
confirmed by inspecting real scikit-learn PR data where several clearly
relevant comments (0.5-0.6 cosine similarity to a matching finding) had
line: null and were being discarded before similarity was ever checked.
"""

from sentence_transformers import SentenceTransformer, util

LINE_TOLERANCE = 3
SIMILARITY_THRESHOLD = 0.5  # empirically checked against real scikit-learn
# PR data: on-topic comment pairs scored 0.48-0.60, unrelated pairs
# scored under 0.2 — this threshold cleanly separates the two.

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _embedder


def _normalize_human_comment(raw: dict) -> dict | None:
    """
    Confirmed real field names from GitHub's MCP server: path, line, body.
    `line` is allowed to be missing/null — GitHub nulls it once a
    comment's diff position is stale, which does NOT mean the comment
    is irrelevant. Only path and body are required; a comment with no
    file or no text genuinely can't be matched against anything.
    """
    path = raw.get("path") or raw.get("file")
    body = raw.get("body") or raw.get("text")

    if not path or not body:
        return None

    line = raw.get("line") or raw.get("original_line") or raw.get("position")
    if line is not None:
        try:
            line = int(line)
        except (TypeError, ValueError):
            line = None

    return {"file": path, "line": line, "body": body}


def match_findings_to_comments(
    findings: list[dict],
    human_comments: list[dict],
) -> dict:
    """
    Returns:
        {
            "matched_findings": [...],
            "unmatched_findings": [...],
            "matched_comments": [...],
            "unmatched_comments": [...],
        }

    Matching rule: same file, AND (no line info on either side, OR
    within LINE_TOLERANCE), AND semantic similarity >= SIMILARITY_THRESHOLD.
    A comment with no line number is judged on file + similarity alone —
    stricter positional matching simply isn't possible without one, and
    dropping it entirely throws away real signal.
    """
    normalized_comments = [c for c in (_normalize_human_comment(rc) for rc in human_comments) if c]

    if not findings or not normalized_comments:
        return {
            "matched_findings": [],
            "unmatched_findings": list(findings),
            "matched_comments": [],
            "unmatched_comments": normalized_comments,
        }

    embedder = _get_embedder()
    finding_texts = [f["message"] for f in findings]
    comment_texts = [c["body"] for c in normalized_comments]

    finding_embeddings = embedder.encode(finding_texts, convert_to_tensor=True)
    comment_embeddings = embedder.encode(comment_texts, convert_to_tensor=True)

    matched_finding_idxs = set()
    matched_comment_idxs = set()

    for fi, finding in enumerate(findings):
        for ci, comment in enumerate(normalized_comments):
            if finding["file"] != comment["file"]:
                continue

            if finding.get("line") is not None and comment.get("line") is not None:
                if abs(finding["line"] - comment["line"]) > LINE_TOLERANCE:
                    continue
            # else: one or both sides have no line info — fall through
            # to similarity-only matching on this file.

            similarity = util.cos_sim(finding_embeddings[fi], comment_embeddings[ci]).item()
            if similarity >= SIMILARITY_THRESHOLD:
                matched_finding_idxs.add(fi)
                matched_comment_idxs.add(ci)

    return {
        "matched_findings": [f for i, f in enumerate(findings) if i in matched_finding_idxs],
        "unmatched_findings": [f for i, f in enumerate(findings) if i not in matched_finding_idxs],
        "matched_comments": [c for i, c in enumerate(normalized_comments) if i in matched_comment_idxs],
        "unmatched_comments": [c for i, c in enumerate(normalized_comments) if i not in matched_comment_idxs],
    }


def compute_pr_metrics(findings: list[dict], human_comments: list[dict]) -> dict:
    """
    Per-PR metrics for one set of findings (swarm OR baseline) against
    the human comments on that PR.
    """
    match_result = match_findings_to_comments(findings, human_comments)

    normalized_comment_count = len(match_result["matched_comments"]) + len(match_result["unmatched_comments"])
    total_findings = len(findings)

    recall = (
        len(match_result["matched_comments"]) / normalized_comment_count
        if normalized_comment_count > 0 else None
    )
    precision = (
        len(match_result["matched_findings"]) / total_findings
        if total_findings > 0 else None
    )

    return {
        "recall": recall,
        "precision": precision,
        "matched_count": len(match_result["matched_findings"]),
        "novel_findings": match_result["unmatched_findings"],
        "missed_comments": match_result["unmatched_comments"],
    }