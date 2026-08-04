"""
Matches agent findings (swarm or baseline) to human review comments, per
the plan's rule: same file AND within ±3 lines AND semantically similar
(using the local embedder — free, same one from retrieval/indexer.py).

A "match" means the agent caught something a human reviewer also flagged.
Findings that don't match anything are either false positives (noise) or
genuinely novel catches humans missed — score.py doesn't judge which;
report.py surfaces novel findings for manual inspection, per the plan.
"""

from sentence_transformers import SentenceTransformer, util

LINE_TOLERANCE = 3
SIMILARITY_THRESHOLD = 0.5  # cosine similarity floor to count as "about the same issue"

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _embedder


def _normalize_human_comment(raw: dict) -> dict | None:
    """
    GitHub MCP review comment shape isn't fully confirmed (same caveat
    as collect.py — untested against a real run yet). Tries a few
    plausible field names for path/line/body; returns None for comments
    that don't have enough info to match against (e.g. a general PR
    comment with no file/line attached).
    """
    path = raw.get("path") or raw.get("file")
    line = raw.get("line") or raw.get("original_line") or raw.get("position")
    body = raw.get("body") or raw.get("text")

    if not path or not line or not body:
        return None

    try:
        line = int(line)
    except (TypeError, ValueError):
        return None

    return {"file": path, "line": line, "body": body}


def match_findings_to_comments(
    findings: list[dict],
    human_comments: list[dict],
) -> dict:
    """
    Returns:
        {
            "matched_findings": [...],    # findings that matched a human comment
            "unmatched_findings": [...],   # findings with no human match (novel or noise)
            "matched_comments": [...],      # human comments that got caught by a finding
            "unmatched_comments": [...],     # human comments no finding caught (recall gap)
        }
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
            if abs(finding["line"] - comment["line"]) > LINE_TOLERANCE:
                continue

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

    recall = fraction of human comments the findings caught
    precision = fraction of findings that matched a real human comment
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
        "novel_findings": match_result["unmatched_findings"],  # unmatched = novel or noise, human-inspect
        "missed_comments": match_result["unmatched_comments"],
    }