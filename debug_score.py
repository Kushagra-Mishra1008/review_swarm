"""
One-off diagnostic: for one scored PR, print every finding alongside
every human comment on the SAME file, with their line numbers and
cosine similarity — so we can see by eye whether matches are being
missed due to line drift, similarity threshold, or something else.

Run from the project root:
    python debug_score.py <pr_number>
"""

import json
import sys

from sentence_transformers import SentenceTransformer, util

RESULTS_DIR = "eval/data/results"


def main():
    pr_number = sys.argv[1] if len(sys.argv) > 1 else None
    if pr_number is None:
        print("Usage: python debug_score.py <pr_number>")
        return

    path = f"{RESULTS_DIR}/{pr_number}.json"
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)

    findings = result.get("swarm_findings", []) + result.get("baseline_findings", [])
    comments = result.get("human_review_comments", [])

    print(f"PR #{pr_number}: {len(findings)} total findings, {len(comments)} human comments\n")

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    for finding in findings:
        f_file = finding.get("file")
        f_line = finding.get("line")
        f_msg = finding.get("message", "")
        print(f"--- FINDING: {f_file}:{f_line} — {f_msg[:80]}")

        same_file_comments = [c for c in comments if c.get("path") == f_file]
        if not same_file_comments:
            print(f"    (no human comments on {f_file} at all)")
            continue

        for c in same_file_comments:
            c_line = c.get("line")
            c_body = c.get("body", "")
            line_diff = abs((f_line or 0) - (c_line or 0)) if c_line is not None else "N/A"

            emb_f = embedder.encode([f_msg], convert_to_tensor=True)
            emb_c = embedder.encode([c_body], convert_to_tensor=True)
            sim = util.cos_sim(emb_f, emb_c).item()

            print(f"    vs comment line {c_line} (Δ{line_diff}) sim={sim:.3f}: {c_body[:80]!r}")
        print()


if __name__ == "__main__":
    main()
