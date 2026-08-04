"""
Aggregates every scored PR result (from eval/runner.py) into a single
eval/report.json with recall/precision/novel-findings/tokens-per-review,
comparing the swarm against the single-agent baseline — the plan's
explicit "does the architecture earn itself" comparison.

Per-PR token cost isn't stored directly by runner.py (it only has
cumulative daily ledger totals at time of scoring), so this recovers it
by sorting results by when they were scored (file mtime, since that's
the order runner.py actually processed them in) and diffing consecutive
cumulative readings.
"""

import json
import os
import statistics

from eval.score import compute_pr_metrics

EVAL_RESULTS_DIR = "eval/data/results"
REPORT_PATH = "eval/report.json"


def _load_all_results() -> list[dict]:
    """
    Loads every scored PR result, sorted by file modification time —
    this recovers the order runner.py actually scored them in, which
    is needed to compute per-PR token deltas from the cumulative
    ledger readings runner.py stored.
    """
    if not os.path.exists(EVAL_RESULTS_DIR):
        return []

    paths = [
        os.path.join(EVAL_RESULTS_DIR, f)
        for f in os.listdir(EVAL_RESULTS_DIR)
        if f.endswith(".json")
    ]
    paths.sort(key=os.path.getmtime)

    results = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def _compute_per_pr_token_deltas(results: list[dict]) -> list[dict]:
    """
    Replaces each result's swarm_tokens (currently a cumulative daily
    total, per runner.py's known limitation) with the actual per-PR
    delta — the difference from the previous PR's cumulative reading.
    The first PR scored each day can't have a delta computed (no prior
    reading to subtract), so it's left as-is with a flag.
    """
    adjusted = []
    prev_cumulative = None

    for r in results:
        cumulative = r.get("swarm_tokens", 0)
        if prev_cumulative is not None and cumulative >= prev_cumulative:
            r = {**r, "swarm_tokens_per_pr": cumulative - prev_cumulative}
        else:
            # First PR of a day (or a ledger reset in between) — no
            # reliable delta available.
            r = {**r, "swarm_tokens_per_pr": None}
        prev_cumulative = cumulative
        adjusted.append(r)

    return adjusted


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def generate_report() -> dict:
    """
    Computes and writes eval/report.json. Returns the report dict too,
    for programmatic use (e.g. printing a summary after a batch run).
    """
    results = _load_all_results()
    results = _compute_per_pr_token_deltas(results)

    per_pr_reports = []
    swarm_recalls, swarm_precisions = [], []
    baseline_recalls, baseline_precisions = [], []
    swarm_token_costs, baseline_token_costs = [], []
    all_novel_findings = []

    for r in results:
        human_comments = r.get("human_review_comments", [])

        swarm_metrics = compute_pr_metrics(r.get("swarm_findings", []), human_comments)
        baseline_metrics = compute_pr_metrics(r.get("baseline_findings", []), human_comments)

        swarm_recalls.append(swarm_metrics["recall"])
        swarm_precisions.append(swarm_metrics["precision"])
        baseline_recalls.append(baseline_metrics["recall"])
        baseline_precisions.append(baseline_metrics["precision"])

        if r.get("swarm_tokens_per_pr") is not None:
            swarm_token_costs.append(r["swarm_tokens_per_pr"])
        baseline_token_costs.append(r.get("baseline_tokens", 0))

        for nf in swarm_metrics["novel_findings"]:
            all_novel_findings.append({"pr_number": r["pr_number"], "pr_url": r["pr_url"], **nf})

        per_pr_reports.append({
            "pr_number": r["pr_number"],
            "pr_url": r["pr_url"],
            "swarm_recall": swarm_metrics["recall"],
            "swarm_precision": swarm_metrics["precision"],
            "baseline_recall": baseline_metrics["recall"],
            "baseline_precision": baseline_metrics["precision"],
            "swarm_tokens": r.get("swarm_tokens_per_pr"),
            "baseline_tokens": r.get("baseline_tokens"),
            "human_comment_count": len(human_comments),
        })

    report = {
        "pr_count": len(results),
        "summary": {
            "swarm": {
                "mean_recall": _safe_mean(swarm_recalls),
                "mean_precision": _safe_mean(swarm_precisions),
                "mean_tokens_per_review": _safe_mean(swarm_token_costs),
            },
            "baseline": {
                "mean_recall": _safe_mean(baseline_recalls),
                "mean_precision": _safe_mean(baseline_precisions),
                "mean_tokens_per_review": _safe_mean(baseline_token_costs),
            },
        },
        "novel_findings_for_manual_review": all_novel_findings,
        "per_pr": per_pr_reports,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def print_summary(report: dict) -> None:
    print(f"Evaluated {report['pr_count']} PRs\n")

    swarm = report["summary"]["swarm"]
    baseline = report["summary"]["baseline"]

    def fmt(v):
        return f"{v:.1%}" if v is not None else "N/A"

    def fmt_tokens(v):
        return f"{v:.0f}" if v is not None else "N/A"

    print(f"{'Metric':<25}{'Swarm':<15}{'Baseline':<15}")
    print(f"{'Recall':<25}{fmt(swarm['mean_recall']):<15}{fmt(baseline['mean_recall']):<15}")
    print(f"{'Precision':<25}{fmt(swarm['mean_precision']):<15}{fmt(baseline['mean_precision']):<15}")
    print(f"{'Tokens/review':<25}{fmt_tokens(swarm['mean_tokens_per_review']):<15}{fmt_tokens(baseline['mean_tokens_per_review']):<15}")

    print(f"\nNovel findings for manual review: {len(report['novel_findings_for_manual_review'])}")


if __name__ == "__main__":
    report = generate_report()
    print_summary(report)
    print(f"\nFull report written to {REPORT_PATH}")