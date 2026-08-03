"""
LangGraph wiring for the full Phase 3 pipeline:

  fetch_pr -> parse_diff -> clone_for_static_analysis -> run_static_analysis
    -> lead_triage
    -> [security, performance, testing, maintainability] (concurrent, async)
    -> collect -> report

Static analysis and the four specialists all run as async nodes so
LangGraph can invoke them concurrently via ainvoke — the actual HTTP
concurrency cap lives in the gateway's semaphore (core/gateway.py),
not here. This file's only job is wiring nodes together correctly.
"""

import asyncio
import os
import shutil
import tempfile

from langgraph.graph import StateGraph, END

from agents.lead import apply_activation_rules, generate_report, lead_triage
from agents.maintainability import amaintainability_specialist
from agents.performance import aperformance_specialist
from agents.security import asecurity_specialist
from agents.state import ReviewState
from agents.testing import atesting_specialist
from core.gateway import LLMGateway
from tools.diff import parse_unified_diff
from tools.github import GitHubClient, fetch_pr, parse_pr_url
from tools.static_analysis import run_static_analysis

# Specialists that always run their node function, but individually
# no-op (return zero findings) if they weren't activated by the lead.
# Kept as a fixed list so the graph shape never changes at runtime —
# only the SPECIALIST_FNS dict below decides who actually does work.
ALL_SPECIALISTS = ["security", "performance", "testing", "maintainability"]

SPECIALIST_FNS = {
    "security": asecurity_specialist,
    "performance": aperformance_specialist,
    "testing": atesting_specialist,
    "maintainability": amaintainability_specialist,
}


def build_graph(gateway: LLMGateway):
    """
    Returns a compiled LangGraph app, built for async execution
    (invoked via app.ainvoke, not app.invoke).
    """

    async def fetch_pr_node(state: ReviewState) -> dict:
        pr_data = await asyncio.to_thread(fetch_pr, state["pr_url"])
        files = parse_unified_diff(pr_data["diff"])
        return {"files": files, "_pr_data": pr_data}

    async def clone_for_static_analysis_node(state: ReviewState) -> dict:
        """
        Clones the PR's head branch to a temp local path so ruff/semgrep
        have real files on disk to scan — they're CLI tools, not diff
        parsers, they need the actual repo checked out.
        """
        pr_data = state.get("_pr_data")
        if pr_data is None:
            # _pr_data isn't declared in ReviewState (internal-only,
            # not review data) — LangGraph still lets nodes stash extra
            # keys in the dict, they just won't be typed. Re-fetch
            # defensively if somehow missing.
            owner, repo, pr_number = parse_pr_url(state["pr_url"])
            client = GitHubClient()
            pr_data = {
                "owner": owner, "repo": repo, "pr_number": pr_number,
                "metadata": client.get_pr_metadata(owner, repo, pr_number),
            }

        metadata = pr_data["metadata"]
        clone_url = metadata["head"]["repo"]["clone_url"]
        head_ref = metadata["head"]["ref"]

        local_path = tempfile.mkdtemp(prefix="review_swarm_")

        def _clone():
            import subprocess
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", head_ref, clone_url, local_path],
                capture_output=True, text=True, timeout=60,
            )

        await asyncio.to_thread(_clone)
        return {"repo_local_path": local_path}

    async def run_static_analysis_node(state: ReviewState) -> dict:
        """
        Runs ruff + semgrep on every changed file, grouped by file path.
        Skipped gracefully (empty dict) if the clone step failed — a
        missing local checkout means no static evidence, not a crash.
        """
        local_path = state.get("repo_local_path")
        if not local_path or not os.path.exists(local_path):
            return {"static_findings_by_file": {}}

        file_paths = [f["file_path"] for f in state["files"]]
        results = await asyncio.to_thread(run_static_analysis, file_paths, local_path)
        return {"static_findings_by_file": results}

    async def lead_triage_node(state: ReviewState) -> dict:
        # apply_activation_rules is sync/cheap; lead_triage's LLM fallback
        # uses the sync gateway.call still (Phase 2 behavior, unchanged) —
        # triage is a single call, not worth the async plumbing.
        return await asyncio.to_thread(lead_triage, state, gateway)

    async def specialists_node(state: ReviewState) -> dict:
        """
        Runs all activated specialists CONCURRENTLY via asyncio.gather.
        Real HTTP concurrency is still capped by the gateway's semaphore
        (MAX_CONCURRENT_CALLS=2) regardless of how many specialists are
        active here.
        """
        active = state.get("active_specialists", [])
        tasks = [SPECIALIST_FNS[name](state, gateway) for name in active if name in SPECIALIST_FNS]

        if not tasks:
            return {"findings": []}

        results = await asyncio.gather(*tasks)

        findings = []
        errors = []
        for name, r in zip([n for n in active if n in SPECIALIST_FNS], results):
            findings.extend(r.get("findings", []))
            errors.extend(r.get("errors", []))  # forward each specialist's own diagnostics/errors
            selected = sorted({f["file"] for f in r.get("findings", [])})
            errors.append(f"[diag] {name} produced findings in: {selected}")

        return {"findings": findings, "errors": errors}
    
    async def report_node(state: ReviewState) -> dict:
        result = generate_report(state)

        # Best-effort cleanup of the temp clone — not critical to the
        # review outcome, so failures here are swallowed.
        local_path = state.get("repo_local_path")
        if local_path and os.path.exists(local_path):
            try:
                shutil.rmtree(local_path, ignore_errors=True)
            except OSError:
                pass

        return result

    graph = StateGraph(ReviewState)

    graph.add_node("fetch_pr", fetch_pr_node)
    graph.add_node("clone_for_static_analysis", clone_for_static_analysis_node)
    graph.add_node("run_static_analysis", run_static_analysis_node)
    graph.add_node("lead_triage", lead_triage_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("fetch_pr")
    graph.add_edge("fetch_pr", "clone_for_static_analysis")
    graph.add_edge("clone_for_static_analysis", "run_static_analysis")
    graph.add_edge("run_static_analysis", "lead_triage")
    graph.add_edge("lead_triage", "specialists")
    graph.add_edge("specialists", "report")
    graph.add_edge("report", END)

    return graph.compile()


async def run_review(pr_url: str, gateway: LLMGateway | None = None) -> ReviewState:
    """
    Async entry point: run the full graph against a PR URL, return the
    final state. Callers (like scripts/test_phase3.py) must run this
    inside asyncio.run() since the graph is now fully async.
    """
    gateway = gateway or LLMGateway()
    app = build_graph(gateway)

    initial_state: ReviewState = {
        "pr_url": pr_url,
        "files": [],
        "repo_local_path": "",
        "static_findings_by_file": {},
        "active_specialists": [],
        "findings": [],
        "final_findings": [],
        "report_text": "",
        "token_spent": 0,
        "errors": [],
    }

    return await app.ainvoke(initial_state)