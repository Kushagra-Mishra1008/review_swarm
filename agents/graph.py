"""
LangGraph wiring for the full pipeline:

  fetch_pr (via GitHub MCP) -> parse_diff -> clone_for_static_analysis
    -> connect_repo_index_mcp -> run_static_analysis
    -> lead_triage
    -> [security, performance, testing, maintainability] (concurrent, async)
    -> collect -> report

Every node publishes node_start/node_complete events (via
backend/events.py) for the frontend's live agent graph and event
stream, and every finding gets its own event for the findings panel.
run_id is set once here via core/run_context.py's contextvar — nothing
below this file needs to know it exists; core/gateway.py picks it up
automatically for cache_hit/throttle_wait/llm_call events.
"""

import asyncio
import os
import shutil
import sys
import tempfile
import uuid

from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.lead import apply_activation_rules, generate_report, lead_triage
from agents.maintainability import amaintainability_specialist
from agents.performance import aperformance_specialist
from agents.security import asecurity_specialist
from agents.state import ReviewState
from agents.testing import atesting_specialist
from core.gateway import LLMGateway
from core.mcp_client import build_static_mcp_client
from core.run_context import current_run_id
from tools.diff import parse_unified_diff
from tools.github_mcp import fetch_pr_via_mcp, parse_pr_url

ALL_SPECIALISTS = ["security", "performance", "testing", "maintainability"]

SPECIALIST_FNS = {
    "security": asecurity_specialist,
    "performance": aperformance_specialist,
    "testing": atesting_specialist,
    "maintainability": amaintainability_specialist,
}


def _publish_event(event_type: str, data: dict) -> None:
    """Same lazy-import pattern as core/gateway.py — keeps agents/ usable
    standalone for any script that never sets a run_id."""
    run_id = current_run_id.get()
    if run_id is None:
        return
    from backend.events import event_bus
    event_bus.publish(run_id, event_type, data)


def build_graph(gateway: LLMGateway, static_mcp_client: MultiServerMCPClient):

    async def fetch_pr_node(state: ReviewState) -> dict:
        _publish_event("node_start", {"node": "fetch_pr"})
        pr_data = await fetch_pr_via_mcp(state["pr_url"], static_mcp_client)
        files = parse_unified_diff(pr_data["diff"])
        _publish_event("node_complete", {"node": "fetch_pr", "file_count": len(files)})
        return {"files": files, "_pr_data": pr_data}

    async def clone_for_static_analysis_node(state: ReviewState) -> dict:
        _publish_event("node_start", {"node": "clone_for_static_analysis"})

        pr_data = state.get("_pr_data")
        if pr_data is None:
            pr_data = await fetch_pr_via_mcp(state["pr_url"], static_mcp_client)

        metadata = pr_data["metadata"]
        head = metadata.get("head", {})
        head_ref = head.get("ref")
        full_name = head.get("repo", {}).get("full_name")

        if not head_ref or not full_name:
            _publish_event("node_complete", {"node": "clone_for_static_analysis", "error": "unresolved head ref"})
            return {
                "repo_local_path": "",
                "errors": [f"[diag] could not resolve head ref/repo from metadata: {head}"],
            }

        clone_url = f"https://github.com/{full_name}.git"
        local_path = tempfile.mkdtemp(prefix="review_swarm_")

        def _clone():
            import subprocess
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", head_ref, clone_url, local_path],
                capture_output=True, text=True, timeout=60,
            )

        await asyncio.to_thread(_clone)
        _publish_event("node_complete", {"node": "clone_for_static_analysis"})
        return {"repo_local_path": local_path}

    async def connect_repo_index_mcp_node(state: ReviewState) -> dict:
        _publish_event("node_start", {"node": "connect_repo_index_mcp"})

        local_path = state.get("repo_local_path")
        if not local_path or not os.path.exists(local_path):
            _publish_event("node_complete", {"node": "connect_repo_index_mcp", "skipped": True})
            return {"errors": ["[diag] repo_index MCP skipped: no local clone available"]}

        try:
            repo_index_client = MultiServerMCPClient(
                {
                    "repo_index": {
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": ["-m", "mcp_servers.repo_index.server", "--repo-root", local_path],
                    }
                }
            )
            tools = await repo_index_client.get_tools(server_name="repo_index")
            tool_names = sorted(t.name for t in tools)
            _publish_event("node_complete", {"node": "connect_repo_index_mcp", "tools": tool_names})
            return {"errors": [f"[diag] repo_index MCP connected, tools: {tool_names}"]}
        except Exception as e:
            _publish_event("node_complete", {"node": "connect_repo_index_mcp", "error": str(e)})
            return {"errors": [f"[diag] repo_index MCP connection failed (non-fatal): {e}"]}

    async def run_static_analysis_node(state: ReviewState) -> dict:
        from tools.static_analysis import run_static_analysis

        _publish_event("node_start", {"node": "run_static_analysis"})

        local_path = state.get("repo_local_path")
        if not local_path or not os.path.exists(local_path):
            _publish_event("node_complete", {"node": "run_static_analysis", "skipped": True})
            return {"static_findings_by_file": {}}

        file_paths = [f["file_path"] for f in state["files"]]
        results = await asyncio.to_thread(run_static_analysis, file_paths, local_path)
        _publish_event("node_complete", {"node": "run_static_analysis", "files_with_evidence": len(results)})
        return {"static_findings_by_file": results}

    async def lead_triage_node(state: ReviewState) -> dict:
        _publish_event("node_start", {"node": "lead_triage"})
        result = await asyncio.to_thread(lead_triage, state, gateway)
        _publish_event("node_complete", {"node": "lead_triage", "active_specialists": result.get("active_specialists", [])})
        return result

    async def specialists_node(state: ReviewState) -> dict:
        active = state.get("active_specialists", [])
        for name in active:
            _publish_event("node_start", {"node": name})

        tasks = [SPECIALIST_FNS[name](state, gateway) for name in active if name in SPECIALIST_FNS]

        if not tasks:
            return {"findings": []}

        results = await asyncio.gather(*tasks)

        findings = []
        errors = []
        for name, r in zip([n for n in active if n in SPECIALIST_FNS], results):
            specialist_findings = r.get("findings", [])
            findings.extend(specialist_findings)
            errors.extend(r.get("errors", []))
            selected = sorted({f["file"] for f in specialist_findings})
            errors.append(f"[diag] {name} produced findings in: {selected}")

            for f in specialist_findings:
                _publish_event("finding", f)
            _publish_event("node_complete", {"node": name, "finding_count": len(specialist_findings)})

        return {"findings": findings, "errors": errors}

    async def report_node(state: ReviewState) -> dict:
        _publish_event("node_start", {"node": "report"})
        result = generate_report(state)

        local_path = state.get("repo_local_path")
        if local_path and os.path.exists(local_path):
            try:
                shutil.rmtree(local_path, ignore_errors=True)
            except OSError:
                pass

        _publish_event("node_complete", {"node": "report", "final_finding_count": len(result.get("final_findings", []))})
        _publish_event("run_complete", {"report_text": result.get("report_text", "")})
        return result

    graph = StateGraph(ReviewState)

    graph.add_node("fetch_pr", fetch_pr_node)
    graph.add_node("clone_for_static_analysis", clone_for_static_analysis_node)
    graph.add_node("connect_repo_index_mcp", connect_repo_index_mcp_node)
    graph.add_node("run_static_analysis", run_static_analysis_node)
    graph.add_node("lead_triage", lead_triage_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("fetch_pr")
    graph.add_edge("fetch_pr", "clone_for_static_analysis")
    graph.add_edge("clone_for_static_analysis", "connect_repo_index_mcp")
    graph.add_edge("connect_repo_index_mcp", "run_static_analysis")
    graph.add_edge("run_static_analysis", "lead_triage")
    graph.add_edge("lead_triage", "specialists")
    graph.add_edge("specialists", "report")
    graph.add_edge("report", END)

    return graph.compile()


async def run_review(pr_url: str, gateway: LLMGateway | None = None, run_id: str | None = None) -> ReviewState:
    """
    Async entry point. run_id defaults to a fresh UUID if not given —
    the backend passes its own generated run_id so the frontend can
    reference it in the SSE URL before the review even starts.
    """
    run_id = run_id or str(uuid.uuid4())
    current_run_id.set(run_id)

    gateway = gateway or LLMGateway()
    static_mcp_client = build_static_mcp_client()

    app = build_graph(gateway, static_mcp_client)

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