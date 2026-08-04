"""
FastAPI backend for the frontend: starts a review run, streams its
events live over SSE, serves the eval report, and exposes the current
token budget for seeding the frontend's budget bar on page load.

Endpoints:
  POST /review           — kick off a review, returns {run_id}
  GET  /stream/{run_id}  — SSE stream of that run's events
  GET  /eval              — the eval/report.json contents
  GET  /budget              — current cumulative daily spend (seeds the
                               budget bar so a fully-cached run doesn't
                               leave it stuck at 0)
  GET  /health

Also supports demo/replay mode: GET /stream/{run_id}?replay=true reads
a recorded run from disk (backend/events.py's recordings) and streams
it back at original timing, zero tokens, zero network.
"""

import asyncio
import json
import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.graph import run_review
from backend.events import event_bus, load_recording
from core.config import MODEL_120B, MODEL_LIMITS
from core.gateway import LLMGateway

app = FastAPI(title="Code Review Swarm API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EVAL_REPORT_PATH = "eval/report.json"

_gateway = LLMGateway()

_active_runs: dict[str, asyncio.Task] = {}


class ReviewRequest(BaseModel):
    pr_url: str


class ReviewResponse(BaseModel):
    run_id: str


@app.post("/review", response_model=ReviewResponse)
async def start_review(request: ReviewRequest) -> ReviewResponse:
    """
    Kicks off a review as a background task and immediately returns its
    run_id — the frontend then opens an SSE connection to /stream/{run_id}
    to watch it progress.
    """
    run_id = str(uuid.uuid4())

    task = asyncio.create_task(run_review(request.pr_url, gateway=_gateway, run_id=run_id))
    _active_runs[run_id] = task

    return ReviewResponse(run_id=run_id)


@app.get("/stream/{run_id}")
async def stream_events(run_id: str, replay: bool = False):
    """
    SSE stream of a run's events. If replay=true, streams a previously
    recorded run back at its original timing instead of subscribing to
    a live run.
    """
    if replay:
        return StreamingResponse(_replay_generator(run_id), media_type="text/event-stream")
    return StreamingResponse(_live_generator(run_id), media_type="text/event-stream")


async def _live_generator(run_id: str):
    async for event in event_bus.subscribe(run_id):
        yield f"data: {json.dumps(event.to_dict())}\n\n"
    event_bus.cleanup_run(run_id)
    _active_runs.pop(run_id, None)


async def _replay_generator(run_id: str):
    events = load_recording(run_id)
    if not events:
        yield f"data: {json.dumps({'event_type': 'error', 'data': {'message': 'No recording found for this run_id'}})}\n\n"
        return

    prev_timestamp = None
    for event in events:
        if prev_timestamp is not None:
            delay = min(event["timestamp"] - prev_timestamp, 3.0)
            if delay > 0:
                await asyncio.sleep(delay)
        prev_timestamp = event["timestamp"]
        yield f"data: {json.dumps(event)}\n\n"


@app.get("/eval")
async def get_eval_report() -> dict:
    """Returns eval/report.json's contents, or a 404 if Phase 5's
    evaluation hasn't been run yet."""
    if not os.path.exists(EVAL_REPORT_PATH):
        raise HTTPException(status_code=404, detail="No eval report found — run eval/report.py first.")

    with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/budget")
async def get_budget() -> dict:
    """
    Current cumulative daily spend on the 120b model. Used to seed the
    frontend's budget bar on page load — without this, a fully-cached
    run (zero real LLM calls) would leave the bar stuck at 0 even
    though real tokens were spent earlier the same day.
    """
    return {
        "tpd_spent": _gateway._ledger.spent_today(MODEL_120B),
        "tpd_limit": MODEL_LIMITS[MODEL_120B].tpd,
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}