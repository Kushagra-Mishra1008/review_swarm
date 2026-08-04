"""
Event bus: a simple in-memory pub/sub that the gateway and graph nodes
publish to directly, and that the SSE endpoint in main.py subscribes to
per-run. This is what makes the frontend's live event stream possible
without agents needing to remember to instrument themselves — the
gateway publishes cache hits, throttle waits, and call completions
automatically on every call.

Also handles recording: every event published for a run_id is appended
to a JSON-lines file on disk, so a completed run can be replayed later
via ?replay=<run_id> with zero tokens and zero network calls.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict

RECORDINGS_DIR = "backend/data/recordings"


@dataclass
class Event:
    run_id: str
    event_type: str          # e.g. "node_start", "llm_call", "cache_hit", "throttle_wait", "finding", "node_complete"
    timestamp: float
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class EventBus:
    """
    One EventBus instance lives for the lifetime of the backend process.
    Each run_id gets its own asyncio.Queue so multiple concurrent
    reviews (multiple SSE clients) don't cross-talk.
    """

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

    def _queue_for(self, run_id: str) -> asyncio.Queue:
        if run_id not in self._queues:
            self._queues[run_id] = asyncio.Queue()
        return self._queues[run_id]

    def publish(self, run_id: str, event_type: str, data: dict | None = None) -> None:
        """
        Synchronous publish — safe to call from anywhere, including
        non-async code inside core/gateway.py. Puts the event on the
        run's queue (non-blocking, since asyncio.Queue.put_nowait
        doesn't need an event loop) and appends it to the recording file.
        """
        event = Event(run_id=run_id, event_type=event_type, timestamp=time.time(), data=data or {})

        queue = self._queue_for(run_id)
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # shouldn't happen (unbounded queue), but never block a review on this

        self._append_to_recording(event)

    async def subscribe(self, run_id: str):
        """
        Async generator: yields events for a run_id as they arrive.
        Used by the SSE endpoint. Stops yielding once a "run_complete"
        event is seen, then the HTTP handler closes the stream.
        """
        queue = self._queue_for(run_id)
        while True:
            event = await queue.get()
            yield event
            if event.event_type == "run_complete":
                break

    def _append_to_recording(self, event: Event) -> None:
        path = os.path.join(RECORDINGS_DIR, f"{event.run_id}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def cleanup_run(self, run_id: str) -> None:
        """Drops the in-memory queue once a run's SSE stream has closed
        and been fully consumed — the recording file on disk persists."""
        self._queues.pop(run_id, None)


def load_recording(run_id: str) -> list[dict]:
    """
    Loads a completed run's recorded events from disk, for replay mode.
    Returns an empty list if no recording exists for that run_id.
    """
    path = os.path.join(RECORDINGS_DIR, f"{run_id}.jsonl")
    if not os.path.exists(path):
        return []

    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# Module-level singleton — imported directly by core/gateway.py and
# agents/graph.py so they don't need the bus threaded through every
# function signature.
event_bus = EventBus()