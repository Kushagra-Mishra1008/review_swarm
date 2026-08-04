"""
Carries the current review's run_id implicitly through async call
chains, so core/gateway.py can publish events without every specialist,
worker, and lead function needing an explicit run_id parameter threaded
through their signatures.

contextvars propagate correctly through asyncio.gather — each spawned
Task inherits the context active at the moment it was created, so once
run_review() sets this at the top of a run, every specialist and worker
call nested underneath it (however deep, however parallel) sees the
same run_id automatically.
"""

import contextvars

current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_run_id", default=None
)