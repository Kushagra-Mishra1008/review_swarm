"""
Model routing table: maps a task type to the model that should handle it.

Agents never choose a model string themselves — they declare a task type
("orchestration", "specialist", "worker", "formatting") and the gateway
looks up which model that maps to. This keeps model choice a config-level
decision, not something scattered across agent code.
"""

from enum import Enum

from core.config import MODEL_120B, MODEL_20B


class TaskType(str, Enum):
    ORCHESTRATION = "orchestration"   # lead agent — needs judgment
    SPECIALIST = "specialist"          # security/perf/testing/maintainability — needs judgment
    WORKER = "worker"                   # per-file scanner — narrow task, separate rate-limit pool
    FORMATTING = "formatting"            # mechanical text shaping


# Task type -> model. This is the single source of truth for routing.
TASK_MODEL_ROUTING: dict[TaskType, str] = {
    TaskType.ORCHESTRATION: MODEL_120B,
    TaskType.SPECIALIST: MODEL_120B,
    TaskType.WORKER: MODEL_20B,
    TaskType.FORMATTING: MODEL_20B,
}


def resolve_model(task_type: TaskType | str) -> str:
    """
    Resolve a task type (enum or raw string) to a concrete model name.
    Raises ValueError on an unknown task type rather than silently
    defaulting — a silent default is how you accidentally burn 120b
    budget on formatting calls.
    """
    if isinstance(task_type, str):
        try:
            task_type = TaskType(task_type)
        except ValueError as e:
            raise ValueError(
                f"Unknown task type: {task_type!r}. "
                f"Valid options: {[t.value for t in TaskType]}"
            ) from e

    return TASK_MODEL_ROUTING[task_type]