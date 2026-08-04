"""
LLMGateway — the only thing in this codebase allowed to talk to Groq.

Every agent call goes through here. This class owns:
  - prompt assembly in a fixed cache-friendly order
  - the disk cache (check before calling, store after)
  - the token bucket + daily ledger (check/wait before, record after)
  - retries on 429 with retry-after honored
  - routing task type -> model
  - safe concurrency: an asyncio.Semaphore limits how many real HTTP
    calls are in flight at once, independent of the token bucket
  - event publishing: every call publishes cache hits, throttle waits,
    and completions to backend/events.py's event bus. The run_id comes
    from core/run_context.py's contextvar, set once at the top of
    agents/graph.py's run_review() — no function signature in between
    needs to know about it. If no run_id is set (e.g. any standalone
    test script from Phases 0-5), events are silently skipped.

  Cache hits ALSO report tpd_spent (cumulative daily spend), not just
  llm_call_complete — otherwise a fully-cached run (zero real API
  calls) would leave the frontend's budget bar stuck at whatever it
  last saw, since cache_hit used to carry no budget info at all.

If you ever see `client.chat.completions.create(...)` outside this file,
the design is broken.
"""

import asyncio
import os
import time

from groq import Groq
from groq import APIStatusError

from core.budget import BudgetExceededError, DailyLedger, TokenBucket
from core.cache import DiskCache
from core.config import (
    CHARS_PER_TOKEN_ESTIMATE,
    DEFAULT_MAX_TOKENS,
    MAX_RETRIES,
    BACKOFF_BASE_SECONDS,
    BACKOFF_MULTIPLIER,
)
from core.models import TaskType, resolve_model
from core.run_context import current_run_id

MAX_CONCURRENT_CALLS = 2


class GatewayError(Exception):
    """Raised for non-retryable gateway failures (e.g. retries exhausted)."""
    pass


def _estimate_tokens(text: str, max_tokens: int) -> int:
    prompt_tokens = int(len(text) / CHARS_PER_TOKEN_ESTIMATE)
    return prompt_tokens + max_tokens


def _publish(event_type: str, data: dict) -> None:
    """
    Publishes an event only if a run_id is currently set in context.
    Lazy import of backend.events keeps core/ usable standalone for any
    script that never sets a run_id — nothing from Phases 0-5 breaks.
    """
    run_id = current_run_id.get()
    if run_id is None:
        return
    from backend.events import event_bus
    event_bus.publish(run_id, event_type, data)


class LLMGateway:
    def __init__(self, api_key: str | None = None):
        self._client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self._bucket = TokenBucket()
        self._ledger = DailyLedger()
        self._cache = DiskCache()
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    def call(
        self,
        task_type: TaskType | str,
        system_prompt: str,
        schema_prompt: str,
        few_shot: str,
        variable_content: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        reasoning_effort: str = "low",
    ) -> dict:
        model = resolve_model(task_type)
        messages, params = self._build_request(
            system_prompt, schema_prompt, few_shot, variable_content, max_tokens, temperature, reasoning_effort
        )

        cached = self._cache.get(model, messages, params)
        if cached is not None:
            _publish("cache_hit", {"model": model, "tpd_spent": self._ledger.spent_today(model)})
            return {**cached, "cached": True}

        full_text = system_prompt + schema_prompt + few_shot + variable_content
        estimated = _estimate_tokens(full_text, max_tokens)

        self._ledger.check_budget(model, estimated)

        _publish("llm_call_start", {"model": model})
        wait_start = time.monotonic()
        self._bucket.wait_if_needed(model, estimated)
        wait_elapsed = time.monotonic() - wait_start
        if wait_elapsed > 0.5:
            _publish("throttle_wait", {"model": model, "seconds": round(wait_elapsed, 1)})

        response = self._call_with_retry(model, messages, params)

        actual_tokens = response["usage"]["total_tokens"]
        self._bucket.record_usage(model, actual_tokens)
        self._ledger.record(model, actual_tokens)
        self._cache.set(model, messages, params, response)

        _publish("llm_call_complete", {
            "model": model,
            "tokens": actual_tokens,
            "tpd_spent": self._ledger.spent_today(model),
        })

        return {**response, "cached": False}

    async def acall(
        self,
        task_type: TaskType | str,
        system_prompt: str,
        schema_prompt: str,
        few_shot: str,
        variable_content: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        reasoning_effort: str = "low",
    ) -> dict:
        model = resolve_model(task_type)
        messages, params = self._build_request(
            system_prompt, schema_prompt, few_shot, variable_content, max_tokens, temperature, reasoning_effort
        )

        cached = self._cache.get(model, messages, params)
        if cached is not None:
            _publish("cache_hit", {"model": model, "tpd_spent": self._ledger.spent_today(model)})
            return {**cached, "cached": True}

        full_text = system_prompt + schema_prompt + few_shot + variable_content
        estimated = _estimate_tokens(full_text, max_tokens)

        self._ledger.check_budget(model, estimated)

        _publish("llm_call_start", {"model": model})

        async with self._semaphore:
            wait_start = time.monotonic()
            await asyncio.to_thread(self._bucket.wait_if_needed, model, estimated)
            wait_elapsed = time.monotonic() - wait_start
            if wait_elapsed > 0.5:
                _publish("throttle_wait", {"model": model, "seconds": round(wait_elapsed, 1)})

            response = await asyncio.to_thread(self._call_with_retry, model, messages, params)

        actual_tokens = response["usage"]["total_tokens"]
        self._bucket.record_usage(model, actual_tokens)
        self._ledger.record(model, actual_tokens)
        self._cache.set(model, messages, params, response)

        _publish("llm_call_complete", {
            "model": model,
            "tokens": actual_tokens,
            "tpd_spent": self._ledger.spent_today(model),
        })

        return {**response, "cached": False}

    @staticmethod
    def _build_request(
        system_prompt: str,
        schema_prompt: str,
        few_shot: str,
        variable_content: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str,
    ) -> tuple[list[dict], dict]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": schema_prompt},
            {"role": "system", "content": few_shot},
            {"role": "user", "content": variable_content},
        ]
        params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
        }
        return messages, params

    def _call_with_retry(self, model: str, messages: list[dict], params: dict) -> dict:
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                completion = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    reasoning_effort=params.get("reasoning_effort", "low"),
                )
                choice = completion.choices[0]
                return {
                    "content": choice.message.content,
                    "usage": {
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "total_tokens": completion.usage.total_tokens,
                    },
                }

            except APIStatusError as e:
                last_error = e
                if e.status_code == 429:
                    retry_after = self._parse_retry_after(e)
                    if attempt < MAX_RETRIES:
                        time.sleep(retry_after)
                        continue
                    raise GatewayError(
                        f"429 from {model} after {MAX_RETRIES} retries: {e}"
                    ) from e
                raise GatewayError(f"API error from {model}: {e}") from e

        raise GatewayError(f"Exhausted retries calling {model}: {last_error}")

    @staticmethod
    def _parse_retry_after(error: APIStatusError) -> float:
        headers = getattr(error, "response", None)
        if headers is not None:
            retry_after = headers.headers.get("retry-after")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        return BACKOFF_BASE_SECONDS * (BACKOFF_MULTIPLIER)