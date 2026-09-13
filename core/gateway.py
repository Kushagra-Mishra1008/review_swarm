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

Empty-content handling: gpt-oss models emit reasoning tokens that do
not appear in choice.message.content. When a generation exhausts
max_tokens before emitting the answer, content comes back as None or
"" with finish_reason="length". Previously that empty string was
returned verbatim, so callers got a confusing JSONDecodeError at
"line 1 column 1" instead of a truncation signal. Now an empty
completion is retried once with a larger max_tokens, and if it is
still empty the gateway raises GatewayError naming finish_reason.

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

# When a completion comes back empty because it ran out of room, retry
# once with this multiplier applied to max_tokens. One retry only — a
# second empty response means something other than truncation.
EMPTY_RETRY_TOKEN_MULTIPLIER = 2.5
EMPTY_RETRY_TOKEN_CEILING = 2000


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

        if self._is_empty(response):
            response = self._retry_empty(model, messages, params, full_text)

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

            if self._is_empty(response):
                response = await asyncio.to_thread(
                    self._retry_empty, model, messages, params, full_text
                )

        self._cache.set(model, messages, params, response)

        _publish("llm_call_complete", {
            "model": model,
            "tokens": actual_tokens,
            "tpd_spent": self._ledger.spent_today(model),
        })

        return {**response, "cached": False}

    @staticmethod
    def _is_empty(response: dict) -> bool:
        content = response.get("content")
        return content is None or not str(content).strip()

    def _retry_empty(self, model: str, messages: list[dict], params: dict, full_text: str) -> dict:
        """
        A completion came back with no content. On gpt-oss this nearly
        always means the reasoning trace consumed the whole max_tokens
        budget before any answer was emitted — finish_reason will be
        "length". Retry once with a bigger ceiling; if it is still empty,
        raise rather than handing an empty string to a JSON parser.
        """
        original_max = params["max_tokens"]
        bigger = min(int(original_max * EMPTY_RETRY_TOKEN_MULTIPLIER), EMPTY_RETRY_TOKEN_CEILING)

        _publish("empty_completion_retry", {
            "model": model,
            "from_max_tokens": original_max,
            "to_max_tokens": bigger,
        })

        if bigger <= original_max:
            raise GatewayError(
                f"Empty completion from {model} at max_tokens={original_max} "
                f"and no headroom left to retry."
            )

        retry_params = {**params, "max_tokens": bigger}
        estimated = _estimate_tokens(full_text, bigger)

        self._ledger.check_budget(model, estimated)
        self._bucket.wait_if_needed(model, estimated)

        response = self._call_with_retry(model, messages, retry_params)

        actual_tokens = response["usage"]["total_tokens"]
        self._bucket.record_usage(model, actual_tokens)
        self._ledger.record(model, actual_tokens)

        if self._is_empty(response):
            raise GatewayError(
                f"Empty completion from {model} twice "
                f"(max_tokens {original_max} then {bigger}, "
                f"finish_reason={response.get('finish_reason')}). "
                f"The model is producing reasoning tokens but no answer."
            )

        return response

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
                    "finish_reason": getattr(choice, "finish_reason", None),
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