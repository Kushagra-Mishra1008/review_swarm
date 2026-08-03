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

# Max real HTTP calls in flight at once, across ALL agents. LangGraph can
# run 4 specialist nodes "in parallel," but 4 concurrent 2K-token calls
# would be the entire per-minute token budget at once. The token bucket
# already blocks correctly for sync code, but blocking inside
# asyncio.gather serializes badly — this semaphore gives the gateway an
# explicit, visible cap on real concurrency instead.
MAX_CONCURRENT_CALLS = 2


class GatewayError(Exception):
    """Raised for non-retryable gateway failures (e.g. retries exhausted)."""
    pass


def _estimate_tokens(text: str, max_tokens: int) -> int:
    """
    Rough estimate: prompt chars / CHARS_PER_TOKEN_ESTIMATE, plus the
    response's max_tokens ceiling. Good enough for pre-call budgeting —
    actual usage always comes from the response afterward.
    """
    prompt_tokens = int(len(text) / CHARS_PER_TOKEN_ESTIMATE)
    return prompt_tokens + max_tokens


class LLMGateway:
    """
    Usage (sync):
        gateway = LLMGateway()
        response = gateway.call(
            task_type=TaskType.SPECIALIST,
            system_prompt="...",      # static
            schema_prompt="...",       # static
            few_shot="...",             # static
            variable_content="...",      # changes per call — MUST be last
            max_tokens=300,
        )

    Usage (async, for parallel specialist fan-out in Phase 3+):
        response = await gateway.acall(task_type=..., ...)
    """

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
        """
        Synchronous entry point — used by Phase 2 nodes and anywhere
        outside an async graph. No semaphore involved here since sync
        calls are inherently sequential already.
        """
        model = resolve_model(task_type)
        messages, params = self._build_request(
            system_prompt, schema_prompt, few_shot, variable_content, max_tokens, temperature, reasoning_effort
        )

        cached = self._cache.get(model, messages, params)
        if cached is not None:
            return {**cached, "cached": True}

        full_text = system_prompt + schema_prompt + few_shot + variable_content
        estimated = _estimate_tokens(full_text, max_tokens)

        self._ledger.check_budget(model, estimated)
        self._bucket.wait_if_needed(model, estimated)

        response = self._call_with_retry(model, messages, params)

        actual_tokens = response["usage"]["total_tokens"]
        self._bucket.record_usage(model, actual_tokens)
        self._ledger.record(model, actual_tokens)
        self._cache.set(model, messages, params, response)

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
        """
        Async entry point — used when multiple specialists run concurrently
        via asyncio.gather (Phase 3+). Wraps the same logic as call(), but
        the actual network request is gated by self._semaphore so at most
        MAX_CONCURRENT_CALLS real HTTP calls are ever in flight, no matter
        how many specialists "fire" at once in the graph.

        Cache/budget/bucket checks happen OUTSIDE the semaphore — a cache
        hit shouldn't wait in line behind real network calls.
        """
        model = resolve_model(task_type)
        messages, params = self._build_request(
            system_prompt, schema_prompt, few_shot, variable_content, max_tokens, temperature, reasoning_effort
        )

        cached = self._cache.get(model, messages, params)
        if cached is not None:
            return {**cached, "cached": True}

        full_text = system_prompt + schema_prompt + few_shot + variable_content
        estimated = _estimate_tokens(full_text, max_tokens)

        self._ledger.check_budget(model, estimated)

        async with self._semaphore:
            # wait_if_needed is a blocking sleep — run it in a thread so
            # it doesn't block the whole event loop while one call waits
            # out the rate-limit window.
            await asyncio.to_thread(self._bucket.wait_if_needed, model, estimated)
            response = await asyncio.to_thread(self._call_with_retry, model, messages, params)

        actual_tokens = response["usage"]["total_tokens"]
        self._bucket.record_usage(model, actual_tokens)
        self._ledger.record(model, actual_tokens)
        self._cache.set(model, messages, params, response)

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
        """
        Fixed order is mandatory for prompt caching: static content
        first (system, schema, few-shot), variable content last.
        Groq doesn't count cached-prefix tokens against rate limits,
        but only if the prefix is byte-identical across calls.
        """
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
                # Non-429 API error — don't retry, surface immediately.
                raise GatewayError(f"API error from {model}: {e}") from e

        raise GatewayError(f"Exhausted retries calling {model}: {last_error}")

    @staticmethod
    def _parse_retry_after(error: APIStatusError) -> float:
        """
        Prefer the server's retry-after header. Fall back to exponential
        backoff only if the header is missing.
        """
        headers = getattr(error, "response", None)
        if headers is not None:
            retry_after = headers.headers.get("retry-after")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        # Fallback exponential backoff.
        return BACKOFF_BASE_SECONDS * (BACKOFF_MULTIPLIER)