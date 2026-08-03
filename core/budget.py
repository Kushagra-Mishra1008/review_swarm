"""
TokenBucket: sliding 60-second window rate limiter (TPM + RPM), per model,
plus a persistent daily ledger (TPD) that refuses calls past a soft limit.

This is the only thing standing between the gateway and a 429. Nothing
here talks to Groq — it just tracks numbers and tells the gateway whether
it's safe to proceed, and if not, how long to wait.
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime

from core.config import (
    BUCKET_WINDOW_SECONDS,
    DAILY_LEDGER_SOFT_LIMIT_FRACTION,
    LEDGER_PATH,
    MODEL_LIMITS,
)


class BudgetExceededError(Exception):
    """Raised when a call would breach the daily token budget."""
    pass


@dataclass
class _CallRecord:
    timestamp: float
    tokens: int


class TokenBucket:
    """
    Tracks token and request usage in a sliding 60-second window, per model.
    Before a call: check estimated cost against the window, sleep if short.
    After a call: record actual usage from the response.
    """

    def __init__(self):
        # model -> deque of _CallRecord, oldest first
        self._windows: dict[str, deque[_CallRecord]] = {}

    def _window_for(self, model: str) -> deque[_CallRecord]:
        if model not in self._windows:
            self._windows[model] = deque()
        return self._windows[model]

    def _prune(self, model: str, now: float) -> None:
        window = self._window_for(model)
        cutoff = now - BUCKET_WINDOW_SECONDS
        while window and window[0].timestamp < cutoff:
            window.popleft()

    def _current_usage(self, model: str, now: float) -> tuple[int, int]:
        """Returns (requests_in_window, tokens_in_window)."""
        self._prune(model, now)
        window = self._window_for(model)
        return len(window), sum(r.tokens for r in window)

    def wait_if_needed(self, model: str, estimated_tokens: int) -> float:
        """
        Blocks (sleeps) until there's room in the window for a call of
        `estimated_tokens`. Returns the number of seconds slept (0 if none
        was needed). Loops because the window keeps moving while we sleep.
        """
        limits = MODEL_LIMITS[model]
        total_slept = 0.0

        while True:
            now = time.monotonic()
            req_count, token_count = self._current_usage(model, now)

            requests_ok = req_count < limits.rpm
            tokens_ok = (token_count + estimated_tokens) <= limits.tpm

            if requests_ok and tokens_ok:
                return total_slept

            # Sleep until the oldest record in the window expires, which
            # frees up both a request slot and its tokens.
            window = self._window_for(model)
            if not window:
                # Nothing to wait on but still over budget somehow (e.g.
                # a single request's estimated tokens exceed TPM outright).
                raise BudgetExceededError(
                    f"{model}: estimated {estimated_tokens} tokens exceeds "
                    f"TPM limit of {limits.tpm} even with an empty window."
                )

            oldest = window[0].timestamp
            sleep_for = max(0.0, (oldest + BUCKET_WINDOW_SECONDS) - now)
            # Small buffer so we don't wake up right on the boundary and
            # re-check into the same stale state.
            sleep_for += 0.05
            time.sleep(sleep_for)
            total_slept += sleep_for

    def record_usage(self, model: str, actual_tokens: int) -> None:
        """Record actual usage after a call completes."""
        now = time.monotonic()
        self._window_for(model).append(_CallRecord(timestamp=now, tokens=actual_tokens))


class DailyLedger:
    """
    Persistent JSON ledger of tokens spent per model per calendar day.
    Refuses calls past DAILY_LEDGER_SOFT_LIMIT_FRACTION of TPD.
    """

    def __init__(self, path: str = LEDGER_PATH):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    @staticmethod
    def _today_key() -> str:
        return date.today().isoformat()

    def spent_today(self, model: str) -> int:
        day = self._data.get(self._today_key(), {})
        return day.get(model, 0)

    def check_budget(self, model: str, estimated_tokens: int) -> None:
        """
        Raises BudgetExceededError if this call would push the model's
        daily spend past the soft limit.
        """
        limits = MODEL_LIMITS[model]
        soft_cap = int(limits.tpd * DAILY_LEDGER_SOFT_LIMIT_FRACTION)
        projected = self.spent_today(model) + estimated_tokens

        if projected > soft_cap:
            raise BudgetExceededError(
                f"{model}: projected daily usage {projected} would exceed "
                f"soft cap {soft_cap} ({DAILY_LEDGER_SOFT_LIMIT_FRACTION:.0%} "
                f"of {limits.tpd} TPD). Already spent {self.spent_today(model)} "
                f"today."
            )

    def record(self, model: str, actual_tokens: int) -> None:
        """Record actual token spend against today's ledger entry."""
        day_key = self._today_key()
        day = self._data.setdefault(day_key, {})
        day[model] = day.get(model, 0) + actual_tokens
        self._save()

    def summary(self) -> dict:
        """Today's spend across all models, for logging/debugging."""
        return dict(self._data.get(self._today_key(), {}))