"""
Disk cache for LLM calls. Keyed by sha256(model + messages + params).

On a hit: return the cached response instantly, record zero usage against
the rate limit or daily ledger. This is what makes development survivable —
you will re-run the same prompt fifty times while iterating on an agent.
"""

import hashlib
import json
import os
from typing import Any

from core.config import CACHE_DIR


def _cache_key(model: str, messages: list[dict], params: dict) -> str:
    """
    Build a stable hash from the exact request. Key order matters for
    hashing, so we sort everything with json.dumps(sort_keys=True) rather
    than relying on dict insertion order.
    """
    payload = {
        "model": model,
        "messages": messages,
        "params": params,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DiskCache:
    """
    Simple JSON-on-disk cache. One file per cache key, named by hash.
    Not thread-safe beyond what the filesystem gives you for free — fine
    for a single-process dev/eval workload.
    """

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path_for_key(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, model: str, messages: list[dict], params: dict) -> dict[str, Any] | None:
        """
        Returns the cached response dict if present, else None.
        Corrupt cache entries are treated as a miss rather than raising —
        a bad cache file should never take down a run.
        """
        key = _cache_key(model, messages, params)
        path = self._path_for_key(key)

        if not os.path.exists(path):
            return None

        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, model: str, messages: list[dict], params: dict, response: dict[str, Any]) -> None:
        """Store a response under the hash of its exact request."""
        key = _cache_key(model, messages, params)
        path = self._path_for_key(key)

        with open(path, "w") as f:
            json.dump(response, f, indent=2)

    def key_for(self, model: str, messages: list[dict], params: dict) -> str:
        """Exposed for logging — lets callers report cache hits by key prefix."""
        return _cache_key(model, messages, params)