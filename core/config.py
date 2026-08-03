"""
Central configuration for the LLM gateway.
All limits, model names, and thresholds live here — nothing else in the
codebase should hardcode a rate limit or model string.
"""

from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelLimits:
    """Rate limits for a single model, per Groq's free-tier constraints."""
    rpm: int              # requests per minute
    tpm: int               # tokens per minute
    rpd: int                # requests per day
    tpd: int                 # tokens per day


# --- Model identifiers -------------------------------------------------

MODEL_120B = "openai/gpt-oss-120b"
MODEL_20B = "openai/gpt-oss-20b"

# --- Per-model hard limits (Groq free tier, per model, per org) --------

MODEL_LIMITS: dict[str, ModelLimits] = {
    MODEL_120B: ModelLimits(rpm=30, tpm=8_000, rpd=1_000, tpd=200_000),
    MODEL_20B: ModelLimits(rpm=30, tpm=8_000, rpd=1_000, tpd=200_000),
}

# --- Daily ledger safety threshold --------------------------------------
# Refuse calls once a model has used this fraction of its TPD budget.
DAILY_LEDGER_SOFT_LIMIT_FRACTION: float = 0.90

# --- Token estimation ----------------------------------------------------
# Rough chars-per-token ratio used to *estimate* prompt cost before a call.
# Actual usage always comes from the response's `usage` field afterward.
CHARS_PER_TOKEN_ESTIMATE: float = 3.5

# --- Retry policy ----------------------------------------------------------
MAX_RETRIES: int = 3
BACKOFF_BASE_SECONDS: float = 1.0   # fallback exponential backoff base
BACKOFF_MULTIPLIER: float = 2.0

# --- Cache ------------------------------------------------------------------
CACHE_DIR: str = ".cache/llm_gateway"

# --- Budget ledger storage ---------------------------------------------------
LEDGER_PATH: str = ".cache/llm_gateway/daily_ledger.json"

# --- Sliding window for TokenBucket -----------------------------------------
BUCKET_WINDOW_SECONDS: int = 60

# --- Default max_tokens ceiling for structured LLM calls --------------------
# Individual call sites still pass max_tokens explicitly; this is just a
# sane fallback if one is somehow omitted.
DEFAULT_MAX_TOKENS: int = 500