# Autonomous Code Review Swarm

A hierarchical multi-agent system that reviews GitHub pull requests. Six agents
across three tiers, coordinated by LangGraph, running entirely inside an 8K
tokens-per-minute free-tier budget via a custom gateway.

Evaluated against 19 merged scikit-learn PRs with real human review comments,
and benchmarked against a single-agent baseline to test whether the
architecture earns its complexity.

---

## Architecture

```
                         +--------------+
                         |  Review Lead |   orchestration - 120b
                         +------+-------+
                                | triage: which specialists?
          +-------------+-------+-------+-----------------+
          |             |               |                 |
     +----v----+   +----v-----+   +-----v------+   +------v--------+
     |Security |   | Testing  |   |Performance |   |Maintainability|
     +----+----+   +----+-----+   +-----+------+   +------+--------+
          |             |               |                 |
          +-------------+-------+-------+-----------------+
                                | one worker per selected file
                         +------v-------+
                         | File Scanner |   worker - 20b
                         +------+-------+
                                |
                    dedupe -> rank -> report
```

**Three tiers.** The Lead decides which specialists to wake. Specialists decide
which files are worth scanning. Workers do the narrow per-file inspection on a
cheaper model in a separate rate-limit pool.

**Everything deterministic is plain Python.** Routing, filtering, dedupe,
ranking, and diff parsing never touch an LLM. Static analysis (semgrep, ruff)
runs first and its output is fed to specialists as evidence to judge, not as
findings to trust.

---

## Results

Measured across 19 merged scikit-learn PRs, each with substantive human review
comments. A finding counts as a match when it lands on the same file as a human
comment and the two are semantically similar (local MiniLM embeddings, 0.5
cosine floor).

| Metric | Swarm | Single-agent baseline |
|---|---|---|
| **Precision** | **17.3%** | 9.8% |
| Recall | 2.5% | 6.3% |
| Tokens per review | 4,026 | 1,196 |
| Novel findings surfaced | 42 | — |

**Precision is where the hierarchy pays off.** Roughly one in six swarm findings
corresponds to something a human reviewer independently flagged, against one in
ten for a single undifferentiated pass. Specialist focus hints and per-file
worker scope produce findings that are more often about something that actually
mattered.

**Recall is lower, and the token cost is real.** The swarm recovers fewer human
comments than the baseline while spending about 3.4x the tokens. On a codebase
as dense as scikit-learn, narrow per-file scanning trades breadth for
specificity. Reporting this rather than quietly dropping the baseline is the
point of running one.

**42 novel findings** — issues the swarm raised that no human commented on — are
dumped to `eval/report.json` for manual inspection. A sample includes a function
defined with no body (guaranteed `SyntaxError` on import), a missing closing
paren in a test call, and several unguarded `None` concatenations.

### Caveats

The two precision figures are averaged over different denominators — swarm
precision counts only PRs where it produced findings, baseline precision counts
all of them — so the gap is directional rather than exact. The baseline also
sees each file truncated to 400 characters while the swarm sees full hunks plus
retrieval context, so it is not a like-for-like comparison. n=19, below the
30-PR target. Token figures are approximate: cached gateway responses record
zero usage, so per-review cost is understated on re-runs.

---

## What measuring actually caught

Building the eval harness surfaced three bugs that all reported "no issues
found" rather than erroring — the worst possible failure mode for a review tool,
because a silent pass is indistinguishable from a clean bill of health.

1. **Findings dropped on off-spec severity strings.** Workers discarded any
   finding whose severity didn't exactly match `blocker|major|minor|nit`. The
   20b model returning `"Critical"` silently voided the finding. Now normalized
   against an alias table.

2. **Empty completions from exhausted reasoning budgets.** gpt-oss models emit
   reasoning tokens that never appear in `content`. On large files the worker's
   500-token ceiling was consumed entirely by reasoning, returning an empty
   string that the parser rejected as malformed JSON. The gateway now detects
   empty completions, retries once with a larger budget, and raises a named
   error rather than passing an empty string downstream.

3. **Specialists selecting zero files.** A valid-but-empty file selection
   spawned no workers, producing a zero-finding review that looked identical to
   a clean PR. Empty selections now fall back to scanning everything.

All three passed the synthetic test-bed gates. They only appeared against real
repository code, where files are large enough to exhaust a token budget and
ambiguous enough for a specialist to decline.

---

## Rate limits as a design constraint

The entire system runs on Groq's free tier: **30 RPM, 8K TPM, 200K TPD** on
`gpt-oss-120b`. Every LLM call in the codebase goes through a single
`LLMGateway`. No agent ever touches the Groq client directly.

The gateway owns:

- **Token bucket** — sliding 60-second window tracking tokens and requests per
  model. Estimates before the call, records actual usage from the response after.
- **Daily ledger** — JSON on disk, per model per calendar day, refusing calls
  past 90% of TPD.
- **Disk cache** — sha256 over model + messages + params. Cache hits return
  instantly and record zero usage, which is what makes iterating on this
  survivable.
- **Cache-friendly prompt assembly** — static system prompt, static schema,
  static few-shot, variable content last. Baked into the gateway so agents
  can't get the ordering wrong.
- **Concurrency cap** — `asyncio.Semaphore(2)` alongside the bucket. Specialists
  fan out in parallel in the graph; the gateway quietly meters them so four
  concurrent 2K-token calls never consume an entire minute's budget at once.

Zero 429 errors across the full evaluation run.

---

## MCP

Two servers consumed, one authored.

| Server | Role |
|---|---|
| GitHub MCP (official) | PR fetch, diffs, files, review comments |
| Filesystem MCP (official) | Reading the cloned repo |
| **Repo Index MCP (authored)** | `search_code`, `find_callers`, `get_definition` |

The Repo Index server wraps the retrieval layer: tree-sitter chunking on
function and class boundaries, MiniLM embeddings in ChromaDB, and hybrid search
merging vector similarity with ripgrep exact-match via reciprocal rank fusion.
Pure vector search is poor at exact symbol lookup; the hybrid is meaningfully
better for the cost of an afternoon.

All three bind into LangGraph through `langchain-mcp-adapters`. No direct GitHub
REST calls remain in the codebase.

---

## Stack

**Backend** — Python, LangGraph, FastAPI, Pydantic, ChromaDB, tree-sitter,
sentence-transformers, semgrep, ruff

**Models** — Groq `gpt-oss-120b` (orchestration, specialists) and `gpt-oss-20b`
(workers, formatting)

**Frontend** — React, Vite, server-sent events

---

## Running it

```bash
git clone https://github.com/Kushagra-Mishra1008/review_swarm
cd review_swarm
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:

```
GROQ_API_KEY=your_key
GITHUB_PERSONAL_ACCESS_TOKEN=your_token
```

Docker must be running — the GitHub MCP server runs in a container.

**Verify the gateway** (no agents, minimal tokens):

```bash
python scripts/test_gateway.py
```

**Run the app:**

```bash
uvicorn backend.main:app --port 8000
cd frontend && npm install && npm run dev
```

Paste a PR URL and watch the run. Replay a recorded run with no tokens and no
network:

```
http://localhost:5173/?replay=<run_id>
```

**Reproduce the evaluation:**

```bash
python -m eval.collect --owner scikit-learn --repo scikit-learn --count 30
python -m eval.runner
python -m eval.report
```

---

## Repository layout

```
core/            LLMGateway, token bucket, daily ledger, disk cache, model routing
retrieval/       tree-sitter chunker, ChromaDB indexer, hybrid search
agents/          lead, four specialists, worker, LangGraph wiring, shared state
tools/           diff parsing, static analysis, GitHub MCP client
mcp_servers/     authored Repo Index MCP server
eval/            PR collection, run harness, scoring, reporting
backend/         FastAPI, SSE event bus, run recordings
frontend/        React UI — agent graph, event stream, findings, budget bar
```