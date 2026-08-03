"""
Thin wrappers over retrieval/search.py's HybridSearch, shaped as plain
functions with JSON-serializable inputs/outputs — this is the layer
server.py exposes as MCP tools. Kept separate from server.py so these
functions are testable directly with a plain Python call, without
spinning up an MCP server at all.
"""

from retrieval.indexer import CodeIndexer
from retrieval.search import HybridSearch, SearchResult

# Module-level singletons: the indexer/search objects are expensive to
# construct (they load the embedding model), so we build them once per
# server process, not per tool call.
_indexer: CodeIndexer | None = None
_search: HybridSearch | None = None
_repo_root: str | None = None


def init_search(repo_root: str) -> None:
    """Must be called once before any tool function, with the local path
    of the repo to search over (e.g. the cloned PR branch)."""
    global _indexer, _search, _repo_root
    _indexer = CodeIndexer()
    _search = HybridSearch(_indexer, repo_root=repo_root)
    _repo_root = repo_root


def _require_initialized() -> HybridSearch:
    if _search is None:
        raise RuntimeError("init_search() must be called before using repo_index tools.")
    return _search


def _result_to_dict(result: SearchResult) -> dict:
    return {
        "file_path": result.file_path,
        "symbol_name": result.symbol_name,
        "kind": result.kind,
        "start_line": result.start_line,
        "end_line": result.end_line,
        "source": result.source,
        "score": result.score,
    }


def search_code_tool(query: str, k: int = 5) -> list[dict]:
    """Semantic + exact-match hybrid search over the indexed repo."""
    search = _require_initialized()
    results = search.search_code(query, k=k)
    return [_result_to_dict(r) for r in results]


def find_callers_tool(symbol: str) -> list[dict]:
    """Exact-match search for call sites of a given symbol."""
    search = _require_initialized()
    results = search.find_callers(symbol)
    return [_result_to_dict(r) for r in results]


def get_definition_tool(symbol: str) -> dict | None:
    """Exact metadata lookup for a symbol's definition."""
    search = _require_initialized()
    result = search.get_definition(symbol)
    return _result_to_dict(result) if result else None