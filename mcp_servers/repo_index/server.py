"""
Repo Index MCP server — exposes retrieval/search.py's hybrid search as
MCP tools, over stdio transport, so any MCP client (including our own
LangGraph agents via langchain-mcp-adapters) can call search_code,
find_callers, and get_definition without importing retrieval/ directly.

Run standalone for testing:
    python -m mcp_servers.repo_index.server --repo-root ./data/fastapi

This is the "author your own MCP server" piece the plan calls out as the
one worth building yourself — consuming an MCP is unremarkable, authoring
one demonstrates understanding of the protocol.
"""

import argparse

from mcp.server.fastmcp import FastMCP

from mcp_servers.repo_index.tools import (
    find_callers_tool,
    get_definition_tool,
    init_search,
    search_code_tool,
)

mcp = FastMCP("repo-index")


@mcp.tool()
def search_code(query: str, k: int = 5) -> list[dict]:
    """
    Search the indexed repository for code relevant to a natural-language
    or symbol query, using hybrid vector + exact-match search.

    Args:
        query: what to search for, e.g. "how are exceptions handled" or a symbol name
        k: max number of results to return

    Returns a list of chunks, each with file_path, symbol_name, kind,
    start_line, end_line, source, and a relevance score.
    """
    return search_code_tool(query, k=k)


@mcp.tool()
def find_callers(symbol: str) -> list[dict]:
    """
    Find call sites of a given symbol (function or class name) across
    the indexed repository.

    Args:
        symbol: the exact symbol name to search for callers of

    Returns a list of chunks where the symbol is referenced, excluding
    (best-effort) the definition itself.
    """
    return find_callers_tool(symbol)


@mcp.tool()
def get_definition(symbol: str) -> dict | None:
    """
    Get the exact definition of a symbol (function or class) in the
    indexed repository.

    Args:
        symbol: the exact symbol name to look up

    Returns the chunk containing the definition, or null if not found.
    """
    return get_definition_tool(symbol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo Index MCP server")
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Local path to the repository to index and search over",
    )
    args = parser.parse_args()

    init_search(args.repo_root)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()