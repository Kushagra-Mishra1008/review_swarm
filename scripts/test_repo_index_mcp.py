"""
Standalone verification for the Repo Index MCP server. Starts the
server as a subprocess (exactly how a real MCP client would), connects
over stdio using a raw MCP client, and calls all three tools against
the FastAPI repo already indexed in Phase 1.

This is deliberately independent of LangGraph and langchain-mcp-adapters
— it proves the server itself is correct before any of that plumbing
gets involved. Per the plan: "test it standalone with a raw MCP client
before wiring it in."
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

FASTAPI_LOCAL_PATH = "./data/fastapi"  # same repo indexed in Phase 1


def check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


async def main() -> int:
    all_passed = True

    server_params = StdioServerParameters(
        command=sys.executable,  # use the current venv's python
        args=["-m", "mcp_servers.repo_index.server", "--repo-root", FASTAPI_LOCAL_PATH],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- Check 1: server advertises all three tools ---
            tools_response = await session.list_tools()
            tool_names = {t.name for t in tools_response.tools}
            expected_tools = {"search_code", "find_callers", "get_definition"}
            all_passed &= check(
                f"Server advertises all 3 tools ({tool_names})",
                expected_tools.issubset(tool_names),
            )

            # --- Check 2: get_definition ---
            print("\n--- get_definition('APIRouter') ---")
            result = await session.call_tool("get_definition", {"symbol": "APIRouter"})
            content = result.content[0].text if result.content else ""
            print(f"       {content[:200]}")
            all_passed &= check(
                "get_definition returned non-empty result",
                bool(content) and content != "null",
            )

            # --- Check 3: find_callers ---
            print("\n--- find_callers('jsonable_encoder') ---")
            result = await session.call_tool("find_callers", {"symbol": "jsonable_encoder"})
            content = result.content[0].text if result.content else ""
            print(f"       {content[:200]}")
            all_passed &= check(
                "find_callers returned results",
                bool(content) and content != "[]",
            )

            # --- Check 4: search_code ---
            print("\n--- search_code('how are exceptions handled in middleware') ---")
            result = await session.call_tool(
                "search_code",
                {"query": "how are exceptions handled in middleware", "k": 5},
            )
            content = result.content[0].text if result.content else ""
            print(f"       {content[:200]}")
            all_passed &= check(
                "search_code returned results",
                bool(content) and content != "[]",
            )

    print()
    if all_passed:
        print("Repo Index MCP standalone verification: ALL CHECKS PASSED")
        return 0
    else:
        print("Repo Index MCP standalone verification: SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))