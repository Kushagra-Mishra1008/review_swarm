"""
Central MCP client configuration: wires up the two "static" MCP servers
(GitHub, Filesystem) that don't depend on any per-run state. The
Repo Index server is NOT here — it needs a repo_root known only after
a PR is cloned, so it's connected dynamically inside graph.py once that
path exists.
"""

import os
import tempfile

from langchain_mcp_adapters.client import MultiServerMCPClient


def build_static_mcp_client() -> MultiServerMCPClient:
    """
    Returns a client connected to:
      - github: the official GitHub MCP server, run via Docker
      - filesystem: the official Filesystem MCP server, rooted at the
        system temp dir (broad enough to cover every PR clone, which
        all live under tempfile.mkdtemp() with a 'review_swarm_' prefix)
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise RuntimeError("GITHUB_TOKEN must be set to use the GitHub MCP server.")

    return MultiServerMCPClient(
        {
            "github": {
                "transport": "stdio",
                "command": "docker",
                "args": [
                    "run", "-i", "--rm",
                    "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                    "ghcr.io/github/github-mcp-server",
                ],
                # Passed to the `docker` CLI process itself; docker then
                # forwards GITHUB_PERSONAL_ACCESS_TOKEN into the container
                # because the -e flag above names it without a value.
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
            },
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y", "@modelcontextprotocol/server-filesystem",
                    tempfile.gettempdir(),
                ],
            },
        }
    )