"""Entry point for running the MCP server via `python -m mcp_research_agent_system.mcp_server`."""

import asyncio
import sys

from mcp.server import InitializationOptions, stdio

from .server import create_server


async def main() -> int:
    """Run the MCP server over stdio transport."""
    server = create_server()

    async with stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-research-agent-system",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(0)
