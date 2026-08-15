# MCP Research Agent System

A production-quality multi-agent research system with an arXiv MCP server and LangGraph-based research agents.

## Architecture

- **MCP Server** — Wraps the arXiv public API with SQLite caching (`search_papers`, `get_paper_details`, `get_cached_summary`)
- **LangGraph Agents** — Planner, Researcher, Synthesizer with validation & retry logic
- **Structured Logging** — Full state-machine trace as JSONL

## Quick Start

```bash
# 1. Copy .env.example and fill in your LLM credentials
cp .env.example .env

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Run the MCP server (stdio transport)
python scripts/run_mcp_server.py

# 4. In another terminal, run a research query
python scripts/run_agent.py --goal "Your research topic here"
```

## Running the MCP Server Standalone

The MCP server can be run directly as a module using stdio transport:

```bash
# Run the MCP server (reads JSON-RPC from stdin, writes to stdout)
python -m mcp_research_agent_system.mcp_server
```

This starts the server over stdio, which is the standard MCP transport for local development. The server exposes three tools:

| Tool | Description |
|------|-------------|
| `search_papers` | Search arXiv for papers with caching. Returns a list of papers with title, authors, abstract snippet, published date, and PDF URL. |
| `get_paper_details` | Get full paper details from the local cache by arXiv ID. Returns the complete paper record including full abstract. |
| `get_cached_summary` | Check if we have already researched a topic by searching cached search results. Returns matching cached entries with query text, category, arxiv IDs, and TTL info. |

Example JSON-RPC interaction:

```bash
# Initialize the session
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | python -m mcp_research_agent_system.mcp_server

# List available tools
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | python -m mcp_research_agent_system.mcp_server

# Search for papers
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_papers","arguments":{"query":"machine learning","category":"cs.AI","max_results":5}}}' | python -m mcp_research_agent_system.mcp_server
```

## Configuration

All settings via `.env` (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `http://localhost:20128/v1` |
| `LLM_API_KEY` | API key | required |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `RESEARCH_AGENT_DB_PATH` | SQLite database path | `data/research_agent.db` |
| `LOG_DIR` | Trace log directory | `logs/` |