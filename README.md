# MCP Research Agent System

![CI](https://github.com/hemanth2k6/mcp-research-agent-system/actions/workflows/ci.yml/badge.svg)

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
#    The project works with any OpenAI-compatible endpoint (OmniRoute, local vLLM, etc.)
python scripts/run_mcp_server.py

# 4. In another terminal, run a research query using the CLI
research-agent "What are the latest advances in transformer architectures?"
```

### CLI Usage

```bash
# Basic usage
research-agent "Your research question here"

# Verbose mode (shows node-by-node progress)
research-agent "Impact of quantum computing on cryptography" --verbose

# Write report to a file
research-agent "LLM alignment techniques" --output report.md

# Both verbose and file output
research-agent "Recent developments in diffusion models" -v -o report.md
```

### Example Output

```bash
$ research-agent "What are the latest advances in transformer architectures?" --verbose
Starting research pipeline for: What are the latest advances in transformer architectures?
Trace logs will be written to: logs/

  → planner: Generated 3 sub-queries: transformer architecture improvements, attention mechanism variants, efficiency optimizations
  → researcher: Query 1: transformer architecture improvements... (attempt 1, 5 papers)
  → validator: Validation passed (total validated: 5)
  → researcher: Query 2: attention mechanism variants... (attempt 1, 4 papers)
  → validator: Validation passed (total validated: 9)
  → researcher: Query 3: efficiency optimizations... (attempt 1, 6 papers)
  → validator: Validation passed (total validated: 15)
  → synthesizer: Report generated (4231 chars)

============================================================
# Latest Advances in Transformer Architectures

## Executive Summary
...

## Key Findings
...

## References
...

Full trace log written to: logs/
```

## Running the MCP Server Standalone

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

## Running with Docker

Containerize the entire system (MCP server + agent CLI) for zero-setup demos and deployments.

### Quick Start

```bash
# 1. Build the image
docker build -t mcp-research-agent-system .

# 2. Run a research query (CLI is the default entrypoint)
docker run --rm \
  -v ./data:/app/data \
  -v ./logs:/app/logs \
  --env-file .env \
  mcp-research-agent-system "What are the latest advances in transformer architectures?" --verbose
```

### Run the MCP Server Standalone

```bash
docker run --rm \
  -v ./data:/app/data \
  -v ./logs:/app/logs \
  --env-file .env \
  --entrypoint python \
  mcp-research-agent-system -m mcp_research_agent_system.mcp_server
```

### With Docker Compose

```bash
# Build and run in one command (interactive CLI)
docker compose run --rm research-agent "Impact of quantum computing on cryptography"

# Run MCP server as a background service
docker compose up research-agent
```

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | SQLite database persistence |
| `./logs` | `/app/logs` | Structured JSONL trace logs |

### Environment Variables

All configuration is read from `.env` (never baked into the image):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `http://localhost:20128/v1` |
| `LLM_API_KEY` | API key | required |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `RESEARCH_AGENT_DB_PATH` | SQLite database path | `data/research_agent.db` |
| `LOG_DIR` | Trace log directory | `logs/` |

## Configuration

All settings via `.env` (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `http://localhost:20128/v1` |
| `LLM_API_KEY` | API key | required |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `RESEARCH_AGENT_DB_PATH` | SQLite database path | `data/research_agent.db` |
| `LOG_DIR` | Trace log directory | `logs/` |