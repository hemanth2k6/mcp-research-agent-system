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

## Configuration

All settings via `.env` (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `http://localhost:20128/v1` |
| `LLM_API_KEY` | API key | required |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `RESEARCH_AGENT_DB_PATH` | SQLite database path | `data/research_agent.db` |
| `LOG_DIR` | Trace log directory | `logs/` |