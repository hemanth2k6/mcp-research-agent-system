"""LangGraph state machine wiring the research agent nodes together.

This module defines the control flow between the Planner, Researcher, Validator,
and Synthesizer nodes.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from .. import logging_utils
from ..config import Settings
from .planner import decompose_goal, validate_research_output
from .researcher import PaperResult, ResearcherError, ResearchResult, run_research
from .state import ResearchState

MAX_RESEARCHER_ATTEMPTS = 3
AGENT_NODE_EVENT = "agent_node"


def planner_node(state: ResearchState) -> ResearchState:
    """Planner node — decomposes the research goal into sub-queries on first entry."""
    research_goal = state.get("research_goal", "")
    logging_utils.log_event(
        AGENT_NODE_EVENT,
        {"node": "planner", "research_goal": research_goal, "phase": "entry"},
    )

    # Only decompose if sub_queries not yet populated
    if not state.get("sub_queries"):
        try:
            decomposition = decompose_goal(research_goal)
            sub_queries = decomposition.sub_queries
            logging_utils.log_event(
                AGENT_NODE_EVENT,
                {
                    "node": "planner",
                    "phase": "decomposed",
                    "research_goal": research_goal,
                    "sub_queries": sub_queries,
                    "sub_query_count": len(sub_queries),
                },
            )
            return {
                **state,
                "sub_queries": sub_queries,
                "current_query_index": 0,
                "researcher_attempts": 0,
                "validation_status": "pending",
            }
        except Exception as e:
            # Log the error and re-raise to halt the graph
            logging_utils.log_event(
                AGENT_NODE_EVENT,
                {
                    "node": "planner",
                    "phase": "error",
                    "research_goal": research_goal,
                    "error": str(e),
                },
            )
            raise

    # Already has sub_queries — pass through
    logging_utils.log_event(
        AGENT_NODE_EVENT,
        {"node": "planner", "phase": "passthrough", "sub_queries": state.get("sub_queries")},
    )
    return state


async def researcher_node(state: ResearchState) -> ResearchState:
    """Researcher node — calls MCP server to search papers for the current sub-query."""
    idx = state.get("current_query_index", 0)
    sub_queries = state.get("sub_queries", [])
    query = sub_queries[idx] if idx < len(sub_queries) else ""
    attempts = state.get("researcher_attempts", 0)

    logging_utils.log_event(
        AGENT_NODE_EVENT,
        {
            "node": "researcher",
            "phase": "entry",
            "current_query_index": idx,
            "query": query,
            "attempt": attempts,
        },
    )

    if not query:
        logging_utils.log_event(
            AGENT_NODE_EVENT,
            {
                "node": "researcher",
                "phase": "error",
                "error": "No sub-query available at current index",
                "current_query_index": idx,
                "sub_queries": sub_queries,
            },
        )
        return {
            **state,
            "researcher_output": [],
            "researcher_attempts": attempts + 1,
            "error": "No sub-query available at current index",
        }

    settings = Settings()

    try:
        result = await run_research(query, settings)

        logging_utils.log_event(
            AGENT_NODE_EVENT,
            {
                "node": "researcher",
                "phase": "success",
                "current_query_index": idx,
                "query": query,
                "paper_count": len(result.papers),
                "cached_summary_count": len(result.cached_summaries),
                "raw_tool_calls": result.raw_tool_calls,
            },
        )

        # Convert PaperResult models to dict for state storage
        papers_as_dict = [p.model_dump(mode="json") for p in result.papers]

        return {
            **state,
            "researcher_output": papers_as_dict,
            # researcher_attempts incremented by validator_node on retry
        }

    except ResearcherError as e:
        logging_utils.log_event(
            AGENT_NODE_EVENT,
            {
                "node": "researcher",
                "phase": "error",
                "current_query_index": idx,
                "query": query,
                "error": str(e),
                "details": e.details,
            },
        )
        return {
            **state,
            "researcher_output": [],
            # researcher_attempts incremented by validator_node on retry
            "error": str(e),
        }


async def validator_node(state: ResearchState) -> ResearchState:
    """Validator node — validates researcher output for the current sub-query.

    Calls validate_research_output() which:
    1. Runs heuristic checks (empty papers, token overlap) - fast, no LLM
    2. Falls back to LLM judge for ambiguous cases
    3. Returns ValidationOutcome with is_valid, reason, and revised_query if invalid

    If invalid and attempts < MAX, updates current_sub_query to revised_query
    and loops back to researcher_node.
    """
    idx = state.get("current_query_index", 0)
    sub_queries = state.get("sub_queries", [])
    attempts = state.get("researcher_attempts", 0)
    researcher_output = state.get("researcher_output", [])

    query = sub_queries[idx] if idx < len(sub_queries) else ""

    logging_utils.log_event(
        AGENT_NODE_EVENT,
        {
            "node": "validator",
            "phase": "entry",
            "current_query_index": idx,
            "query": query,
            "attempts": attempts,
            "paper_count": len(researcher_output),
        },
    )

    if not query:
        logging_utils.log_event(
            AGENT_NODE_EVENT,
            {
                "node": "validator",
                "phase": "error",
                "error": "No sub-query available at current index",
                "current_query_index": idx,
                "sub_queries": sub_queries,
            },
        )
        return {
            **state,
            "validation_status": "invalid",
            "error": "No sub-query available at current index",
        }

    # Convert researcher_output (list of dicts) back to ResearchResult for validation
    papers = []
    for p in researcher_output:
        paper = PaperResult(
            arxiv_id=p.get("arxiv_id", ""),
            title=p.get("title", ""),
            authors=p.get("authors", []),
            abstract=p.get("abstract", ""),
            category=p.get("category", ""),
            published_date=p.get("published_date", ""),
            updated_date=p.get("updated_date", ""),
            pdf_url=p.get("pdf_url", ""),
        )
        papers.append(paper)

    research_result = ResearchResult(
        sub_query=query,
        papers=papers,
        cached_summaries=[],
        raw_tool_calls=[],
    )

    # Run validation
    validation_result = await validate_research_output(query, research_result)

    # Log the full validation decision
    log_payload = {
        "node": "validator",
        "phase": "validation_result",
        "current_query_index": idx,
        "query": query,
        "attempts": attempts,
        "is_valid": validation_result.is_valid,
        "reason": validation_result.reason,
        "heuristic_used": validation_result.reason.startswith("Research returned") or validation_result.reason.startswith("Papers have"),
    }

    if validation_result.revised_query:
        log_payload["revised_query"] = validation_result.revised_query
        log_payload["revised_query_changed"] = validation_result.revised_query != query

    logging_utils.log_event(AGENT_NODE_EVENT, log_payload)

    if validation_result.is_valid:
        # Valid result - accumulate findings and advance
        validated_findings = state.get("validated_findings", []) + researcher_output

        if idx + 1 < len(sub_queries):
            # More queries: advance to next, reset attempts
            logging_utils.log_event(
                AGENT_NODE_EVENT,
                {
                    "node": "validator",
                    "phase": "valid_advance",
                    "next_query_index": idx + 1,
                    "validated_findings_count": len(validated_findings),
                },
            )
            return {
                **state,
                "validation_status": "pending",  # Reset for next query's validation
                "current_query_index": idx + 1,
                "researcher_attempts": 0,
                "validated_findings": validated_findings,
            }
        else:
            # Last query was valid: keep valid status, don't advance (router will go to synthesizer)
            logging_utils.log_event(
                AGENT_NODE_EVENT,
                {
                    "node": "validator",
                    "phase": "valid_done",
                    "validated_findings_count": len(validated_findings),
                    "going_to": "synthesizer",
                },
            )
            return {
                **state,
                "validation_status": "valid",
                "validated_findings": validated_findings,
            }

    # Invalid result
    if attempts < MAX_RESEARCHER_ATTEMPTS:
        # Retry with revised query
        revised_query = validation_result.revised_query or query
        logging_utils.log_event(
            AGENT_NODE_EVENT,
            {
                "node": "validator",
                "phase": "invalid_retry",
                "current_query_index": idx,
                "attempt": attempts + 1,
                "max_attempts": MAX_RESEARCHER_ATTEMPTS,
                "original_query": query,
                "revised_query": revised_query,
                "reason": validation_result.reason,
            },
        )

        # Update the current sub-query in state to the revised query
        new_sub_queries = list(sub_queries)
        new_sub_queries[idx] = revised_query

        return {
            **state,
            "validation_status": "invalid",
            "sub_queries": new_sub_queries,
            "researcher_attempts": attempts + 1,
        }

    # Exhausted attempts - proceed to synthesizer with what we have
    logging_utils.log_event(
        AGENT_NODE_EVENT,
        {
            "node": "validator",
            "phase": "invalid_exhausted",
            "current_query_index": idx,
            "attempts": attempts,
            "max_attempts": MAX_RESEARCHER_ATTEMPTS,
            "query": query,
            "reason": validation_result.reason,
            "going_to": "synthesizer",
            "note": "max attempts reached, proceeding with partial results",
        },
    )

    return {
        **state,
        "validation_status": "invalid",
        "error": f"Validation failed after {attempts} attempts: {validation_result.reason}",
    }


def synthesizer_node(state: ResearchState) -> ResearchState:
    """Stub synthesizer node — logs execution and creates final report."""
    logging_utils.log_event(
        AGENT_NODE_EVENT,
        {
            "node": "synthesizer",
            "validated_findings_count": len(state.get("validated_findings", [])),
        },
    )
    # Accumulate findings and generate dummy report
    new_findings = state.get("validated_findings", []) + state.get("researcher_output", [])
    return {
        **state,
        "validated_findings": new_findings,
        "final_report": f"Synthesized report from {len(new_findings)} findings",
        "error": state.get("error"),
    }


def _route_after_validator(state: ResearchState) -> str:
    """Conditional edge from validator_node.

    Routing logic:
      - valid:
          * more sub-queries remain -> researcher (advance to next query, reset attempts)
          * no more sub-queries -> synthesizer
      - invalid and attempts < MAX:
          -> researcher (increment attempts, retry same query)
      - invalid and attempts >= MAX:
          -> synthesizer (give up on this query, proceed with what we have)
    """
    status = state.get("validation_status", "pending")
    attempts = state.get("researcher_attempts", 0)
    idx = state.get("current_query_index", 0)
    sub_queries = state.get("sub_queries", [])

    if status == "valid":
        if idx + 1 < len(sub_queries):
            logging_utils.log_event(
                AGENT_NODE_EVENT,
                {
                    "node": "validator",
                    "decision": "valid_continue",
                    "next_query_index": idx + 1,
                },
            )
            return "researcher"
        logging_utils.log_event(
            AGENT_NODE_EVENT,
            {"node": "validator", "decision": "valid_done", "going_to": "synthesizer"},
        )
        return "synthesizer"

    # status is "invalid" (or "failed"/"pending" fallback)
    if attempts < MAX_RESEARCHER_ATTEMPTS:
        logging_utils.log_event(
            AGENT_NODE_EVENT,
            {
                "node": "validator",
                "decision": "invalid_retry",
                "attempt": attempts,
                "going_to": "researcher",
            },
        )
        return "researcher"

    logging_utils.log_event(
        AGENT_NODE_EVENT,
        {
            "node": "validator",
            "decision": "invalid_exhausted",
            "attempt": attempts,
            "going_to": "synthesizer",
            "note": "max attempts reached, proceeding with partial results",
        },
    )
    return "synthesizer"


def build_graph() -> Any:
    """Build and compile the research agent state graph.

    Returns:
        A compiled LangGraph StateGraph ready to be invoked with a ResearchState.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("validator", validator_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "validator")
    graph.add_conditional_edges("validator", _route_after_validator)
    graph.add_edge("synthesizer", END)

    return graph.compile()


# Pre-compiled graph instance for convenience
compiled_graph: Any = build_graph()
