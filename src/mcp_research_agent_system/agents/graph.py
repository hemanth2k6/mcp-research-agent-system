"""LangGraph state machine wiring the research agent nodes together.

This module defines the control flow between the Planner, Researcher, Validator,
and Synthesizer nodes. The node implementations are STUBS for now — they log
execution and make minimal state updates so the graph is runnable end-to-end
before any real agent logic is implemented.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from .. import logging_utils
from .state import ResearchState

MAX_RESEARCHER_ATTEMPTS = 3
AGENT_NODE_EVENT = "agent_node"


def planner_node(state: ResearchState) -> ResearchState:
    """Stub planner node — logs execution and generates dummy sub-queries if none exist."""
    logging_utils.log_event(
        AGENT_NODE_EVENT,
        {"node": "planner", "research_goal": state.get("research_goal", "")},
    )
    # If no sub_queries yet, create dummy ones for the graph to progress
    if not state.get("sub_queries"):
        return {
            **state,
            "sub_queries": ["sub-query 1", "sub-query 2"],
            "current_query_index": 0,
            "researcher_attempts": 0,
            "validation_status": "pending",
        }
    return state


def researcher_node(state: ResearchState) -> ResearchState:
    """Stub researcher node — logs execution and simulates finding papers."""
    idx = state.get("current_query_index", 0)
    sub_queries = state.get("sub_queries", [])
    query = sub_queries[idx] if idx < len(sub_queries) else ""
    attempts = state.get("researcher_attempts", 0)
    logging_utils.log_event(
        AGENT_NODE_EVENT,
        {
            "node": "researcher",
            "current_query_index": idx,
            "query": query,
            "attempt": attempts,
        },
    )
    # Simulate researcher output - add some dummy papers
    return {
        **state,
        "researcher_output": [
            {"arxiv_id": f"dummy.{idx}.{attempts}", "title": f"Paper for {query}"}
        ],
        "researcher_attempts": attempts + 1,
    }


def validator_node(state: ResearchState) -> ResearchState:
    """Stub validator node — logs execution, advances state on valid, retries on invalid."""
    status = state.get("validation_status", "pending")
    attempts = state.get("researcher_attempts", 0)
    idx = state.get("current_query_index", 0)
    sub_queries = state.get("sub_queries", [])

    logging_utils.log_event(
        AGENT_NODE_EVENT,
        {
            "node": "validator",
            "validation_status": status,
            "current_query_index": idx,
            "attempts": attempts,
        },
    )

    # Handle "valid" status: advance to next query or stay for synthesizer
    if status == "valid":
        if idx + 1 < len(sub_queries):
            # More queries: advance to next, reset attempts
            return {
                **state,
                "validation_status": "pending",  # Reset for next query's validation
                "current_query_index": idx + 1,
                "researcher_attempts": 0,
            }
        else:
            # Last query was valid: keep valid status, don't advance (router will go to synthesizer)
            return state

    # Handle "pending" status: mark as valid and advance state
    if status == "pending":
        if idx + 1 < len(sub_queries):
            # More queries: advance to next, reset attempts
            return {
                **state,
                "validation_status": "valid",
                "current_query_index": idx + 1,
                "researcher_attempts": 0,
            }
        else:
            # Last query: mark valid, don't advance (will go to synthesizer)
            return {**state, "validation_status": "valid"}

    # If invalid/failed, return as-is (router will handle retry or exhaustion)
    return state


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
