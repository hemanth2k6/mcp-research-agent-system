"""LangGraph state machine wiring the research agent nodes together.

This module defines the control flow between the Planner, Researcher, Validator,
and Synthesizer nodes.
"""

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from .. import logging_utils
from ..config import Settings
from .planner import decompose_goal, validate_research_output
from .researcher import PaperResult, ResearcherError, ResearchResult, run_research
from .state import ResearchState
from .synthesizer import synthesize_report

MAX_RESEARCHER_ATTEMPTS = 3


def planner_node(state: ResearchState) -> ResearchState:
    """Planner node — decomposes the research goal into sub-queries on first entry."""
    start_time = logging_utils.log_node_entry(
        "planner",
        logging_utils.safe_state_snapshot(
            cast(dict[str, Any], state), ["research_goal", "sub_queries"]
        ),
    )

    research_goal = state.get("research_goal", "")

    # Only decompose if sub_queries not yet populated
    if not state.get("sub_queries"):
        try:
            decomposition = decompose_goal(research_goal)
            sub_queries = decomposition.sub_queries

            result_state = {
                **state,
                "sub_queries": sub_queries,
                "current_query_index": 0,
                "researcher_attempts": 0,
                "validation_status": "pending",
            }
            logging_utils.log_node_exit(
                "planner",
                logging_utils.safe_state_snapshot(result_state),
                start_time,
                status="success",
            )
            return cast(ResearchState, result_state)
        except Exception as e:
            logging_utils.log_node_error(
                "planner",
                str(e),
                logging_utils.safe_state_snapshot(cast(dict[str, Any], state), ["research_goal"]),
                start_time,
            )
            raise

    # Already has sub_queries — pass through
    logging_utils.log_node_exit(
        "planner",
        logging_utils.safe_state_snapshot(cast(dict[str, Any], state)),
        start_time,
        status="passthrough",
    )
    return state


async def researcher_node(state: ResearchState) -> ResearchState:
    """Researcher node — calls MCP server to search papers for the current sub-query."""
    start_time = logging_utils.log_node_entry(
        "researcher",
        logging_utils.safe_state_snapshot(
            cast(dict[str, Any], state),
            [
                "research_goal",
                "sub_queries",
                "current_query_index",
                "researcher_attempts",
            ],
        ),
    )

    idx = state.get("current_query_index", 0)
    sub_queries = state.get("sub_queries", [])
    query = sub_queries[idx] if idx < len(sub_queries) else ""
    attempts = state.get("researcher_attempts", 0)

    if not query:
        logging_utils.log_node_error(
            "researcher",
            "No sub-query available at current index",
            logging_utils.safe_state_snapshot(
                cast(dict[str, Any], state), ["current_query_index", "sub_queries"]
            ),
            start_time,
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

        papers_as_dict = [p.model_dump(mode="json") for p in result.papers]

        result_state = {
            **state,
            "researcher_output": papers_as_dict,
            # researcher_attempts incremented by validator_node on retry
        }
        logging_utils.log_node_exit(
            "researcher",
            logging_utils.safe_state_snapshot(
                result_state,
                ["researcher_output", "current_query_index", "researcher_attempts"],
            ),
            start_time,
            status="success",
        )
        return cast(ResearchState, result_state)

    except ResearcherError as e:
        logging_utils.log_node_error(
            "researcher",
            str(e),
            logging_utils.safe_state_snapshot(
                cast(dict[str, Any], state), ["current_query_index", "query"]
            ),
            start_time,
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
    start_time = logging_utils.log_node_entry(
        "validator",
        logging_utils.safe_state_snapshot(
            cast(dict[str, Any], state),
            [
                "research_goal",
                "sub_queries",
                "current_query_index",
                "researcher_attempts",
                "validation_status",
                "researcher_output",
            ],
        ),
    )

    idx = state.get("current_query_index", 0)
    sub_queries = state.get("sub_queries", [])
    attempts = state.get("researcher_attempts", 0)
    researcher_output = state.get("researcher_output", [])

    query = sub_queries[idx] if idx < len(sub_queries) else ""

    if not query:
        logging_utils.log_node_error(
            "validator",
            "No sub-query available at current index",
            logging_utils.safe_state_snapshot(
                cast(dict[str, Any], state), ["current_query_index", "sub_queries"]
            ),
            start_time,
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

    if validation_result.is_valid:
        # Valid result - accumulate findings and advance
        validated_findings = state.get("validated_findings", []) + researcher_output

        if idx + 1 < len(sub_queries):
            # More queries: advance to next, reset attempts
            result_state = {
                **state,
                "validation_status": "pending",  # Reset for next query's validation
                "current_query_index": idx + 1,
                "researcher_attempts": 0,
                "validated_findings": validated_findings,
            }
            logging_utils.log_node_exit(
                "validator",
                logging_utils.safe_state_snapshot(
                    result_state,
                    ["validation_status", "current_query_index", "validated_findings"],
                ),
                start_time,
                status="valid_advance",
            )
            return cast(ResearchState, result_state)
        else:
            # Last query was valid: keep valid status, don't advance (router will go to synthesizer)
            result_state = {
                **state,
                "validation_status": "valid",
                "validated_findings": validated_findings,
            }
            logging_utils.log_node_exit(
                "validator",
                logging_utils.safe_state_snapshot(
                    result_state,
                    ["validation_status", "validated_findings"],
                ),
                start_time,
                status="valid_done",
            )
            return cast(ResearchState, result_state)

    # Invalid result
    if attempts < MAX_RESEARCHER_ATTEMPTS:
        # Retry with revised query
        revised_query = validation_result.revised_query or query
        new_sub_queries = list(sub_queries)
        new_sub_queries[idx] = revised_query

        result_state = {
            **state,
            "validation_status": "invalid",
            "sub_queries": new_sub_queries,
            "researcher_attempts": attempts + 1,
        }
        logging_utils.log_node_exit(
            "validator",
            logging_utils.safe_state_snapshot(
                result_state,
                ["validation_status", "sub_queries", "researcher_attempts"],
            ),
            start_time,
            status="invalid_retry",
        )
        return cast(ResearchState, result_state)

    # Exhausted attempts - proceed to synthesizer with what we have
    result_state = {
        **state,
        "validation_status": "invalid",
        "error": f"Validation failed after {attempts} attempts: {validation_result.reason}",
    }
    logging_utils.log_node_exit(
        "validator",
        logging_utils.safe_state_snapshot(
            result_state,
            ["validation_status", "error"],
        ),
        start_time,
        status="invalid_exhausted",
    )
    return cast(ResearchState, result_state)


async def synthesizer_node(state: ResearchState) -> ResearchState:
    """Synthesizer node — turns accumulated validated findings into a final report.

    Calls synthesize_report() with the accumulated state.validated_findings and
    research_goal, stores the result in state.final_report, and logs input/output.
    """
    start_time = logging_utils.log_node_entry(
        "synthesizer",
        logging_utils.safe_state_snapshot(
            cast(dict[str, Any], state),
            ["research_goal", "validated_findings", "validation_status"],
        ),
    )

    validated_findings = state.get("validated_findings", [])
    research_goal = state.get("research_goal", "")

    try:
        final_report = await synthesize_report(research_goal, validated_findings)

        result_state = {
            **state,
            "final_report": final_report,
            "error": state.get("error"),
        }
        logging_utils.log_node_exit(
            "synthesizer",
            logging_utils.safe_state_snapshot(
                result_state,
                ["final_report", "error"],
            ),
            start_time,
            status="success",
        )
        return cast(ResearchState, result_state)

    except Exception as e:
        logging_utils.log_node_error(
            "synthesizer",
            str(e),
            logging_utils.safe_state_snapshot(
                cast(dict[str, Any], state), ["validated_findings", "research_goal"]
            ),
            start_time,
        )
        raise


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
            return "researcher"
        return "synthesizer"

    # status is "invalid" (or "failed"/"pending" fallback)
    if attempts < MAX_RESEARCHER_ATTEMPTS:
        return "researcher"

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
