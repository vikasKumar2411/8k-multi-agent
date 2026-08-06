# src/workflows/bounded_research/graph.py

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.workflows.bounded_research.nodes import (
    BoundedResearchNodes,
)
from src.workflows.bounded_research.routes import (
    route_after_planning,
)
from src.workflows.bounded_research.state import (
    BoundedResearchState,
)


def build_bounded_research_graph(
    *,
    nodes: BoundedResearchNodes | None = None,
    checkpointer: InMemorySaver | None = None,
) -> CompiledStateGraph:
    """
    Build and compile the bounded SEC research workflow.

    The default in-memory checkpointer is appropriate for local
    development only. It will later be replaced by PostgreSQL.
    """

    workflow_nodes = nodes or BoundedResearchNodes()
    workflow_checkpointer = (
        checkpointer or InMemorySaver()
    )

    builder = StateGraph(BoundedResearchState)
    
    builder.add_node(
        "plan_query",
        workflow_nodes.plan_query,
    )
    builder.add_node(
        "respond_to_rejection",
        workflow_nodes.respond_to_rejection,
    )
    builder.add_node(
        "respond_to_clarification",
        workflow_nodes.respond_to_clarification,
    )
    builder.add_node(
        "retrieve_evidence",
        workflow_nodes.retrieve_evidence,
    )
    builder.add_node(
        "analyze_evidence",
        workflow_nodes.analyze_evidence,
    )
    builder.add_node(
        "verify_analysis",
        workflow_nodes.verify_analysis,
    )
    builder.add_node(
        "generate_answer",
        workflow_nodes.generate_answer,
    )

    builder.add_edge(
        START,
        "plan_query",
    )

    builder.add_conditional_edges(
        "plan_query",
        route_after_planning,
        {
            "research": "retrieve_evidence",
            "reject": "respond_to_rejection",
            "clarify": "respond_to_clarification",
        },
    )

    builder.add_edge(
        "respond_to_rejection",
        END,
    )
    builder.add_edge(
        "respond_to_clarification",
        END,
    )

    builder.add_edge(
        "retrieve_evidence",
        "analyze_evidence",
    )
    builder.add_edge(
        "analyze_evidence",
        "verify_analysis",
    )
    builder.add_edge(
        "verify_analysis",
        "generate_answer",
    )
    builder.add_edge(
        "generate_answer",
        END,
    )

    return builder.compile(
        checkpointer=workflow_checkpointer
    )
