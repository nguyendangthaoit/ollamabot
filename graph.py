from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
from states import State
from nodes import (
    call_model,
    retrieve_node,
    route_question,
    summarize_conversation_async,
)
from tools import tools_node


def create_workflow() -> StateGraph:
    workflow = StateGraph(State)

    # 1. Register Nodes (Explicitly map string names to your functions)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tools_node)
    workflow.add_node("summarize_node", summarize_conversation_async)
    workflow.add_node("retrieve", retrieve_node)

    # 2. Wire up Graph Topology
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "retrieve": "retrieve",
            "direct_generate": "agent",
        },
    )
    workflow.add_edge("retrieve", "agent")

    # 3. Tool Loop Execution Paths
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow
