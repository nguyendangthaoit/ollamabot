from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# state for the graph
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str  # Store long-term memory summary
    retrieved_context: str


class RouteDecision(BaseModel):
    """Determine if the query requires internal corporate database knowledge."""

    next_node: Literal["retrieve", "direct_generate"]


# request or respon shema
class ChatRequest(BaseModel):
    prompt: str
    session_id: str


class TimeTravelReplayRequest(BaseModel):
    session_id: str
    checkpoint_id: str
    user_prompt_override: str | None = None


class ActionReviewRequest(BaseModel):
    session_id: str
    feedback: str | None = None
