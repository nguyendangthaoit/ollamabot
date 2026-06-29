from typing import Annotated, TypedDict
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str  # Store long-term memory summary


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
