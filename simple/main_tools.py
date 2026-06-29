from contextlib import asynccontextmanager
from typing import Annotated, TypedDict

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# =========================================================
# TOOLS
# =========================================================


@tool
def check_database_inventory(product_id: str) -> str:
    """Check the real-time stock availability level for a specific product ID in the warehouse."""
    # Mock inventory data
    db_mock = {
        "prod-01": "12 items remaining",
        "prod-02": "Out of Stock",
        "prod-03": "85 items remaining",
    }
    print(f"\n[TOOL EXECUTION] Checked database inventory for {product_id}")
    return db_mock.get(product_id, "Not found")


@tool
def send_email_to_customer(customer_email: str, content: str) -> str:
    """Send an official update notification email to a specific customer email address."""
    return f"Email sent to {customer_email}"


tools = [check_database_inventory, send_email_to_customer]
tool_node = ToolNode(tools)


# =========================================================
# STATE
# =========================================================


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


# =========================================================
# MODEL
# =========================================================

llm = ChatOllama(model="llama3.1")
llm_with_tools = llm.bind_tools(tools)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system}"),
        MessagesPlaceholder("messages"),
    ]
)

CHAIN = prompt | llm_with_tools


# =========================================================
# CONFIG
# =========================================================

K = 6  # last K messages


# =========================================================
# AGENT NODE
# =========================================================


def agent_node(state: State):
    summary = state.get("summary", "")
    messages = state["messages"]

    recent_messages = messages[-K:]

    system = "You are a helpful assistant."

    if summary:
        system += f"\nSummary: {summary}"

    response = CHAIN.invoke({"system": system, "messages": recent_messages})

    return {"messages": [response]}


# =========================================================
# SUMMARIZER NODE (IN GRAPH - NO BACKGROUND TASK)
# =========================================================


def summarize_node(state: State):
    messages = state["messages"]
    summary = state.get("summary", "")

    if len(messages) < 10:
        return {}

    old = messages[:-K]

    text = f"Update summary:\nPrevious: {summary}\n\n"

    for m in old:
        role = "User" if isinstance(m, HumanMessage) else "AI"
        text += f"{role}: {m.content}\n"

    new_summary = llm.invoke(text).content

    return {"summary": new_summary}


# =========================================================
# SHOULD SUMMARIZE ROUTER
# =========================================================


def should_summarize(state: State):
    if len(state["messages"]) > 10:
        return "summarize"
    return "end"


# =========================================================
# GRAPH
# =========================================================

workflow = StateGraph(State)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("summarize", summarize_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges("agent", tools_condition)

workflow.add_edge("tools", "agent")

workflow.add_conditional_edges(
    "agent", should_summarize, {"summarize": "summarize", "end": END}
)

workflow.add_edge("summarize", "agent")


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = RedisSaver.from_conn_string("redis://localhost:6380")

    with redis as saver:
        saver.setup()

        app.state.graph = workflow.compile(checkpointer=saver)
        yield


app = FastAPI(lifespan=lifespan)


# =========================================================
# REQUEST
# =========================================================


class ChatRequest(BaseModel):
    prompt: str
    session_id: str


# =========================================================
# STREAMING
# =========================================================


def stream(prompt: str, session_id: str, graph):

    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    inputs = {"messages": [HumanMessage(content=prompt)]}

    for chunk, meta in graph.stream(inputs, config, stream_mode="messages"):
        if meta.get("langgraph_node") == "agent":
            if chunk.content:
                yield f"data: {chunk.content}\n\n"


# =========================================================
# API
# =========================================================


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):

    graph = request.app.state.graph

    return StreamingResponse(
        stream(req.prompt, req.session_id, graph), media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn

    # Run the application using uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
