import os
import traceback
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Annotated, Any, TypedDict
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.redis.aio import AsyncRedisSaver  # Standard safe import path
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

load_dotenv()


@tool
def check_database_inventory(product_id: str) -> str:
    """Check the real-time stock availability level for a specific product ID in the warehouse."""
    # Mock inventory data
    db_mock = {
        "prod-01": "12 items remaining",
        "prod-02": "Out of Stock",
        "prod-03": "85 items remaining",
    }
    status = db_mock.get(
        product_id.lower(), "Product ID not found in inventory catalog."
    )
    print(f"\n[TOOL EXECUTION] Checked database inventory for {product_id}: {status}")
    return f"Inventory Status for {product_id}: {status}"


@tool
def send_email_to_customer(customer_email: str, content: str) -> str:
    """Send an official update notification email to a specific customer email address."""
    print(
        f"\n[TOOL EXECUTION] Dispatching email to {customer_email} with content: '{content[:30]}...'"
    )
    return f"Success: Email dispatched cleanly to {customer_email}."


# Combine defined tools into a single list
tools_list = [check_database_inventory, send_email_to_customer]
tools_node = ToolNode(tools_list)


# 1. Manage Redis connection lifecycle using Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    REDIS_URL = "redis://localhost:6380"
    print("[SYSTEM] Connecting and initializing Async Redis Checkpointer...")

    try:
        # 1. Initialize Context Manager
        redis_context = AsyncRedisSaver.from_conn_string(REDIS_URL)

        # 2. Use 'async with' block to extract the actual 'saver' instance
        async with redis_context as saver:
            # Set up necessary Redis schemas asynchronously
            await saver.setup()
            print(
                "[SYSTEM] Successfully initialized indices and Async Redis Checkpointer configuration!"
            )

            # Store the actual checkpointer object into app state
            app.state.checkpointer = saver

            # 3. Compile the graph once here
            workflow = StateGraph(State)
            workflow.add_node("agent", call_model)
            workflow.add_node("tools", tools_node)
            workflow.add_node("summarize_node", summarize_conversation_async)
            # 1. Start by routing traffic into the agent node
            workflow.add_edge(START, "agent")
            # 2. Let tools_condition decide whether to go to "tools" OR to END
            workflow.add_conditional_edges("agent", tools_condition)
            # 3. If a tool finishes running, always loop back into the agent node
            workflow.add_edge("tools", "agent")
            # Notice: There is NO 'workflow.add_edge("agent", END)' here!
            # Attach the async checkpointer safely
            app.state.graph_app = workflow.compile(
                checkpointer=saver,
                interrupt_before=[
                    "tools"
                ],  # Tells the engine to pause right BEFORE entering the "tools" node
            )

            # 4. Keep FastAPI running to serve requests
            yield

        # When you stop the server (Ctrl + C), FastAPI exits the yield
        print("[SYSTEM] Closing Redis Checkpointer safely...")

    except Exception as env_err:
        print(f"[SYSTEM CRITICAL ERROR INITIALIZING REDIS]: {traceback.format_exc()}")
        raise env_err


app = FastAPI(title="Modern LangGraph Chat with Lifespan", lifespan=lifespan)

# 2. Initialize LLM model
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", streaming=True).bind_tools(
    tools_list
)

llmOllama = ChatOllama(model="llama3.1")


# 3. Define State structure
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str  # store long-term memory summary


# 4. Define LLM processing node
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_context}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


async def call_model(state: State):
    chain = prompt_template | llm
    existing_summary = state.get("summary", "")

    system_context = (
        "You are an intelligent AI assistant. Respond naturally in English."
    )
    if existing_summary:
        system_context += (
            f" Summary of previous conversation context: {existing_summary}"
        )

    response = await chain.ainvoke(
        {"system_context": system_context, "messages": state["messages"]}
    )
    return {"messages": [response]}


# 5. Client request structure
class ChatRequest(BaseModel):
    prompt: str
    session_id: str


# 6. Generator for LangGraph token streaming
async def stream_langgraph_response(prompt: str, session_id: str, graph_app: Any):
    try:
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        input_data: Any = {"messages": [HumanMessage(content=prompt)]}

        async for event in graph_app.astream_events(
            input=input_data, config=config, version="v2"
        ):
            if (
                event["event"] == "on_chat_model_stream"
                and event["metadata"].get("langgraph_node") == "agent"
            ):
                chunk_content = event["data"]["chunk"].content
                token = ""

                # CRITICAL FIX: Properly extract the inner text string if Gemini passes a list
                if isinstance(chunk_content, list) and len(chunk_content) > 0:
                    first_item = chunk_content[0]  # Grab the first element dictionary
                    if isinstance(first_item, dict) and "text" in first_item:
                        token = first_item["text"]
                elif isinstance(chunk_content, str):
                    token = chunk_content

                if token:
                    # Send token, immediately clearing out Python memory
                    yield f"data: {token}\n\n"

    except Exception as e:
        print(f"\n[STREAM ERROR]: {traceback.format_exc()}")
        yield "data: [Error]\n\n"


# 7. Convert summarizer logic to run asynchronously
async def summarize_conversation_async(state: State):
    """Compresses oldest messages into text and issues deletion commands to Redis."""
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    if len(messages) < 6:
        return {}

    print(f"\n[MEMORY SYSTEM] Detected {len(messages)} messages. Compressing memory...")
    messages_to_summarize = messages[:-2]

    summary_prompt = (
        f"Progressively summarize the lines of conversation provided below concisely. "
        f"If a previous summary exists ({existing_summary}), incorporate the new context into it. "
        f"Return ONLY the plain summary text.\n\n"
    )
    for m in messages_to_summarize:
        role = "User" if isinstance(m, HumanMessage) else "AI"
        summary_prompt += f"{role}: {m.content}\n"

    # Use ainvoke here to ensure it doesn't freeze your background loop tasks
    new_summary = await llmOllama.ainvoke(summary_prompt)
    print(f"[MEMORY SYSTEM] Updated Summary: {new_summary.content}")

    # Generate directives to drop strings from Redis memory window
    delete_messages = [
        RemoveMessage(id=str(m.id)) for m in messages_to_summarize if m.id is not None
    ]

    return {"summary": str(new_summary.content), "messages": delete_messages}


# 8. Fixed Background memory supervisor using async interfaces
async def background_memory_check(session_id: str, graph_app: Any):
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    try:
        # FIX: Changed get_state to aget_state with an await statement
        state_snapshot = await graph_app.aget_state(config)
        current_state = state_snapshot.values
        messages = current_state.get("messages", [])

        # If message history window is overflowing, handle it
        if len(messages) >= 6:
            print(
                f"\n[BACKGROUND MEMORY] Triggering background summary for session: {session_id}"
            )

            summary_results = await summarize_conversation_async(current_state)

            if summary_results:
                # FIX: Changed update_state to aupdate_state with an await statement
                await graph_app.aupdate_state(
                    config, summary_results, as_node="summarize_node"
                )
                print(
                    "[BACKGROUND MEMORY] Background summary saved successfully to Redis!"
                )

    except Exception as bg_err:
        print(f"[BACKGROUND MEMORY ERROR]: {traceback.format_exc()}")


# --- TIME TRAVEL EXTENSIONS ---


class TimeTravelReplayRequest(BaseModel):
    session_id: str
    checkpoint_id: str
    user_prompt_override: str | None = None  # Optional: Fork history with a new input


async def stream_langgraph_time_travel(
    input_data: Any, historic_config: RunnableConfig, graph_app: Any
):
    """
    Streams responses specifically executing out of an explicit historical configuration checkpoint.
    """
    try:
        print(
            f"[TIME TRAVEL] Spawning timeline out of Checkpoint: {historic_config['configurable'].get('checkpoint_id')}"  # type: ignore
        )

        async for event in graph_app.astream_events(
            input=input_data, config=historic_config, version="v2"
        ):
            if (
                event["event"] == "on_chat_model_stream"
                and event["metadata"].get("langgraph_node") == "agent"
            ):
                chunk_content = event["data"]["chunk"].content
                token = ""

                if isinstance(chunk_content, list) and len(chunk_content) > 0:
                    first_item = chunk_content[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        token = first_item["text"]
                elif isinstance(chunk_content, str):
                    token = chunk_content

                if token:
                    yield f"data: {token}\n\n"

    except Exception as e:
        print(f"\n[TIME TRAVEL STREAM ERROR]: {traceback.format_exc()}")
        yield "data: [Time Travel Stream Error]\n\n"


@app.get("/api/chat/history/{session_id}")
async def get_session_history(session_id: str, fastapi_req: Request):
    """
    1. Fetch the state history stream from the Redis Checkpointer.
    Returns a list of past checkpoints, their timestamps, and the next node to execute.
    """
    graph_app = fastapi_req.app.state.graph_app
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    history_records = []

    # Iterate through all saved checkpoints for this specific thread_id
    async for state in graph_app.aget_state_history(config):
        history_records.append(
            {
                "checkpoint_id": state.config["configurable"]["checkpoint_id"],
                "next_executable_node": state.next,  # Tells you where the graph paused
                "has_summary": bool(state.values.get("summary")),
                "message_count": len(state.values.get("messages", [])),
                "last_message": (
                    state.values["messages"][-1].content
                    if state.values.get("messages")
                    else None
                ),
            }
        )

    return {"session_id": session_id, "history": history_records}


@app.post("/api/chat/time-travel")
async def time_travel_replay(request: TimeTravelReplayRequest, fastapi_req: Request):
    """
    2. Resume or Fork execution from a specific historical checkpoint ID.
    """
    graph_app = fastapi_req.app.state.graph_app

    # Build the historic config targeting the exact checkpoint snapshot
    historic_config: RunnableConfig = {
        "configurable": {
            "thread_id": request.session_id,
            "checkpoint_id": request.checkpoint_id,
        }
    }

    # Fetch the state at that exact microsecond in history
    historic_state = await graph_app.aget_state(historic_config)
    if not historic_state.values:
        return {"error": "Target checkpoint ID not found or expired in Redis memory."}

    # Scenario A: The user wants to fork history by providing a brand new prompt from that point
    if request.user_prompt_override:
        input_data = {"messages": [HumanMessage(content=request.user_prompt_override)]}

        # We stream the new alternate timeline starting precisely from the old checkpoint config
        return StreamingResponse(
            stream_langgraph_time_travel(input_data, historic_config, graph_app),
            media_type="text/event-stream",
        )

    # Scenario B: Just replay/resume execution on the exact existing state values
    else:
        # None passes no new inputs, meaning it forces the graph to run whatever node was up next
        return StreamingResponse(
            stream_langgraph_time_travel(None, historic_config, graph_app),
            media_type="text/event-stream",
        )


# --- TIME TRAVEL EXTENSIONS ---


# --- HUMAN IN THE LOOP---
class ActionReviewRequest(BaseModel):
    session_id: str
    feedback: str | None = None  # Optional feedback if rejecting or modifying


async def stream_langgraph_response_hitl(
    input_data: Any, config: RunnableConfig, graph_app: Any
):
    try:
        async for event in graph_app.astream_events(
            input=input_data, config=config, version="v2"
        ):
            if (
                event["event"] == "on_chat_model_stream"
                and event["metadata"].get("langgraph_node") == "agent"
            ):
                chunk_content = event["data"]["chunk"].content
                token = ""

                if isinstance(chunk_content, list) and len(chunk_content) > 0:
                    first_item = chunk_content[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        token = first_item["text"]
                elif isinstance(chunk_content, str):
                    token = chunk_content

                if token:
                    yield f"data: {token}\n\n"
    except Exception as e:
        print(f"\n[HITL STREAM ERROR]: {traceback.format_exc()}")
        yield "data: [Error]\n\n"


@app.post("/api/chat/approve")
async def approve_action(request: ActionReviewRequest, fastapi_req: Request):
    """
    Approves the pending tool execution and resumes the graph loop.
    """
    graph_app = fastapi_req.app.state.graph_app
    config: RunnableConfig = {"configurable": {"thread_id": request.session_id}}

    # 1. Verify the graph is actually waiting for an interrupt at the tools node
    state_snapshot = await graph_app.aget_state(config)
    if "tools" not in state_snapshot.next:
        return {
            "status": "error",
            "message": "The agent is not currently awaiting approval.",
        }

    print(
        f"[HITL] Action approved for session {request.session_id}. Resuming stream..."
    )

    # 2. Passing None tells the graph: "Change nothing in the state, just continue running."
    return StreamingResponse(
        stream_langgraph_response_hitl(None, config, graph_app),
        media_type="text/event-stream",
    )


@app.post("/api/chat/reject")
async def reject_action(request: ActionReviewRequest, fastapi_req: Request):
    """
    Rejects the tool execution, injects human feedback into the state,
    and forces the graph to route BACK to the agent instead of running the tool.
    """
    graph_app = fastapi_req.app.state.graph_app
    config: RunnableConfig = {"configurable": {"thread_id": request.session_id}}

    state_snapshot = await graph_app.aget_state(config)
    if "tools" not in state_snapshot.next:
        return {
            "status": "error",
            "message": "The agent is not currently awaiting approval.",
        }

    # Fetch the pending tool calls that the model requested
    last_message = state_snapshot.values["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if not tool_calls:
        return {"status": "error", "message": "No pending tool calls found to reject."}

    print(f"[HITL] Action rejected by human. Mocking tool rejection response...")

    # 1. Create fake/mocked ToolMessages explaining that the human rejected this action
    rejection_feedback = request.feedback or "Action denied by administrator."

    rejection_messages = [
        ToolMessage(
            content=f"Rejected by Human Overseer: {rejection_feedback}",
            tool_call_id=tc["id"],
        )
        for tc in tool_calls
    ]

    # 2. Update the state *as if* the tools node ran, but with our rejection message.
    # We use as_node="tools" so LangGraph thinks the tools phase is satisfied.
    await graph_app.aupdate_state(
        config, {"messages": rejection_messages}, as_node="tools"
    )

    # 3. Resume the graph. Because we supplied the ToolMessages, the conditional edge
    # will automatically route traffic back to the 'agent' node to read the rejection.
    return StreamingResponse(
        stream_langgraph_response_hitl(None, config, graph_app),
        media_type="text/event-stream",
    )


# --- HUMAN IN THE LOOP---


@app.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest, fastapi_req: Request, background_tasks: BackgroundTasks
):
    graph_app = fastapi_req.app.state.graph_app
    background_tasks.add_task(background_memory_check, request.session_id, graph_app)

    return StreamingResponse(
        stream_langgraph_response(request.prompt, request.session_id, graph_app),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
