import traceback
from contextlib import asynccontextmanager
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

# 1. Load env vars first
load_dotenv()

# 2. Import and initialize telemetry immediately
from telemetry import init_telemetry

init_telemetry()

# 3. NOW import your agent components so they get instrumented correctly
from states import ChatRequest, TimeTravelReplayRequest, ActionReviewRequest
from graph import create_workflow
from nodes import background_memory_check


@asynccontextmanager
async def lifespan(app: FastAPI):
    REDIS_URL = "redis://localhost:6380"
    print("[SYSTEM] Connecting and initializing Async Redis Checkpointer...")
    try:
        redis_context = AsyncRedisSaver.from_conn_string(REDIS_URL)
        async with redis_context as saver:
            await saver.setup()
            app.state.checkpointer = saver
            # Setup work-loop graph matching workflow configurations
            workflow = create_workflow()
            app.state.graph_app = workflow.compile(checkpointer=saver)

            print(
                "[SYSTEM] Successfully initialized Async Redis Checkpointer and Graph configurations!"
            )
            yield
        print("[SYSTEM] Closing Redis Checkpointer safely...")
    except Exception as env_err:
        print(f"[SYSTEM CRITICAL ERROR INITIALIZING REDIS]: {traceback.format_exc()}")
        raise env_err


app = FastAPI(title="Modern LangGraph Chat Architecture", lifespan=lifespan)


# --- UTILITIES FOR STREAMING ---
async def stream_langgraph_events(
    prompt: str | None,
    session_id: str,
    graph_app: Any,
    historic_config: RunnableConfig | None = None,
    node_target: str = "agent",
):
    try:
        config = historic_config or {"configurable": {"thread_id": session_id}}
        input_data = {"messages": [HumanMessage(content=prompt)]} if prompt else None

        async for event in graph_app.astream_events(
            input=input_data, config=config, version="v2"
        ):
            if (
                event["event"] == "on_chat_model_stream"
                and event["metadata"].get("langgraph_node") == node_target
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
    except Exception:
        print(f"\n[STREAM ERROR]: {traceback.format_exc()}")
        yield "data: [Error occurred during streaming]\n\n"


# --- API ENDPOINTS ---


@app.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest, fastapi_req: Request, background_tasks: BackgroundTasks
):
    graph_app = fastapi_req.app.state.graph_app
    background_tasks.add_task(background_memory_check, request.session_id, graph_app)
    return StreamingResponse(
        stream_langgraph_events(request.prompt, request.session_id, graph_app),
        media_type="text/event-stream",
    )


@app.get("/api/chat/history/{session_id}")
async def get_session_history(session_id: str, fastapi_req: Request):
    graph_app = fastapi_req.app.state.graph_app
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}
    history_records = []

    async for state in graph_app.aget_state_history(config):
        history_records.append(
            {
                "checkpoint_id": state.config["configurable"]["checkpoint_id"],
                "next_executable_node": state.next,
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
    graph_app = fastapi_req.app.state.graph_app
    historic_config: RunnableConfig = {
        "configurable": {
            "thread_id": request.session_id,
            "checkpoint_id": request.checkpoint_id,
        }
    }
    historic_state = await graph_app.aget_state(historic_config)
    if not historic_state.values:
        return {"error": "Target checkpoint ID not found or expired."}

    prompt = (
        request.user_prompt_override
    )  # will pass None if user wants exact historical resume
    return StreamingResponse(
        stream_langgraph_events(
            prompt, request.session_id, graph_app, historic_config=historic_config
        ),
        media_type="text/event-stream",
    )


@app.post("/api/chat/approve")
async def approve_action(request: ActionReviewRequest, fastapi_req: Request):
    graph_app = fastapi_req.app.state.graph_app
    config: RunnableConfig = {"configurable": {"thread_id": request.session_id}}
    state_snapshot = await graph_app.aget_state(config)

    if "tools" not in state_snapshot.next:
        return {
            "status": "error",
            "message": "The agent is not currently awaiting approval.",
        }

    return StreamingResponse(
        stream_langgraph_events(
            None, request.session_id, graph_app, historic_config=config
        ),
        media_type="text/event-stream",
    )


@app.post("/api/chat/reject")
async def reject_action(request: ActionReviewRequest, fastapi_req: Request):
    graph_app = fastapi_req.app.state.graph_app
    config: RunnableConfig = {"configurable": {"thread_id": request.session_id}}
    state_snapshot = await graph_app.aget_state(config)

    if "tools" not in state_snapshot.next:
        return {
            "status": "error",
            "message": "The agent is not currently awaiting approval.",
        }

    last_message = state_snapshot.values["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    if not tool_calls:
        return {"status": "error", "message": "No pending tool calls found to reject."}

    rejection_feedback = request.feedback or "Action denied by administrator."
    rejection_messages = [
        ToolMessage(
            content=f"Rejected by Human Overseer: {rejection_feedback}",
            tool_call_id=tc["id"],
        )
        for tc in tool_calls
    ]

    await graph_app.aupdate_state(
        config, {"messages": rejection_messages}, as_node="tools"
    )

    return StreamingResponse(
        stream_langgraph_events(
            None, request.session_id, graph_app, historic_config=config
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
