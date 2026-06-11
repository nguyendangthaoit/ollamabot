import traceback
from contextlib import asynccontextmanager
from typing import Annotated, Any, TypedDict
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# =====================================================================
# 1. DEFINE TOOLS (Core Python Functions for your Agent)
# =====================================================================


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


# =====================================================================
# 2. STATE & APPLICATION INITIALIZATION (Lifespan)
# =====================================================================


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str  # stores long-term memory summary


# Initialize LLM model
llm = ChatOllama(model="llama3.1")

# CRITICAL STEP: Bind the tool structures directly to the Ollama model instance
llm_with_tools = llm.bind_tools(tools_list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    REDIS_URL = "redis://localhost:6380"
    print("[SYSTEM] Connecting and initializing Redis Checkpointer...")

    try:
        # 1. Instantiate the sync Redis saver context manager
        redis_context = RedisSaver.from_conn_string(REDIS_URL)

        # 2. Open the connection context cleanly
        with redis_context as saver:
            # Crucial: Call setup() explicitly to make sure RediSearch and registries are fully mapped
            saver.setup()
            print(
                "[SYSTEM] Successfully initialized indices and Redis Checkpointer configuration!"
            )

            # 3. Compile the graph directly using the open saver instance
            workflow = StateGraph(State)
            workflow.add_node("agent", call_model)
            workflow.add_node("tools", tools_node)
            workflow.add_node("summarize_node", summarize_conversation)

            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges("agent", tools_condition)
            workflow.add_edge("tools", "agent")

            # Store the compiled app cleanly inside the global state
            app.state.graph_app = workflow.compile(checkpointer=saver)

            # 4. Freeze execution here while keeping the 'with' block active for streaming operations
            yield

        print("[SYSTEM] Closing Redis Checkpointer safely...")

    except Exception as env_err:
        print(f"[SYSTEM CRITICAL ERROR INITIALIZING REDIS]: {traceback.format_exc()}")
        raise env_err


app = FastAPI(title="Modern LangGraph Agent with Lifespan & Tools", lifespan=lifespan)


# =====================================================================
# 3. GRAPH NODE ACTIONS
# =====================================================================

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_context}"),  # Injected dynamically here
        MessagesPlaceholder(variable_name="messages"),
    ]
)

MODEL_CHAIN = prompt_template | llm_with_tools


def call_model(state: State):
    existing_summary = state.get("summary", "")

    # Clean, direct instructions. No mention of specific tools to prevent verbal leakage.
    system_context = (
        "You are a helpful, conversational AI assistant. Respond naturally and directly to the user.\n"
        "CRITICAL DIRECTIVE: Speak only to the user's intent. Never mention your tools, functions, "
        "or capabilities to the user. Do not explain why you are or are not using a tool. "
        "If a question can be answered with your general knowledge, answer it immediately and concisely."
    )

    if existing_summary:
        system_context += f"\nContext of previous conversation: {existing_summary}"

    response = MODEL_CHAIN.invoke(
        {"system_context": system_context, "messages": state["messages"]}
    )
    return {"messages": [response]}


def summarize_conversation(state: State):
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

    new_summary = llm.invoke(summary_prompt)
    print(f"[MEMORY SYSTEM] Updated Summary: {new_summary}")

    delete_messages = [
        RemoveMessage(id=str(m.id)) for m in messages_to_summarize if m.id is not None
    ]

    return {"summary": str(new_summary), "messages": delete_messages}


# =====================================================================
# 4. ENDPOINTS & BACKGROUND ASYNC OPERATIONS
# =====================================================================


class ChatRequest(BaseModel):
    prompt: str
    session_id: str


def stream_langgraph_response(prompt: str, session_id: str, graph_app: Any):
    try:
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        input_data: Any = {"messages": [HumanMessage(content=prompt)]}

        print(
            f"[BACKEND LOG] Running multi-mode graph stream for Thread: {session_id}..."
        )

        # We request BOTH token-level (messages) and node-level (updates) events
        for chunk_type, chunk_data in graph_app.stream(
            input=input_data, config=config, stream_mode=["messages", "updates"]
        ):

            # 1. Stream RAW text tokens immediately as Ollama types them out!
            if chunk_type == "messages":
                msg_chunk, metadata = chunk_data
                # Only stream text coming from the core AI agent node
                if metadata.get("langgraph_node") == "agent":
                    if hasattr(msg_chunk, "content") and msg_chunk.content:
                        yield f"data: {msg_chunk.content}\n\n"

            # 2. Watch for node structural changes to catch tool executions
            elif chunk_type == "updates":
                if "agent" in chunk_data:
                    last_msg = chunk_data["agent"]["messages"][-1]
                    # If the completed agent step requested a tool, log it on backend
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(
                            f"[BACKEND] Tool execution requested: {last_msg.tool_calls[0]['name']}"
                        )
                        yield f"data: \n[System: Executing tool parameters...]\n\n"

                elif "tools" in chunk_data:
                    print("[BACKEND] Tool node execution completed successfully.")

    except Exception as e:
        print("\n" + "=" * 50)
        print("[CRITICAL ERROR IN STREAM]:")
        print(traceback.format_exc())
        print("=" * 50 + "\n")
        yield "data: Error details: Check Backend Terminal\n\n"


async def background_memory_check(session_id: str, graph_app: Any):
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    # 1. Get current status of session from Redis
    state_snapshot = graph_app.get_state(config)
    current_state = state_snapshot.values
    messages = current_state.get("messages", [])

    # 2. If the message count crosses the threshold, compress memory via background job
    if len(messages) >= 6:
        print(
            f"\n[BACKGROUND MEMORY] Triggering background summary for session: {session_id}"
        )

        summary_results = summarize_conversation(current_state)

        if summary_results:
            # 3. Overwrite state (include new Summary and RemoveMessage flags) into Redis
            graph_app.update_state(config, summary_results, as_node="summarize_node")
            print("[BACKGROUND MEMORY] Background summary saved successfully!")


@app.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest, fastapi_req: Request, background_tasks: BackgroundTasks
):
    print(f"\n[BACKEND LOG] Received request from Session: {request.session_id}")
    graph_app = fastapi_req.app.state.graph_app

    # Inject background task for pruning memory
    background_tasks.add_task(background_memory_check, request.session_id, graph_app)

    return StreamingResponse(
        stream_langgraph_response(request.prompt, request.session_id, graph_app),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
