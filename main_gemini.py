import os
import traceback
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Annotated, Any, TypedDict
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
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
            app.state.graph_app = workflow.compile(checkpointer=saver)

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
