import traceback
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from states import RouteDecision, State
from tools import tools_list
from rag.retrieve import query_knowledge_base

# Initialize Models
llm_primary = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", streaming=True
).bind_tools(tools_list)

# llm_assistant = ChatOllama(model="llama3.1")
llm_assistant = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

llm_support = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_context}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


async def call_model(state: State):
    print("[GRAPH NODE] Entering Agent call_model Node...")

    # 1. Re-initialize your original chain / template variables
    chain = prompt_template | llm_primary

    # 2. Extract long-term memory summary from state
    existing_summary = state.get("summary", "")

    # 3. Build your original baseline system context string
    system_context = (
        "You are an intelligent AI assistant. Respond naturally in English."
    )
    if existing_summary:
        system_context += (
            f" Summary of previous conversation context: {existing_summary}"
        )

    # 4. Extract your RAG context and append it directly to the system_context
    retrieved_context = state.get("retrieved_context", "")
    if retrieved_context:
        system_context += (
            f"\n\nUse ONLY the following verified database records to answer the user query accurately. "
            f"If the answer cannot be found in these records, state clearly that you do not possess "
            f"that information.\n"
            f"--- CORPORATE POLICY RECORDS ---\n{retrieved_context}\n--------------------------------"
        )

    # 5. Execute your original native chain call pattern asynchronously
    response = await chain.ainvoke(
        {"system_context": system_context, "messages": state["messages"]}
    )
    return {"messages": [response]}


async def summarize_conversation_async(state: State):
    """Compresses oldest messages into text and issues deletion commands."""
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

    new_summary = await llm_assistant.ainvoke(summary_prompt)
    print(f"[MEMORY SYSTEM] Updated Summary: {new_summary.content}")

    delete_messages = [
        RemoveMessage(id=str(m.id)) for m in messages_to_summarize if m.id is not None
    ]

    return {"summary": str(new_summary.content), "messages": delete_messages}


async def background_memory_check(session_id: str, graph_app):
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}
    try:
        state_snapshot = await graph_app.aget_state(config)
        current_state = state_snapshot.values
        messages = current_state.get("messages", [])

        if len(messages) >= 6:
            print(
                f"\n[BACKGROUND MEMORY] Triggering background summary for session: {session_id}"
            )
            summary_results = await summarize_conversation_async(current_state)

            if summary_results:
                await graph_app.aupdate_state(
                    config, summary_results, as_node="summarize_node"
                )
                print(
                    "[BACKGROUND MEMORY] Background summary saved successfully to Redis!"
                )
    except Exception:
        print(f"[BACKGROUND MEMORY ERROR]: {traceback.format_exc()}")


# --- 2. THE CONDITIONAL ROUTING FUNCTION ---
def route_question(state: State):
    """
    Evaluates the user query and routes it to the correct path.
    """
    print("[ROUTER] Analyzing query intent...")
    user_query = state["messages"][-1].content

    # Force the LLM to output structured JSON matching our Pydantic schema
    structured_llm = llm_assistant.with_structured_output(RouteDecision)

    system_prompt = (
        "You are an incoming request router. Analyze the user's input and determine "
        "if they are asking about specific rule, policy, company documents, Work Schedules & Hybrid Model, Leave and Time-Off Allocations..."
    )

    decision = structured_llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=str(user_query))]
    )
    print(f"[ROUTER] Route determined: {decision.next_node}")  # type: ignore
    return decision.next_node  # type: ignore


async def retrieve_node(state: State):
    print("\n[GRAPH NODE] Entering Retrieve Node...")

    # 1. Safely pull the user's latest message text from the graph state
    messages = state.get("messages", [])
    if not messages:
        print("[WARN] No messages found in state to search with.")
        return {"retrieved_context": "No search context available."}

    user_query = messages[-1].content

    # 2. Call your separate retrieval layer function (Requesting top 2 matches for context richness)
    # Since your query function is synchronous, we run it normally.
    context_data = query_knowledge_base(user_query=str(user_query), top_k=2)

    # 3. Return the payload to update the graph state.
    # This automatically syncs the "retrieved_context" key for the next node!
    return {"retrieved_context": context_data}
