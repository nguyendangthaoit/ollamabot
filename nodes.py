import traceback
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from states import State
from tools import tools_list

# Initialize Models
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", streaming=True).bind_tools(
    tools_list
)
llmOllama = ChatOllama(model="llama3.1")

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

    new_summary = await llmOllama.ainvoke(summary_prompt)
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
