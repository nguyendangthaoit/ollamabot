import traceback
from contextlib import asynccontextmanager
from typing import Annotated, Any, TypedDict
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel


# 1. Quản lý vòng đời kết nối Redis thông qua Lifespan (Sửa lỗi context manager)
@asynccontextmanager
async def lifespan(app: FastAPI):
    REDIS_URL = "redis://localhost:6380"
    print("[SYSTEM] Đang kết nối và khởi tạo Redis Checkpointer...")

    try:
        # 1. Khởi tạo Context Manager (đây là một Sync Context Manager)
        redis_context = RedisSaver.from_conn_string(REDIS_URL)

        # 2. Sử dụng 'with' thuần túy để bóc tách thực thể 'saver' ra
        with redis_context as saver:
            print(
                "[SYSTEM] Khởi tạo indices và cấu hình Redis Checkpointer THÀNH CÔNG!"
            )

            # Lưu đối tượng checkpointer thực sự (đã đúng kiểu dữ liệu) vào state ứng dụng
            app.state.checkpointer = saver

            # 3. Biên dịch đồ thị (Graph Compile) duy nhất 1 lần tại đây
            workflow = StateGraph(State)
            workflow.add_node("agent", call_model)
            workflow.add_edge(START, "agent")
            workflow.add_edge("agent", END)

            # Lúc này saver đã mang kiểu dữ liệu chuẩn (BaseCheckpointSaver), Pylance sẽ không báo đỏ nữa
            app.state.graph_app = workflow.compile(checkpointer=saver)

            # 4. Giữ luồng của FastAPI tại đây để lắng nghe API Request
            # Kết nối Redis sẽ ĐƯỢC GIỮ NGUYÊN vì block 'with' chưa kết thúc
            yield

        # Khi bạn bấm tắt Server (Ctrl + C), FastAPI đi qua lệnh yield,
        # block 'with' sẽ tự động đóng kết nối Redis một cách an toàn tại đây.
        print("[SYSTEM] Đang đóng kết nối Redis Checkpointer an toàn...")

    except Exception as env_err:
        print(f"[SYSTEM CRITICAL LỖI KHỞI TẠO REDIS]: {traceback.format_exc()}")
        raise env_err


app = FastAPI(title="Modern LangGraph Chat with Lifespan", lifespan=lifespan)

# 2. Khởi tạo mô hình LLM
llm = ChatOllama(model="llama3:8b")


# 3. Định nghĩa Cấu trúc Trạng thái (State)
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 4. Định nghĩa Node xử lý LLM
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Bạn là một trợ lý AI thông minh. Hãy dựa vào lịch sử trò chuyện để phản hồi một cách logic.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def call_model(state: State):
    print("[NODE] Thực thi Node Agent...")
    chain = prompt_template | llm
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


# 5. Cấu trúc Request từ Client
class ChatRequest(BaseModel):
    prompt: str
    session_id: str


# 6. Luồng xử lý Generator đồng bộ hỗ trợ Token Streaming từ LangGraph
def stream_langgraph_response(prompt: str, session_id: str, graph_app: Any):
    try:
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        input_data: Any = {"messages": [HumanMessage(content=prompt)]}

        print(f"[BACKEND LOG] Đang chạy stream đồ thị cho Thread: {session_id}...")

        # Đổi stream_mode sang "messages" để hứng TỪNG TOKEN (từ ngữ) sinh ra từ LLM
        for msg, metadata in graph_app.stream(
            input=input_data, config=config, stream_mode="messages"
        ):
            # Chỉ lấy các token được sinh ra từ Node 'agent' (tránh lấy lại câu hỏi của user)
            if metadata.get("langgraph_node") == "agent":
                if hasattr(msg, "content") and msg.content:
                    # In ra terminal để debug xem token chạy
                    print(msg.content, end="", flush=True)
                    # Yield dữ liệu về cho StreamingResponse dưới dạng chuẩn Event-Stream
                    yield f"data: {msg.content}\n\n"

    except Exception as e:
        print("\n" + "=" * 50)
        print("[CRITICAL ERROR IN STREAM]:")
        print(traceback.format_exc())
        print("=" * 50 + "\n")
        yield "data: Error details: Check Backend Terminal\n\n"


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, fastapi_req: Request):
    print(f"\n[BACKEND LOG] Nhận Request từ Session: {request.session_id}")

    # Lấy đối tượng Graph đã được compile sẵn từ app.state
    graph_app = fastapi_req.app.state.graph_app

    # Trả về Event-Stream trực tiếp cho Client
    return StreamingResponse(
        stream_langgraph_response(request.prompt, request.session_id, graph_app),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    # Chạy ứng dụng trực tiếp bằng uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
