import asyncio
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_redis import RedisSemanticCache, RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.outputs import Generation

app = FastAPI(title="LLM Local Chat with Redis Memory")

# 1. Khởi tạo LLM và Embedding
llm = OllamaLLM(model="llama3:8b")
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# 2. Khởi tạo Semantic Cache (Cổng 6380 chuẩn Docker của bạn)
REDIS_URL = "redis://localhost:6380"
LLM_IDENTIFIER = "ollama/llama3:8b"
try:
    semantic_cache = RedisSemanticCache(
        embeddings=embeddings_model, redis_url=REDIS_URL, distance_threshold=0.2
    )
    print("[SYSTEM] Khởi tạo Redis Semantic Cache thành công!")
except Exception as e:
    print(f"[SYSTEM] LỖI KHỞI TẠO CACHE: {str(e)}")

# 3. Cấu hình tầng Prompt và Memory bằng LangChain
# Định nghĩa template hướng dẫn AI cách hành xử và cấu trúc lịch sử chat
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Bạn là một trợ lý AI thông minh và lịch sự. Hãy trả lời dựa vào nội dung cuộc trò chuyện.",
        ),
        MessagesPlaceholder(
            variable_name="history"
        ),  # Nơi lịch sử chat từ Redis sẽ được nhét vào ngầm
        ("human", "{input}"),  # Câu hỏi mới của người dùng
    ]
)

# Kết nối Prompt với LLM tạo thành một Chain
chain = prompt_template | llm


def get_redis_history(session_id: str):
    """Hàm helper để LangChain tự động kết nối vào Redis lấy/ghi lịch sử theo session_id"""
    return RedisChatMessageHistory(session_id, redis_url=REDIS_URL)


# Bọc Chain lại bằng Bộ quản lý lịch sử tin nhắn toàn diện
chain_with_history = RunnableWithMessageHistory(
    chain, get_redis_history, input_messages_key="input", history_messages_key="history"
)


# 4. Định nghĩa cấu trúc Request từ Client
class ChatRequest(BaseModel):
    prompt: str
    session_id: str  # Thêm session_id để phân biệt các cuộc trò chuyện


async def generate_llm_stream(prompt: str, session_id: str):
    full_response = ""
    start_time = time.time()
    try:
        # Gọi luồng astream của chain_with_history
        # LangChain sẽ tự ngầm vào Redis bốc lịch sử ra trước khi gộp vào gửi cho Ollama
        async for chunk in chain_with_history.astream(
            {"input": prompt}, config={"configurable": {"session_id": session_id}}
        ):
            full_response += chunk
            yield chunk
            await asyncio.sleep(0.01)

        # Sau khi sinh chữ xong, lưu cặp (Prompt hiện tại, Kết quả) vào Semantic Cache để tối ưu lần sau
        semantic_cache.update(prompt, LLM_IDENTIFIER, [Generation(text=full_response)])
        print(f"[BACKEND LOG] Đã lưu câu trả lời vào Cache và cập nhật lịch sử chat.")

    except Exception as e:
        print(f"[BACKEND LOG] LỖI STREAM: {str(e)}")
        yield f"Error: {str(e)}"


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    print(
        f"\n[BACKEND LOG] Nhận request từ Session: '{request.session_id}' | Prompt: '{request.prompt}'"
    )

    # 1. Kiểm tra Semantic Cache trước
    cached_result = None
    try:
        cached_result = semantic_cache.lookup(request.prompt, LLM_IDENTIFIER)
    except Exception as e:
        print(f"[BACKEND LOG] LỖI LOOKUP CACHE: {str(e)}")

    if cached_result:
        print("[BACKEND LOG] HIT CACHE!")

        async def stream_cached_result():
            yield f"[HIT CACHE] {cached_result[0].text}"

        return StreamingResponse(stream_cached_result(), media_type="text/event-stream")

    # 2. MISS CACHE -> Chạy luồng có Memory
    print("[BACKEND LOG] MISS CACHE! Đang nạp lịch sử từ Redis và gọi Ollama...")
    return StreamingResponse(
        generate_llm_stream(request.prompt, request.session_id),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
