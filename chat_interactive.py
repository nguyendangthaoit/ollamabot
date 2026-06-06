import httpx
import uuid

url = "http://127.0.0.1:8000/api/chat"
# Tạo một Session ngẫu nhiên cho lượt chat này (hoặc bạn tự đặt chuỗi bất kỳ)
session_id = f"session_{uuid.uuid4().hex[:6]}"

print("====================================================")
print(f" HỆ THỐNG CHAT LOCAL ĐÃ SẴN SÀNG (Session: {session_id})")
print(" Gõ 'exit' hoặc 'quit' để kết thúc cuộc trò chuyện.")
print("====================================================")

while True:
    user_input = input("\nBạn: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Tạm biệt!")
        break

    if not user_input.strip():
        continue

    print("AI: ", end="", flush=True)
    data = {"prompt": user_input, "session_id": session_id}

    try:
        # Gọi stream lên FastAPI
        with httpx.stream("POST", url, json=data, timeout=60.0) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    # Bóc tách chữ "data: " ra để lấy nội dung thật
                    content = line[6:]  # Lấy từ ký tự thứ 6 trở đi
                    # In ra màn hình (giữ nguyên end="" để các từ nối liền nhau)
                    print(content, end="", flush=True)

        print()  # Xuống dòng khi AI trả lời xong toàn bộ câu
    except Exception as e:
        print(f"\nLỗi kết nối Backend: {e}")
