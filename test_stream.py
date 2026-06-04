import httpx
import time

url = "http://127.0.0.1:8000/api/chat"
session_id = "A_session_001"  # Định danh phiên chat cố định


def send_chat(prompt_text):
    print(f"\nUser: {prompt_text}")
    print("AI: ", end="", flush=True)
    data = {"prompt": prompt_text, "session_id": session_id}

    with httpx.stream("POST", url, json=data, timeout=60.0) as response:
        for chunk in response.iter_text():
            print(chunk, end="", flush=True)
    print()


# Lượt 1: Giới thiệu bản thân
send_chat("Chào bạn, tôi tên là A và tôi là một marketing person.")

# Nghỉ 2 giây rồi hỏi câu kiểm tra bộ nhớ
time.sleep(2)
send_chat("Tôi vừa nói tôi tên là gì và làm nghề gì ấy nhỉ?")
