import httpx
import time

url = "http://127.0.0.1:8000/api/chat"
session_id = "A_session_001"  # Fixed chat session identifier


def send_chat(prompt_text):
    print(f"\nUser: {prompt_text}")
    print("AI: ", end="", flush=True)
    data = {"prompt": prompt_text, "session_id": session_id}

    with httpx.stream("POST", url, json=data, timeout=60.0) as response:
        for chunk in response.iter_text():
            print(chunk, end="", flush=True)
    print()


# Turn 1: Introduce yourself
send_chat("Hello, my name is A and I am a marketing person.")

# Wait 2 seconds, then ask a memory check question
time.sleep(2)
send_chat("What did I just say my name and job were?")
