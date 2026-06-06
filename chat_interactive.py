import httpx
import uuid

url = "http://127.0.0.1:8000/api/chat"
# Create a random session for this chat session (or you can define your own string)
session_id = f"session_{uuid.uuid4().hex[:6]}"

print("====================================================")
print(f" LOCAL CHAT SYSTEM READY (Session: {session_id})")
print(" Type 'exit' or 'quit' to end the conversation.")
print("====================================================")

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not user_input.strip():
        continue

    print("AI: ", end="", flush=True)
    data = {"prompt": user_input, "session_id": session_id}

    try:
        # Call FastAPI with streaming response
        with httpx.stream("POST", url, json=data, timeout=60.0) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    # Remove "data: " prefix to get actual content
                    content = line[6:]  # Take everything after character 6
                    # Print to screen (keep end="" so text flows naturally)
                    print(content, end="", flush=True)

        print()  # New line after AI finishes responding
    except Exception as e:
        print(f"\nBackend connection error: {e}")
