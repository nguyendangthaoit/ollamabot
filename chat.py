import httpx
import uuid
import ast

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
        # Open an HTTP connection stream with standard timeouts
        with httpx.stream("POST", url, json=data, timeout=60.0) as response:
            # 2. Force it to split chunks strictly by line breaks
            for line in response.iter_lines():
                if line.startswith("data: "):
                    # Extract text payload
                    token = line[6:]

                    # 3. Print out immediately
                    print(token, end="", flush=True)

    except Exception as e:
        print(f"\n[Backend connection error]: {e}")
