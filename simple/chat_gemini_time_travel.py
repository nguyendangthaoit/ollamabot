import httpx
import uuid

url = "http://127.0.0.1:8000/api/chat/time-travel"
# Create a random session for this chat session (or you can define your own string)


print("====================================================")
print(f" LOCAL CHAT SYSTEM READY (Session: session_4fb1a7)")
print(" Type 'exit' or 'quit' to end the conversation.")
print("====================================================")

while True:
    user_checkpointID = input("\nYour checkpoint ID: ")
    user_input = input("Your new prompt: ")

    if user_checkpointID.strip().lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not user_checkpointID.strip():
        continue

    print("AI: ", end="", flush=True)
    data = {
        "checkpoint_id": user_checkpointID,
        "session_id": "session_4fb1a7",
        "user_prompt_override": user_input,
    }

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
