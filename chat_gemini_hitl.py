import httpx

print("====================================================")
print(" Type 'exit' or 'quit' to end the conversation.")
print("====================================================")

while True:
    action = input("\nAction 1:approve or 0:reject: ")
    print("action", action)
    session_id = input("Session_id: ")
    if action.strip().lower() not in ["1", "0"]:
        print("Goodbye!")
        break

    if not session_id.strip():
        continue

    print("AI: ", end="", flush=True)
    data = {"session_id": session_id}
    url = f"http://127.0.0.1:8000/api/chat/{'approve' if action == '1' else 'reject'}"
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
