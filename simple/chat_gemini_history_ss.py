import requests

url = "http://127.0.0.1:8000/api/chat/history/session_4fb1a7"
# Create a random session for this chat session (or you can define your own string)

print("====================================================")
try:
    # 3. Execute the GET request
    print(f"Sending GET request to: {url}...\n")
    response = requests.get(url)

    # 4. Check if the request was successful (Status 200 OK)
    if response.status_code == 200:
        data = response.json()
        print("Successfully fetched LangGraph history:")
        print(f"Session ID: {data['session_id']}")

        # Loop through and print the history checkpoints
        for record in data["history"]:
            print("-" * 40)
            print(f"Checkpoint ID: {record['checkpoint_id']}")
            print(f"Next Executable Node: {record['next_executable_node']}")
            print(f"Has Summary: {record['has_summary']}")
            print(f"Message Count: {record['message_count']}")
            print(f"Last Message Preview: {record['last_message']}")
    else:
        print(f"Server returned an error status code: {response.status_code}")
        print(f"Response text: {response.text}")

except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server.")
    print("Make sure your FastAPI server is running at http://127.0.0.1:8000")

# the list will run from bottom to up

# Checkpoint ID: 1f16664f-c7d0-6a5b-8004-d31556e7d855
# Next Executable Node: []
# Has Summary: False
# Message Count: 4
# Last Message Preview: [{'type': 'text', 'text': 'Vietnam has a total area of approximately 331,210 square kilometers (about 127,881 square miles).', xxxx}]
# ----------------------------------------
# Checkpoint ID: 1f16664f-bfac-6852-8003-6651cf342305
# Next Executable Node: ['agent']
# Has Summary: False
# Message Count: 3
# Last Message Preview: what is Vietnam's area?
# ----------------------------------------
# Checkpoint ID: 1f16664f-bfa8-6d88-8002-ba207a3244cb
# Next Executable Node: ['__start__']
# Has Summary: False
# Message Count: 2
# Last Message Preview: [{'type': 'text', 'text': 'Hello! How can I help you today?', xxxx}]
# ----------------------------------------
# Checkpoint ID: 1f16664e-9814-6b45-8001-2ad9ebed7620
# Next Executable Node: []
# Has Summary: False
# Message Count: 2
# Last Message Preview: [{'type': 'text', 'text': 'Hello! How can I help you today?', xxxx}]
# ----------------------------------------
# Checkpoint ID: 1f16664e-9318-6977-8000-8b292952b374
# Next Executable Node: ['agent']
# Has Summary: False
# Message Count: 1
# Last Message Preview: hi
# ----------------------------------------
# Checkpoint ID: 1f16664e-9307-6ef3-bfff-a203a1e387ea
# Next Executable Node: ['__start__']
# Has Summary: False
# Message Count: 0
# Last Message Preview: None


# ┌──────────────────────────────┐
# │       Init: Empty State
# └──────────────┬───────────────┘
#                │
#                ▼
# 💾 Checkpoint: ...387ea
#                │
#                ▼
#        [User Input]
#                │
#                ▼
# ┌──────────────────────────────┐
# │  Execute: __start__          │
# └──────────────┬───────────────┘
#                │  (Outputs state changes)
#                ▼
#  💾 Checkpoint: ...b374  ◄─────── You are paused here.
#                │                  Next Executable Node: ['agent']
#                ▼
# ┌──────────────────────────────┐
# │  Execute: agent node         │
# └──────────────┬───────────────┘
#                │  (Outputs LLM response)
#                ▼
#  💾 Checkpoint: ...7620  ◄─────── You are paused here.
#                                   Next Executable Node: [] (END)
