"""
Quick test script to validate all API endpoints.
Run AFTER starting the server with uvicorn.
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def print_response(endpoint: str, response: dict):
    print(f"\n{'='*55}")
    print(f"  {endpoint}")
    print(f"{'='*55}")
    print(f"  Status : {response.status_code}")
    data = response.json()
    if "answer" in data:
        print(f"  Answer : {data['answer'][:200]}...")
        print(f"  Latency: {data.get('latency_ms')}ms")
    else:
        print(f"  Response: {json.dumps(data, indent=2)}")


# Test 1: Health check
r = requests.get(f"{BASE_URL}/health")
print_response("GET /health", r)

# Test 2: Ask a question
r = requests.post(f"{BASE_URL}/ask", json={
    "question": "What are the symptoms of the stroke patient?",
    "verbose": False
})
print_response("POST /ask", r)

# Test 3: Ask with verbose (shows context)
r = requests.post(f"{BASE_URL}/ask", json={
    "question": "Which patient has the lowest oxygen saturation?",
    "verbose": True
})
print_response("POST /ask (verbose)", r)

# Test 4: Chat with history
r = requests.post(f"{BASE_URL}/chat", json={
    "question": "What treatment was given to that patient?",
    "chat_history": [
        {"role": "user",      "content": "Tell me about patient P010"},
        {"role": "assistant", "content": "P010 is a stroke patient on tPA..."}
    ]
})
print_response("POST /chat", r)

# Test 5: Risk triage
r = requests.get(f"{BASE_URL}/risk")
print_response("GET /risk", r)

# Test 6: Patient summary
r = requests.get(f"{BASE_URL}/patient/P001")
print_response("GET /patient/P001", r)

# Test 7: Invalid patient ID
r = requests.get(f"{BASE_URL}/patient/INVALID")
print_response("GET /patient/INVALID (should 422)", r)

print("\n✅ All API tests complete!")
