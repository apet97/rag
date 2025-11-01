#!/usr/bin/env python3
"""
Test session-based conversational RAG flow.

Tests:
1. Create session
2. Send first question
3. Send follow-up question (verify context maintained)
4. Delete session
"""
import json
import sys
from datetime import datetime

import httpx

# Configuration
BASE_URL = "http://localhost:7001"
API_TOKEN = "change-me"

HEADERS = {
    "x-api-token": API_TOKEN,
    "Content-Type": "application/json"
}

def test_session_flow():
    """Test complete session flow."""
    print("\n" + "="*60)
    print("SESSION FLOW TEST")
    print("="*60)

    try:
        # Test 1: Create session
        print("\n[1] Creating new session...")
        response = httpx.post(
            f"{BASE_URL}/chat/session",
            headers=HEADERS,
            timeout=10.0
        )

        if response.status_code != 200:
            print(f"❌ Failed to create session: {response.status_code}")
            print(f"Response: {response.text}")
            return False

        session_data = response.json()
        session_id = session_data["session_id"]
        print(f"✅ Session created: {session_id}")
        print(f"   Created at: {session_data['created_at']}")
        print(f"   TTL: {session_data['ttl_seconds']}s")

        # Test 2: Send first question
        print("\n[2] Sending first question...")
        question1 = "How do I create a project in Clockify?"

        response = httpx.post(
            f"{BASE_URL}/chat/session/{session_id}",
            headers=HEADERS,
            json={"question": question1, "k": 5},
            timeout=30.0
        )

        if response.status_code != 200:
            print(f"❌ First question failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

        chat_data = response.json()
        print(f"✅ First question answered")
        print(f"   Question: {question1}")
        print(f"   Session info:")
        print(f"      - Turn: {chat_data['meta']['turn']}")
        print(f"      - Has history: {chat_data['meta']['has_conversation_history']}")
        print(f"      - Answerability: {chat_data['meta']['answerability_score']:.3f}")
        print(f"   Answer preview: {chat_data['answer'][:150]}...")

        # Test 3: Send follow-up question
        print("\n[3] Sending follow-up question...")
        question2 = "How do I add team members?"

        response = httpx.post(
            f"{BASE_URL}/chat/session/{session_id}",
            headers=HEADERS,
            json={"question": question2, "k": 5},
            timeout=30.0
        )

        if response.status_code != 200:
            print(f"❌ Follow-up question failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

        chat_data = response.json()
        print(f"✅ Follow-up question answered")
        print(f"   Question: {question2}")
        print(f"   Session info:")
        print(f"      - Turn: {chat_data['meta']['turn']}")
        print(f"      - Has history: {chat_data['meta']['has_conversation_history']}")
        print(f"      - Answerability: {chat_data['meta']['answerability_score']:.3f}")

        # Check if response mentions project context from Q1
        if "project" in chat_data['answer'].lower():
            print(f"   ✓ Response references project context from Q1")
        else:
            print(f"   ⚠ Response doesn't reference project context")

        print(f"   Answer preview: {chat_data['answer'][:150]}...")

        # Test 4: Get conversation history
        print("\n[4] Retrieving conversation history...")
        response = httpx.get(
            f"{BASE_URL}/chat/session/{session_id}",
            headers=HEADERS,
            timeout=10.0
        )

        if response.status_code != 200:
            print(f"❌ Failed to retrieve history: {response.status_code}")
            print(f"Response: {response.text}")
            return False

        history_data = response.json()
        turns = history_data.get("conversation", [])
        print(f"✅ Retrieved conversation history")
        print(f"   Total turns: {len(turns)}")
        for i, turn in enumerate(turns, 1):
            print(f"   Turn {i}: {turn['user_question'][:60]}...")

        # Test 5: Delete session
        print("\n[5] Deleting session...")
        response = httpx.delete(
            f"{BASE_URL}/chat/session/{session_id}",
            headers=HEADERS,
            timeout=10.0
        )

        if response.status_code != 200:
            print(f"❌ Failed to delete session: {response.status_code}")
            print(f"Response: {response.text}")
            return False

        print(f"✅ Session deleted")

        # Test 6: Verify session is gone
        print("\n[6] Verifying session is deleted...")
        response = httpx.get(
            f"{BASE_URL}/chat/session/{session_id}",
            headers=HEADERS,
            timeout=10.0
        )

        if response.status_code == 404:
            print(f"✅ Session properly deleted (404 as expected)")
        else:
            print(f"⚠ Expected 404, got {response.status_code}")
            return False

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_session_flow()
    sys.exit(0 if success else 1)
