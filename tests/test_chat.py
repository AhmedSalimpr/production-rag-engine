"""Unit tests for multi-turn conversational RAG and query condensation."""

import pytest
from src.generation.condenser import QueryCondenser
from src.api.app import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_condenser_standalone_turn_one():
    condenser = QueryCondenser()
    query = "What is the 1:2 brew ratio for espresso?"
    result = condenser.condense(query, [])
    assert result == query


def test_condenser_resolves_pronoun_heuristic():
    condenser = QueryCondenser()
    history = [
        {"role": "user", "content": "What is the 1:2 brew ratio for espresso?"},
        {"role": "assistant", "content": "The golden ratio is 1:2, extracting 36g espresso from 18g coffee beans [1]."},
    ]
    followup = "Why is it important?"
    result = condenser.condense(followup, history)
    # Result should resolve 'it' with espresso / brew ratio
    assert "espresso" in result.lower() or "brew ratio" in result.lower()


def test_api_chat_multi_turn_flow():
    # Turn 1: Seeded knowledge base question (RRF)
    session_id = "test-session-rrf-1"
    res1 = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "What is the formula and smoothing constant for Reciprocal Rank Fusion?"},
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["session_id"] == session_id
    assert len(data1["answer"]) > 10
    assert data1["history_length"] == 2
    assert len(data1["citations"]) >= 1

    # Turn 2: Follow-up question with pronoun
    res2 = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "Why is it chosen over score averaging?"},
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["session_id"] == session_id
    assert data2["history_length"] == 4
    # Check that rewritten_query was formed
    assert len(data2["rewritten_query"]) > 5
    assert len(data2["answer"]) > 10
    assert len(data2["citations"]) >= 1

    # Turn 3: Check session history endpoint
    res3 = client.get(f"/api/chat/history/{session_id}")
    assert res3.status_code == 200
    assert res3.json()["count"] == 4

    # Turn 4: Reset session
    res4 = client.post(f"/api/chat/reset/{session_id}")
    assert res4.status_code == 200
    assert res4.json()["status"] == "reset"
