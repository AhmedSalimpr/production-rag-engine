"""Unit tests for e-commerce support chatbot functionality."""

import pytest
from fastapi.testclient import TestClient
from src.ecommerce.mock_api import (
    get_customer_profile,
    get_order_history,
    get_order_by_id,
    get_recent_order,
    format_customer_summary,
)
from src.generation.ecommerce_prompt import build_support_system_prompt
from src.api.app import app


client = TestClient(app)


def test_customer_profile():
    customer = get_customer_profile()
    assert customer["customer_id"] == "CUST-7291"
    assert customer["name"] == "Sarah Mitchell"
    assert customer["tier"] == "Gold Member"
    assert customer["loyalty_points"] == 2450
    assert "email" in customer


def test_order_history_and_lookup():
    orders = get_order_history()
    assert len(orders) == 4
    
    # Test most recent order
    recent = get_recent_order()
    assert recent["order_id"] == "ORD-88421"
    assert recent["status"] == "shipped"

    # Test lookup by ID (case-insensitive)
    order = get_order_by_id("ord-88421")
    assert order is not None
    assert order["carrier"] == "UPS"
    assert len(order["items"]) == 2

    # Non-existent order
    assert get_order_by_id("ORD-NONEXISTENT") is None


def test_system_prompt_builder():
    customer = get_customer_profile()
    orders = get_order_history()
    prompt = build_support_system_prompt(customer, orders)

    assert "Sarah Mitchell" in prompt
    assert "Gold Member" in prompt
    assert "ORD-88421" in prompt
    assert "TechHaven" in prompt
    assert "POLICY" in prompt
    assert "returns" in prompt


def test_api_support_customer_endpoint():
    res = client.get("/api/support/customer")
    assert res.status_code == 200
    data = res.json()
    assert "customer" in data
    assert "orders" in data
    assert data["customer"]["name"] == "Sarah Mitchell"
    assert len(data["orders"]) == 4


def test_api_support_order_lookup_endpoint():
    res = client.get("/api/support/order/ORD-88421")
    assert res.status_code == 200
    order = res.json()
    assert order["order_id"] == "ORD-88421"
    assert order["status"] == "shipped"

    # 404 for unknown order
    res_404 = client.get("/api/support/order/UNKNOWN-999")
    assert res_404.status_code == 404


def test_api_support_chat_query():
    res = client.post(
        "/api/support/chat",
        json={
            "session_id": "test_support_sess_1",
            "message": "Where is my latest order and when will it arrive?",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "test_support_sess_1"
    assert data["customer_name"] == "Sarah Mitchell"
    assert len(data["answer"]) > 10
    assert "retrieved_chunks" in data
    assert data["total_latency_ms"] > 0
