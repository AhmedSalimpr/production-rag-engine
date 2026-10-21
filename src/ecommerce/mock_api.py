"""Mock e-commerce API service — customer profiles, orders, and product catalog."""

from typing import Dict, Any, List, Optional


DEMO_CUSTOMER: Dict[str, Any] = {
    "customer_id": "CUST-7291",
    "name": "Sarah Mitchell",
    "email": "sarah.mitchell@email.com",
    "tier": "Gold Member",
    "shipping_address": "742 Evergreen Terrace, Springfield, IL 62704",
    "phone": "+1 (217) 555-0142",
    "member_since": "2024-03-15",
    "loyalty_points": 2450,
    "preferred_payment": "Visa ending in 4821",
}

ORDERS: List[Dict[str, Any]] = [
    {
        "order_id": "ORD-88421",
        "date": "2026-10-18",
        "status": "shipped",
        "tracking_number": "1Z999AA10123456784",
        "carrier": "UPS",
        "estimated_delivery": "2026-10-23",
        "items": [
            {"name": "Sony WH-1000XM5 Wireless Headphones", "sku": "ELEC-HP-001", "qty": 1, "price": 348.00},
            {"name": "USB-C to 3.5mm Audio Adapter", "sku": "ACC-ADP-012", "qty": 1, "price": 12.99},
        ],
        "total": 360.99,
    },
    {
        "order_id": "ORD-87103",
        "date": "2026-10-02",
        "status": "delivered",
        "tracking_number": "1Z999AA10198765432",
        "carrier": "FedEx",
        "delivered_date": "2026-10-06",
        "items": [
            {"name": "Anker 65W GaN USB-C Charger", "sku": "ACC-CHG-003", "qty": 1, "price": 35.99},
            {"name": "Braided USB-C Cable 6ft (2-Pack)", "sku": "ACC-CBL-007", "qty": 1, "price": 14.99},
            {"name": "Philips Hue White Ambiance Starter Kit", "sku": "HOME-LGT-002", "qty": 1, "price": 99.99},
        ],
        "total": 150.97,
    },
    {
        "order_id": "ORD-85670",
        "date": "2026-09-14",
        "status": "delivered",
        "tracking_number": "9400111899223100345671",
        "carrier": "USPS",
        "delivered_date": "2026-09-19",
        "items": [
            {"name": "JBL Flip 6 Portable Bluetooth Speaker", "sku": "ELEC-SPK-002", "qty": 1, "price": 129.95},
        ],
        "total": 129.95,
    },
    {
        "order_id": "ORD-83200",
        "date": "2026-08-05",
        "status": "delivered",
        "tracking_number": "9261290100130435465558",
        "carrier": "USPS",
        "delivered_date": "2026-08-10",
        "items": [
            {"name": "TP-Link Kasa Smart Plug (4-Pack)", "sku": "HOME-PLG-001", "qty": 1, "price": 29.99},
            {"name": "Ring Indoor Cam (2nd Gen)", "sku": "HOME-CAM-003", "qty": 2, "price": 59.99},
        ],
        "total": 149.97,
    },
]


def get_customer_profile() -> Dict[str, Any]:
    """Returns the demo customer's profile."""
    return DEMO_CUSTOMER.copy()


def get_order_history() -> List[Dict[str, Any]]:
    """Returns the demo customer's full order history, most recent first."""
    return [order.copy() for order in ORDERS]


def get_order_by_id(order_id: str) -> Optional[Dict[str, Any]]:
    """Looks up a specific order by ID. Returns None if not found."""
    normalized = order_id.strip().upper()
    for order in ORDERS:
        if order["order_id"] == normalized:
            return order.copy()
    return None


def get_recent_order() -> Dict[str, Any]:
    """Returns the most recent order."""
    return ORDERS[0].copy()


def format_customer_summary(customer: Dict[str, Any], orders: List[Dict[str, Any]]) -> str:
    """Formats a human-readable customer summary for system prompt injection."""
    order_lines = []
    for o in orders:
        item_names = ", ".join(i["name"] for i in o["items"])
        order_lines.append(
            f"  • Order #{o['order_id']} ({o['date']}): {o['status'].upper()} "
            f"— ${o['total']:.2f} [{item_names}]"
        )
    order_block = "\n".join(order_lines)

    return (
        f"Customer: {customer['name']} ({customer['customer_id']})\n"
        f"Tier: {customer['tier']} | Loyalty Points: {customer['loyalty_points']}\n"
        f"Email: {customer['email']}\n"
        f"Shipping Address: {customer['shipping_address']}\n"
        f"Member Since: {customer['member_since']}\n"
        f"\nRecent Orders:\n{order_block}"
    )
