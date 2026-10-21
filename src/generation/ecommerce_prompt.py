"""E-commerce customer support prompt templates with customer context injection."""

from typing import List, Dict, Any


def build_support_system_prompt(customer: Dict[str, Any], orders: List[Dict[str, Any]]) -> str:
    """Builds a system prompt that transforms the LLM into a TechHaven support agent."""
    order_lines = []
    for o in orders:
        item_names = ", ".join(i["name"] for i in o["items"])
        status_detail = o["status"].upper()
        if o["status"] == "shipped":
            status_detail += f" (Tracking: {o['tracking_number']}, ETA: {o.get('estimated_delivery', 'N/A')})"
        elif o["status"] == "delivered":
            status_detail += f" (Delivered: {o.get('delivered_date', 'N/A')})"
        order_lines.append(
            f"  • Order #{o['order_id']} ({o['date']}): {status_detail}\n"
            f"    Items: {item_names} — Total: ${o['total']:.2f}"
        )
    order_block = "\n".join(order_lines)

    return f"""You are a friendly, professional customer support assistant for TechHaven, an online electronics and accessories store.

CUSTOMER CONTEXT (loaded from CRM — treat as ground truth for this customer):
- Name: {customer['name']}
- Account: {customer['customer_id']} ({customer['tier']})
- Email: {customer['email']}
- Phone: {customer.get('phone', 'N/A')}
- Shipping Address: {customer['shipping_address']}
- Loyalty Points: {customer['loyalty_points']}
- Member Since: {customer['member_since']}
- Preferred Payment: {customer.get('preferred_payment', 'N/A')}

RECENT ORDERS:
{order_block}

CRITICAL GROUNDING & CITATION RULES:
1. For PRODUCT questions (search, recommendations, comparisons): Answer ONLY from the retrieved product catalog context chunks. Every factual product claim MUST include inline citations [1], [2] with verbatim quotes.
2. For ORDER questions (tracking, status, history): Use the CUSTOMER CONTEXT above. No need for citations since it's injected CRM data.
3. For POLICY questions (returns, shipping, warranty, loyalty): Answer from the retrieved policy/FAQ context chunks with citations.
4. Be warm, concise, and helpful. Use the customer's first name ("{customer['name'].split()[0]}").
5. When recommending products, always mention the product name, price, and key differentiator. Base recommendations on catalog context.
6. Never fabricate product details, prices, inventory status, or order information.
7. If the retrieved context doesn't contain an answer, clearly say so and offer to connect with a human agent."""


def build_support_user_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Combines user query with grounded product/policy context for support chat."""
    context_lines = []
    for idx, c in enumerate(chunks, start=1):
        source = c.get("source_title", "TechHaven Catalog")
        section = c.get("section_title", "General")
        cid = c.get("chunk_id", f"chk-{idx}")
        content = c.get("content", "").strip()

        context_lines.append(
            f"--- [CHUNK {idx}] ---\n"
            f"Document: {source}\n"
            f"Section: {section}\n"
            f"Chunk ID: {cid}\n"
            f"Content:\n{content}\n"
        )
    context_text = "\n".join(context_lines)

    return (
        f"Retrieved Product & Policy Context:\n{context_text}\n\n"
        f"Customer Question:\n{query}\n\n"
        "Provide a helpful, grounded answer with numbered citations [1], [2] referencing the context chunks. "
        "Extract the exact verbatim quote for each citation."
    )
