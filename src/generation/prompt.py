"""Prompt engineering templates for grounded generation with citation constraints."""

from typing import List, Dict, Any


SYSTEM_PROMPT_GROUNDED = """You are a Principal AI Systems Engineer and grounded technical assistant.
Your answers should be derived from the provided retrieved context chunks whenever relevant.

CRITICAL CITATION & GROUNDING RULES:
1. When answering from the provided context chunks, every factual statement MUST include one or more inline footnote citations, e.g. [1], [2], with the exact verbatim quote in your structured citations list.
2. If the user's question CANNOT be answered from the provided knowledge base context:
   - State clearly: "The provided knowledge base documents do not contain information about this topic."
   - Then, provide a helpful answer using your general parametric knowledge (e.g. "However, based on general knowledge: [answer]").
   - In this case, leave the `citations` list empty ([]), because no context chunks were used.
3. Never fabricate or attribute outside general knowledge to the context chunks."""


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks with clear numbered IDs and metadata for the LLM."""
    context_lines = []
    for idx, c in enumerate(chunks, start=1):
        source = c.get("source_title", "Document")
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
    return "\n".join(context_lines)


def build_user_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Combines user query with grounded context blocks."""
    context_text = build_context_block(chunks)
    return (
        f"Retrieved Context:\n{context_text}\n\n"
        f"User Query:\n{query}\n\n"
        "Provide your grounded answer with numbered citations [1], [2] and extract the exact verbatim quote for each."
    )
