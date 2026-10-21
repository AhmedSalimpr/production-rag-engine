"""Query condenser and reformulation engine for conversational multi-turn RAG."""

import re
from typing import List, Dict, Optional
from src.config import settings


CONDENSE_SYSTEM_PROMPT = """You are a query reformulation assistant for a technical Retrieval-Augmented Generation (RAG) engine.
Your task is: Given a conversation history and a follow-up user message, rewrite the follow-up into a complete, standalone search query.
- Resolve any pronouns ("it", "they", "this", "these", "that", "those", "the latter", "the former") using the entities mentioned in previous messages.
- Preserve technical keywords, acronyms, and constraints.
- Output ONLY the reformulated standalone query. Do not add introductory remarks, explanations, or quotes.
- If the follow-up message is already a complete, standalone question, return it unchanged.
"""


class QueryCondenser:
    """Reformulates multi-turn conversational queries into standalone retrieval queries."""

    def __init__(self):
        pass

    def condense(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Transforms a conversational follow-up into a standalone retrieval query."""
        message = user_message.strip()
        if not chat_history or len(chat_history) == 0:
            return message

        # Filter to recent exchanges (last 3 turns max)
        recent_history = [
            turn for turn in chat_history[-6:]
            if turn.get("role") in ("user", "assistant") and turn.get("content")
        ]
        if not recent_history:
            return message

        # 1. Try LLM reformulation if API is configured
        if settings.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                )

                history_text = "\n".join(
                    f"{turn['role'].capitalize()}: {turn['content'][:250]}"
                    for turn in recent_history
                )

                prompt = (
                    f"Chat History:\n{history_text}\n\n"
                    f"Follow-up Question: {message}\n\n"
                    f"Standalone Search Query:"
                )

                response = client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=80,
                )

                condensed = response.choices[0].message.content.strip()
                # Clean any outer quotes
                condensed = re.sub(r'^["\']|["\']$', '', condensed).strip()
                if condensed and len(condensed) > 4:
                    return condensed
            except Exception as e:
                print(f"[Warning] Query condenser LLM call failed ({e}); using heuristic reformulation.")

        # 2. Heuristic fallback for offline execution
        return self._heuristic_reformulate(message, recent_history)

    def _heuristic_reformulate(self, message: str, history: List[Dict[str, str]]) -> str:
        """Deterministic heuristic pronoun resolver for offline or rate-limited environments."""
        pronoun_pattern = re.compile(r'\b(it|its|this|that|these|those|they|them|the former|the latter)\b', re.IGNORECASE)
        has_pronoun = bool(pronoun_pattern.search(message))
        is_short_followup = len(message.split()) <= 6

        if not has_pronoun and not is_short_followup:
            return message

        # Find the primary subject from the most recent assistant and user messages
        subject = ""
        for turn in reversed(history):
            content = turn.get("content", "")
            # Look for common technical capitalized phrases or quoted terms
            quoted = re.findall(r'["\']([^"\']+)["\']', content)
            if quoted:
                subject = quoted[0]
                break

            # Or take title/header phrases if present
            headers = re.findall(r'#+\s+([^\n]+)', content)
            if headers:
                subject = headers[0]
                break

            # Or common noun phrases like 'espresso', 'Raft', 'BM25', 'RRF', 'HNSW'
            for kw in ["espresso", "brew ratio", "Raft", "leader election", "BM25", "Reciprocal Rank Fusion", "RRF", "HNSW", "LoRA", "quantization", "chunking"]:
                if kw.lower() in content.lower():
                    subject = kw
                    break
            if subject:
                break

        if not subject:
            # Fallback to the first query in history
            for turn in history:
                if turn.get("role") == "user":
                    first_words = turn.get("content", "").split()[:4]
                    subject = " ".join(first_words)
                    break

        if subject:
            if has_pronoun:
                # Replace pronouns with the subject
                reformulated = pronoun_pattern.sub(subject, message)
                return reformulated
            else:
                return f"{subject} {message}"

        return message
