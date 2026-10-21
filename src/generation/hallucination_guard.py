"""Hallucination detection and quote grounding verification guardrail."""

import re
from typing import List, Dict, Any, Tuple


class HallucinationGuard:
    """Verifies that generated citations and answer claims are factually grounded in context."""

    def verify_citations(
        self,
        citations: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Validates that each cited quote is an exact or near-exact substring of retrieved chunks."""
        if not citations:
            return [], 1.0

        all_chunk_text = " ".join([c.get("content", "").lower() for c in retrieved_chunks])
        # Clean whitespace
        all_chunk_text_normalized = re.sub(r"\s+", " ", all_chunk_text)

        verified_citations = []
        valid_count = 0

        for cit in citations:
            quote = cit.get("quote", "").strip()
            quote_lower = quote.lower()
            quote_normalized = re.sub(r"\s+", " ", quote_lower)

            is_grounded = False
            match_confidence = 0.0

            if quote_normalized and quote_normalized in all_chunk_text_normalized:
                is_grounded = True
                match_confidence = 1.0
                valid_count += 1
            elif len(quote_normalized) > 15:
                # Check sliding sub-phrase overlap
                words = quote_normalized.split()
                if len(words) >= 4:
                    sample_phrase = " ".join(words[:4])
                    if sample_phrase in all_chunk_text_normalized:
                        is_grounded = True
                        match_confidence = 0.85
                        valid_count += 1

            cit_copy = dict(cit)
            cit_copy["is_grounded"] = is_grounded
            cit_copy["match_confidence"] = match_confidence
            verified_citations.append(cit_copy)

        citation_precision = valid_count / len(citations) if citations else 1.0
        return verified_citations, round(citation_precision, 4)

    def evaluate_faithfulness(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> float:
        """Measures lexical and semantic grounding of the answer sentences in the source chunks."""
        if not answer.strip():
            return 0.0

        sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 10]
        if not sentences:
            return 1.0

        all_text = " ".join([c.get("content", "").lower() for c in retrieved_chunks])
        chunk_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", all_text))

        grounded_sentences = 0
        for sent in sentences:
            sent_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", sent.lower()))
            if not sent_words:
                continue
            overlap = sent_words.intersection(chunk_words)
            ratio = len(overlap) / len(sent_words)
            # A sentence is considered grounded if >= 60% of substantive words originate in context
            if ratio >= 0.60:
                grounded_sentences += 1

        faithfulness = grounded_sentences / len(sentences)
        return round(faithfulness, 4)
