"""Grounded answer generator with structured Pydantic output and citations."""

import time
import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.config import settings
from src.generation.prompt import SYSTEM_PROMPT_GROUNDED, build_user_prompt
from src.generation.hallucination_guard import HallucinationGuard


class Citation(BaseModel):
    citation_index: int
    source_id: str
    source_title: str
    section_title: str
    quote: str
    is_grounded: bool = True
    match_confidence: float = 1.0


class RAGResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    pipeline_trace: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = 1.0
    faithfulness_score: float = 1.0
    citation_precision: float = 1.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class GroundedGenerator:
    """Generates strictly grounded answers with citations and hallucination guards."""

    def __init__(self, guard: Optional[HallucinationGuard] = None):
        self.guard = guard or HallucinationGuard()

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        retrieval_latency_ms: float = 0.0,
        prior_stages: Optional[List[Dict[str, Any]]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt_override: Optional[str] = None,
    ) -> RAGResponse:
        """Executes LLM generation with citation extraction and validation."""
        start_time = time.perf_counter()
        trace = list(prior_stages) if prior_stages else []

        if not chunks:
            gen_latency = (time.perf_counter() - start_time) * 1000
            trace.append({
                "stage": "Structured Generation",
                "status": "warning",
                "latency_ms": round(gen_latency, 2),
                "details": {"message": "No candidates provided for generation"}
            })
            return RAGResponse(
                query=query,
                answer="No relevant documents were retrieved from the knowledge base to answer this query.",
                citations=[],
                retrieved_chunks=[],
                pipeline_trace=trace,
                confidence_score=0.0,
                faithfulness_score=1.0,
                citation_precision=1.0,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=round(gen_latency, 2),
                total_latency_ms=round(retrieval_latency_ms + gen_latency, 2),
            )

        # Call OpenAI / TokenRouter if API key configured
        provider_used = "Offline Grounded Engine"
        if settings.openai_api_key:
            try:
                raw_res = self._call_openai(query, chunks, chat_history, system_prompt_override)
                answer = raw_res.get("answer", "")
                raw_citations = raw_res.get("citations", [])
                provider_used = f"TokenRouter Gateway ({settings.llm_model})"
            except Exception as e:
                print(f"[Warning] TokenRouter/OpenAI call failed ({e}); using offline grounded generator.")
                answer, raw_citations = self._offline_grounded_answer(query, chunks)
        else:
            answer, raw_citations = self._offline_grounded_answer(query, chunks)

        # Hallucination Guard verification
        t_guard = time.perf_counter()
        verified_citations, citation_precision = self.guard.verify_citations(raw_citations, chunks)
        faithfulness = self.guard.evaluate_faithfulness(answer, chunks)
        guard_latency = (time.perf_counter() - t_guard) * 1000

        # Calculate overall confidence
        confidence = round((citation_precision * 0.6) + (faithfulness * 0.4), 3)
        gen_latency = (time.perf_counter() - start_time) * 1000

        structured_citations = [
            Citation(
                citation_index=c.get("citation_index", i + 1),
                source_id=c.get("source_id", "doc"),
                source_title=c.get("source_title", "Technical Reference"),
                section_title=c.get("section_title", "Overview"),
                quote=c.get("quote", ""),
                is_grounded=c.get("is_grounded", True),
                match_confidence=c.get("match_confidence", 1.0),
            )
            for i, c in enumerate(verified_citations)
        ]

        trace.append({
            "stage": "Structured Generation",
            "status": "completed",
            "latency_ms": round(gen_latency - guard_latency, 2),
            "details": {
                "provider": provider_used,
                "base_url": settings.openai_base_url,
                "model": settings.llm_model,
                "citations_extracted": len(structured_citations),
            }
        })

        trace.append({
            "stage": "Hallucination & Quote Guard",
            "status": "completed",
            "latency_ms": round(guard_latency, 2),
            "details": {
                "faithfulness_score": f"{faithfulness * 100:.1f}%",
                "citation_precision": f"{citation_precision * 100:.1f}%",
                "grounded_citations": sum(1 for c in structured_citations if c.is_grounded),
            }
        })

        return RAGResponse(
            query=query,
            answer=answer,
            citations=structured_citations,
            retrieved_chunks=chunks,
            pipeline_trace=trace,
            confidence_score=confidence,
            faithfulness_score=faithfulness,
            citation_precision=citation_precision,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(gen_latency, 2),
            total_latency_ms=round(retrieval_latency_ms + gen_latency, 2),
        )

    def _call_openai(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calls OpenAI with structured JSON schema response."""
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=8.0,
        )
        user_prompt = build_user_prompt(query, chunks)

        json_schema = {
            "name": "grounded_rag_response",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Grounded answer with inline citation footnotes [1], [2].",
                    },
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "citation_index": {"type": "integer"},
                                "source_id": {"type": "string"},
                                "source_title": {"type": "string"},
                                "section_title": {"type": "string"},
                                "quote": {"type": "string", "description": "Exact verbatim quote from the chunk."},
                            },
                            "required": ["citation_index", "source_id", "source_title", "section_title", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["answer", "citations"],
                "additionalProperties": False,
            },
            "strict": True,
        }

        active_system_prompt = system_prompt_override or SYSTEM_PROMPT_GROUNDED
        messages = [{"role": "system", "content": active_system_prompt}]
        if chat_history:
            for turn in chat_history[-4:]:
                if turn.get("role") in ("user", "assistant") and turn.get("content"):
                    messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_prompt})

        completion = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": json_schema},
            temperature=0.0,
        )

        content = completion.choices[0].message.content
        return json.loads(content)

    def _offline_grounded_answer(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Synthesizes high-fidelity grounded answer with direct quotes for test/demo environments."""
        top_chunk = chunks[0]
        second_chunk = chunks[1] if len(chunks) > 1 else top_chunk

        # Extract most informative sentences containing query keywords
        query_words = set(re.findall(r"\b[a-z0-9_]+\b", query.lower()))
        sentences = [s.strip() for s in re.split(r"[.!?]\n|[.!?]\s+", top_chunk["content"]) if len(s.strip()) > 20]

        best_sentence = ""
        for s in sentences:
            if any(w in s.lower() for w in query_words if len(w) > 3):
                best_sentence = s
                break

        if not best_sentence:
            # Query words do not exist in the retrieved chunks (out-of-domain question)
            return (
                "The provided knowledge base documents do not contain information about this question. (General knowledge fallback active: verified out-of-domain query with zero context citations).",
                [],
            )

        second_sentence = ""
        if second_chunk != top_chunk:
            sec_sents = [s.strip() for s in re.split(r"[.!?]\n|[.!?]\s+", second_chunk["content"]) if len(s.strip()) > 20]
            for s in sec_sents:
                if any(w in s.lower() for w in query_words if len(w) > 3) and s != best_sentence:
                    second_sentence = s
                    break

        answer = (
            f"Based on the technical documentation, {best_sentence} [1]. "
            + (f"Furthermore, {second_sentence} [2]." if second_sentence else "")
        )

        citations = [
            {
                "citation_index": 1,
                "source_id": top_chunk["doc_id"],
                "source_title": top_chunk["source_title"],
                "section_title": top_chunk["section_title"],
                "quote": best_sentence,
            }
        ]

        if second_sentence:
            citations.append(
                {
                    "citation_index": 2,
                    "source_id": second_chunk["doc_id"],
                    "source_title": second_chunk["source_title"],
                    "section_title": second_chunk["section_title"],
                    "quote": second_sentence,
                }
            )

        return answer, citations
