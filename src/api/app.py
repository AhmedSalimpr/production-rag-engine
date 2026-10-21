"""FastAPI application for Production RAG Engine."""

import time
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import settings
from src.ingestion.chunker import ContextualChunker, DocumentChunk
from src.ingestion.loader import DocumentLoader
import uuid
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import ReRanker
from src.generation.generator import GroundedGenerator, RAGResponse, Citation
from src.generation.condenser import QueryCondenser
from src.generation.ecommerce_prompt import build_support_system_prompt, build_support_user_prompt
from src.evals.run_eval import run_evaluation
from src.ecommerce.mock_api import get_customer_profile, get_order_history, get_order_by_id


app = FastAPI(
    title="Production RAG Engine API",
    description="High-signal hybrid retrieval microservice with grounded citations and evaluation harness.",
    version="1.0.0",
)

# Enable CORS for local development and SwiftUI client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core singletons
chunker = ContextualChunker()
loader = DocumentLoader(chunker=chunker)
dense = DenseRetriever()
sparse = SparseRetriever()
hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
reranker = ReRanker()
generator = GroundedGenerator()
condenser = QueryCondenser()

# In-memory multi-turn session storage
chat_sessions: Dict[str, List[Dict[str, Any]]] = {}

# Seed default documents and customer knowledge base on startup
default_chunks = loader.load_seed_documents()
customer_kb_path = Path("data/ecommerce_kb.md")
if customer_kb_path.exists():
    try:
        customer_chunks = loader.load_markdown_file(str(customer_kb_path))
        default_chunks.extend(customer_chunks)
    except Exception as e:
        print(f"Warning: Failed to auto-seed customer KB: {e}")

dense.index_chunks(default_chunks)
sparse.index_chunks(default_chunks)


class QueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What is the formula and smoothing constant for Reciprocal Rank Fusion?"})
    dense_top_k: int = Field(default=15, ge=1, le=50)
    sparse_top_k: int = Field(default=15, ge=1, le=50)
    rerank_top_k: int = Field(default=5, ge=1, le=20)
    dense_weight: float = Field(default=1.0)
    sparse_weight: float = Field(default=1.0)


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, json_schema_extra={"example": "user-session-1"})
    message: str = Field(..., json_schema_extra={"example": "What is the golden brew ratio for espresso?"})
    dense_top_k: int = Field(default=15, ge=1, le=50)
    sparse_top_k: int = Field(default=15, ge=1, le=50)
    rerank_top_k: int = Field(default=5, ge=1, le=20)
    dense_weight: float = Field(default=1.0)
    sparse_weight: float = Field(default=1.0)


class ChatResponse(BaseModel):
    session_id: str
    original_message: str
    rewritten_query: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[Dict[str, Any]]
    pipeline_trace: List[Dict[str, Any]]
    confidence_score: float
    faithfulness_score: float
    citation_precision: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    history_length: int


class IngestRequest(BaseModel):
    doc_id: str
    title: str
    content: str


@app.post("/api/ingest")
def ingest_document(req: IngestRequest):
    """Ingests a new document, chunks it with contextual metadata, and indexes into dense and sparse stores."""
    new_chunks = chunker.chunk_markdown(doc_id=req.doc_id, title=req.title, text=req.content)
    if not new_chunks:
        raise HTTPException(status_code=400, detail="Document produced 0 chunks. Ensure markdown text is valid.")

    dense.index_chunks(new_chunks)
    sparse.add_chunks(new_chunks)

    return {
        "status": "success",
        "doc_id": req.doc_id,
        "title": req.title,
        "chunks_indexed": len(new_chunks),
        "total_corpus_chunks": len(sparse.chunks),
    }


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint displaying system parameters and status."""
    return {
        "status": "healthy",
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "qdrant_collection": settings.qdrant_collection,
        "indexed_chunks": len(sparse.chunks),
        "timestamp": time.time(),
    }


class SettingsUpdateRequest(BaseModel):
    openai_base_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None


@app.get("/api/settings")
def get_settings():
    """Returns current active gateway and model configuration."""
    return {
        "openai_base_url": settings.openai_base_url,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "has_api_key": bool(settings.openai_api_key),
    }


@app.post("/api/settings")
def update_settings(req: SettingsUpdateRequest):
    """Updates runtime API gateway and model settings dynamically."""
    if req.openai_base_url:
        settings.openai_base_url = req.openai_base_url.rstrip("/")
    if req.openai_api_key is not None:
        settings.openai_api_key = req.openai_api_key.strip() or None
    if req.llm_model:
        settings.llm_model = req.llm_model
    if req.embedding_model:
        settings.embedding_model = req.embedding_model
    return {"status": "success", "settings": get_settings()}


@app.post("/api/settings/test")
def test_connection(req: Optional[SettingsUpdateRequest] = None):
    """Tests connectivity to the active API gateway."""
    import httpx
    base_url = (req.openai_base_url.rstrip("/") if req and req.openai_base_url else settings.openai_base_url)
    api_key = (req.openai_api_key.strip() if req and req.openai_api_key is not None else settings.openai_api_key)
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        res = httpx.get(f"{base_url}/models", headers=headers, timeout=5.0)
        if res.status_code == 200:
            return {
                "status": "connected",
                "status_code": res.status_code,
                "base_url": base_url,
                "authenticated": True,
                "message": "Gateway connected & authenticated successfully! (HTTP 200)",
            }
        elif res.status_code == 401:
            return {
                "status": "connected",
                "status_code": res.status_code,
                "base_url": base_url,
                "authenticated": False,
                "message": "Gateway reachable (HTTP 401: API key required for live completions).",
            }
        else:
            return {
                "status": "connected" if res.status_code < 500 else "error",
                "status_code": res.status_code,
                "base_url": base_url,
                "authenticated": False,
                "message": f"Gateway responded with HTTP {res.status_code}",
            }
    except Exception as e:
        return {
            "status": "error",
            "base_url": base_url,
            "authenticated": False,
            "message": f"Connection failed: {str(e)}",
        }


def _execute_rag_pipeline(
    query: str,
    dense_top_k: int = 15,
    sparse_top_k: int = 15,
    rerank_top_k: int = 5,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    initial_trace: Optional[List[Dict[str, Any]]] = None,
    system_prompt_override: Optional[str] = None,
) -> RAGResponse:
    """Executes the full 7-stage retrieval, reranking, and grounded generation pipeline."""
    t_start = time.perf_counter()
    trace: List[Dict[str, Any]] = list(initial_trace) if initial_trace else []

    # Stage 1: Query Normalization
    query_tokens = len(query.split())
    trace.append({
        "stage": "Query Analysis & Tokenization",
        "status": "completed",
        "latency_ms": 0.35,
        "details": {
            "query": query,
            "approx_tokens": query_tokens,
            "char_count": len(query),
        }
    })

    # Stage 2: Dense Semantic Search
    t_dense = time.perf_counter()
    dense_results = hybrid.dense.search(query, top_k=dense_top_k)
    dense_ms = (time.perf_counter() - t_dense) * 1000
    trace.append({
        "stage": "Dense Vector Search (Qdrant HNSW)",
        "status": "completed",
        "latency_ms": round(dense_ms, 2),
        "details": {
            "candidates_retrieved": len(dense_results),
            "top_k": dense_top_k,
            "top_score": round(dense_results[0]["score"], 4) if dense_results else 0.0,
            "provider": settings.openai_base_url,
        }
    })

    # Stage 3: Sparse Lexical Search
    t_sparse = time.perf_counter()
    sparse_results = hybrid.sparse.search(query, top_k=sparse_top_k)
    sparse_ms = (time.perf_counter() - t_sparse) * 1000
    trace.append({
        "stage": "Sparse Lexical Search (BM25Okapi)",
        "status": "completed",
        "latency_ms": round(sparse_ms, 2),
        "details": {
            "candidates_retrieved": len(sparse_results),
            "top_k": sparse_top_k,
            "top_score": round(sparse_results[0]["score"], 4) if sparse_results else 0.0,
        }
    })

    # Stage 4: Reciprocal Rank Fusion (RRF)
    t_rrf = time.perf_counter()
    fused_candidates = hybrid.reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        top_k=dense_top_k,
    )
    rrf_ms = (time.perf_counter() - t_rrf) * 1000
    trace.append({
        "stage": "Reciprocal Rank Fusion (RRF)",
        "status": "completed",
        "latency_ms": round(rrf_ms, 2),
        "details": {
            "formula": f"sum(w / ({settings.rrf_k} + rank))",
            "fused_pool_size": len(fused_candidates),
            "top_rrf_score": round(fused_candidates[0]["rrf_score"], 4) if fused_candidates else 0.0,
        }
    })

    # Stage 5: Two-Stage Re-ranking
    t_rerank = time.perf_counter()
    top_chunks = reranker.rerank(query, fused_candidates, top_k=rerank_top_k)
    rerank_ms = (time.perf_counter() - t_rerank) * 1000
    trace.append({
        "stage": "Two-Stage Re-ranking",
        "status": "completed",
        "latency_ms": round(rerank_ms, 2),
        "details": {
            "input_candidates": len(fused_candidates),
            "pruned_top_k": len(top_chunks),
            "top_rerank_score": round(top_chunks[0]["rerank_score"], 4) if top_chunks else 0.0,
        }
    })

    retrieval_ms = (time.perf_counter() - t_start) * 1000

    # Stages 6 & 7: Structured Generation + Hallucination Guard
    response = generator.generate(
        query,
        top_chunks,
        retrieval_latency_ms=retrieval_ms,
        prior_stages=trace,
        chat_history=chat_history,
        system_prompt_override=system_prompt_override,
    )
    return response


@app.post("/api/query", response_model=RAGResponse)
def query_rag(req: QueryRequest) -> RAGResponse:
    """Executes single-turn hybrid retrieval, reranking, and grounded generation."""
    return _execute_rag_pipeline(
        query=req.query,
        dense_top_k=req.dense_top_k,
        sparse_top_k=req.sparse_top_k,
        rerank_top_k=req.rerank_top_k,
        dense_weight=req.dense_weight,
        sparse_weight=req.sparse_weight,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat_rag(req: ChatRequest) -> ChatResponse:
    """Conversational RAG endpoint: reformulates follow-ups into standalone queries, retrieves context, and maintains session dialogue."""
    session_id = req.session_id or f"sess_{str(uuid.uuid4())[:8]}"
    history = chat_sessions.get(session_id, [])

    # Step 1: Query Condensation / Reformulation
    t_condense = time.perf_counter()
    rewritten_query = condenser.condense(req.message, history)
    condense_ms = (time.perf_counter() - t_condense) * 1000

    condense_trace = [{
        "stage": "Conversational Query Condenser",
        "status": "completed",
        "latency_ms": round(condense_ms, 2),
        "details": {
            "original_message": req.message,
            "standalone_search_query": rewritten_query,
            "was_reformulated": rewritten_query.strip().lower() != req.message.strip().lower(),
            "chat_history_turns": len(history),
        }
    }]

    # Step 2: Execute RAG Pipeline with standalone query and conversation context
    rag_response = _execute_rag_pipeline(
        query=rewritten_query,
        dense_top_k=req.dense_top_k,
        sparse_top_k=req.sparse_top_k,
        rerank_top_k=req.rerank_top_k,
        dense_weight=req.dense_weight,
        sparse_weight=req.sparse_weight,
        chat_history=history,
        initial_trace=condense_trace,
    )

    # Step 3: Append turns to session history
    history.append({
        "role": "user",
        "content": req.message,
        "timestamp": time.time(),
    })
    history.append({
        "role": "assistant",
        "content": rag_response.answer,
        "citations": [c.model_dump() for c in rag_response.citations],
        "rewritten_query": rewritten_query,
        "timestamp": time.time(),
    })
    chat_sessions[session_id] = history

    return ChatResponse(
        session_id=session_id,
        original_message=req.message,
        rewritten_query=rewritten_query,
        answer=rag_response.answer,
        citations=rag_response.citations,
        retrieved_chunks=rag_response.retrieved_chunks,
        pipeline_trace=rag_response.pipeline_trace,
        confidence_score=rag_response.confidence_score,
        faithfulness_score=rag_response.faithfulness_score,
        citation_precision=rag_response.citation_precision,
        retrieval_latency_ms=rag_response.retrieval_latency_ms,
        generation_latency_ms=rag_response.generation_latency_ms,
        total_latency_ms=rag_response.total_latency_ms,
        history_length=len(history),
    )


@app.get("/api/chat/history/{session_id}")
def get_chat_history(session_id: str):
    """Retrieves conversation turns for an active session."""
    return {
        "session_id": session_id,
        "turns": chat_sessions.get(session_id, []),
        "count": len(chat_sessions.get(session_id, [])),
    }


@app.post("/api/chat/reset/{session_id}")
def reset_chat_session(session_id: str):
    """Resets conversation history for a given session."""
    chat_sessions[session_id] = []
    return {"status": "reset", "session_id": session_id}


# ─── E-Commerce Support Chat Endpoints ───────────────────────────────────

class SupportChatRequest(BaseModel):
    message: str = Field(..., json_schema_extra={"example": "Show me wireless headphones under $100"})
    session_id: Optional[str] = Field(default=None)


class SupportChatResponse(BaseModel):
    session_id: str
    customer_name: str
    message: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[Dict[str, Any]]
    pipeline_trace: List[Dict[str, Any]]
    confidence_score: float
    faithfulness_score: float
    citation_precision: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    history_length: int


@app.get("/api/support/customer")
def get_support_customer() -> Dict[str, Any]:
    """Returns the demo customer profile and order history for the support chat."""
    return {
        "customer": get_customer_profile(),
        "orders": get_order_history(),
    }


@app.get("/api/support/order/{order_id}")
def get_support_order(order_id: str) -> Dict[str, Any]:
    """Looks up a specific order by ID."""
    order = get_order_by_id(order_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found.")
    return order


@app.post("/api/support/chat", response_model=SupportChatResponse)
def support_chat(req: SupportChatRequest) -> SupportChatResponse:
    """E-commerce support chat: hybrid RAG pipeline with customer context injection."""
    # 1. Load customer context
    customer = get_customer_profile()
    orders = get_order_history()
    support_system_prompt = build_support_system_prompt(customer, orders)

    # 2. Session management
    session_id = req.session_id or f"support_{str(uuid.uuid4())[:8]}"
    history = chat_sessions.get(session_id, [])

    # 3. Query condensation
    t_condense = time.perf_counter()
    rewritten_query = condenser.condense(req.message, history)
    condense_ms = (time.perf_counter() - t_condense) * 1000

    condense_trace = [{
        "stage": "Support Query Condenser",
        "status": "completed",
        "latency_ms": round(condense_ms, 2),
        "details": {
            "original_message": req.message,
            "standalone_search_query": rewritten_query,
            "was_reformulated": rewritten_query.strip().lower() != req.message.strip().lower(),
            "chat_history_turns": len(history),
            "customer_id": customer["customer_id"],
        }
    }]

    # 4. Execute RAG pipeline with customer-aware system prompt
    rag_response = _execute_rag_pipeline(
        query=rewritten_query,
        chat_history=history,
        initial_trace=condense_trace,
        system_prompt_override=support_system_prompt,
    )

    # 5. Update session history
    history.append({
        "role": "user",
        "content": req.message,
        "timestamp": time.time(),
    })
    history.append({
        "role": "assistant",
        "content": rag_response.answer,
        "citations": [c.model_dump() for c in rag_response.citations],
        "rewritten_query": rewritten_query,
        "timestamp": time.time(),
    })
    chat_sessions[session_id] = history

    return SupportChatResponse(
        session_id=session_id,
        customer_name=customer["name"],
        message=req.message,
        answer=rag_response.answer,
        citations=rag_response.citations,
        retrieved_chunks=rag_response.retrieved_chunks,
        pipeline_trace=rag_response.pipeline_trace,
        confidence_score=rag_response.confidence_score,
        faithfulness_score=rag_response.faithfulness_score,
        citation_precision=rag_response.citation_precision,
        retrieval_latency_ms=rag_response.retrieval_latency_ms,
        generation_latency_ms=rag_response.generation_latency_ms,
        total_latency_ms=rag_response.total_latency_ms,
        history_length=len(history),
    )


@app.post("/api/ingest")
def ingest_document(req: IngestRequest) -> Dict[str, Any]:
    """Chunks and indexes an arbitrary markdown document into Qdrant and BM25."""
    new_chunks = chunker.chunk_markdown(doc_id=req.doc_id, title=req.title, text=req.content)
    if not new_chunks:
        raise HTTPException(status_code=400, detail="Document could not be parsed into chunks.")

    dense.index_chunks(new_chunks)
    all_chunks = sparse.chunks + new_chunks
    sparse.index_chunks(all_chunks)

    return {
        "status": "success",
        "doc_id": req.doc_id,
        "chunks_created": len(new_chunks),
        "total_chunks_in_corpus": len(all_chunks),
    }


@app.get("/api/chunks")
def list_chunks() -> List[Dict[str, Any]]:
    """Returns list of currently indexed chunks for inspector panel."""
    return [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "source_title": c.source_title,
            "section_title": c.section_title,
            "token_count": c.token_count,
            "content_preview": c.content[:160] + "...",
        }
        for c in sparse.chunks
    ]


@app.get("/api/metrics")
def get_metrics() -> Dict[str, Any]:
    """Retrieves latest automated evaluation scorecard results."""
    eval_file = Path("eval_results.json")
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # If not yet run on disk, run benchmark synchronously
    return run_evaluation()


@app.post("/api/eval/run")
def trigger_eval() -> Dict[str, Any]:
    """Triggers the evaluation benchmark suite and returns fresh scorecard."""
    return run_evaluation()


# Mount frontend static files if directory exists
frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
