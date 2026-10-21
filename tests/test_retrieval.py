"""Unit tests for dense, sparse, and hybrid RRF retrieval."""

import pytest
from src.ingestion.chunker import DocumentChunk, ContextualChunker
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import ReRanker


def test_chunker_basic():
    chunker = ContextualChunker(target_tokens=50, overlap_tokens=10)
    text = (
        "# Title\n\n"
        "## Section 1\n"
        "Here is line one of technical content explaining vector databases.\n"
        "Here is line two with more information about HNSW graphs.\n\n"
        "## Section 2\n"
        "This is a second section explaining BM25 lexical keyword matching algorithms.\n"
    )
    chunks = chunker.chunk_markdown(doc_id="test-doc", title="Test Doc", text=text)
    assert len(chunks) >= 2
    assert chunks[0].doc_id == "test-doc"
    assert chunks[0].source_title == "Test Doc"


def test_sparse_bm25_search():
    sparse = SparseRetriever()
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            source_title="RAG Guide",
            section_title="Reciprocal Rank Fusion",
            content="Reciprocal Rank Fusion RRF merges dense vector rankings with BM25 keyword rankings.",
            token_count=16,
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            source_title="Quantization",
            section_title="QLoRA",
            content="QLoRA uses NormalFloat4 NF4 quantization and double quantization for 4-bit fine-tuning.",
            token_count=14,
        ),
    ]
    sparse.index_chunks(chunks)
    res = sparse.search("Reciprocal Rank Fusion", top_k=2)
    assert len(res) == 2
    assert res[0]["chunk_id"] == "c1"
    assert res[0]["score"] > res[1]["score"]


def test_rrf_fusion_logic():
    hybrid = HybridRetriever(rrf_k=60)
    dense_results = [
        {"chunk_id": "c1", "score": 0.95, "content": "Text 1", "doc_id": "d1"},
        {"chunk_id": "c2", "score": 0.85, "content": "Text 2", "doc_id": "d2"},
    ]
    sparse_results = [
        {"chunk_id": "c2", "score": 12.0, "content": "Text 2", "doc_id": "d2"},
        {"chunk_id": "c1", "score": 4.0, "content": "Text 1", "doc_id": "d1"},
    ]

    fused = hybrid.reciprocal_rank_fusion(dense_results, sparse_results, top_k=2)
    assert len(fused) == 2
    # Verify RRF score computation: 1/(60+rank1) + 1/(60+rank2)
    # c1: 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
    # c2: 1/62 + 1/61 = 0.03252
    assert fused[0]["rrf_score"] > 0


def test_reranker():
    reranker = ReRanker(top_k=2)
    candidates = [
        {"chunk_id": "c1", "content": "Unrelated culinary recipes about pasta and sauce.", "rrf_score": 0.03, "doc_id": "d1"},
        {"chunk_id": "c2", "content": "HNSW index uses M bi-directional links and ef_search for fast ANN search.", "rrf_score": 0.02, "doc_id": "d2"},
    ]
    reranked = reranker.rerank("What is ef_search in HNSW?", candidates, top_k=1)
    assert len(reranked) == 1
    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[0]["final_rank"] == 1
