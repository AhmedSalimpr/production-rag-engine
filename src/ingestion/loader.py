"""Document loader and built-in technical AI knowledge base seeder."""

import os
from pathlib import Path
from typing import List, Dict, Any
from src.ingestion.chunker import DocumentChunk, ContextualChunker


SAMPLE_KNOWLEDGE_DOCS = [
    {
        "doc_id": "rag-hybrid-search-spec",
        "title": "Production RAG & Hybrid Search Architecture",
        "content": """# Production RAG: Hybrid Search and Fusion Systems

## 1. Overview and Problem Statement
Standard dense vector retrieval using approximate nearest neighbors (ANN) often suffers from vocabulary mismatch and fails on exact technical keywords, error codes, and unique identifiers. Conversely, lexical search (BM25) excels at keyword matching but struggles with semantic paraphrasing and conceptual queries.

Production RAG systems resolve this dilemma by operating a hybrid retrieval pipeline that executes dense embedding search and sparse lexical search in parallel, followed by fusion and re-ranking stages.

## 2. Reciprocal Rank Fusion (RRF) Formulation
Reciprocal Rank Fusion is an unsupervised ranking algorithm that merges the ranked lists of multiple information retrieval engines. Given a set of documents D and a set of rankers R, the RRF score is computed as:

RRF_Score(d) = sum_{r in R} (1.0 / (k + rank_r(d)))

Where:
- k is a smoothing constant, typically set to 60. This constant prevents high outliers in one retriever from dominating the overall score.
- rank_r(d) is the 1-indexed position of document d in ranker r's result list.

RRF does not require calibrating or normalizing dissimilar score scales (e.g. cosine similarity vs BM25 raw scores), making it extremely robust across diverse query types.

## 3. Two-Stage Retrieval and Cross-Encoder Re-ranking
Because evaluating deep cross-encoders over an entire corpus of 1,000,000 chunks is computationally prohibitive (O(N) inference cost), production systems use a two-stage retrieve-and-rerank design:
1. Stage 1 (Fast Retrieval): High-recall dense search (Qdrant HNSW) + sparse search (BM25) pull the top 20 to 50 candidates in under 15ms.
2. Stage 2 (Precision Re-ranking): A cross-encoder or neural reranker inspects full query-document interactions, rescoring and pruning down to the top 3-5 high-signal chunks for the LLM context window.

This minimizes context window noise, lowers token expenditure, and mitigates the 'lost in the middle' phenomenon."""
    },
    {
        "doc_id": "vector-indexing-hnsw",
        "title": "Hierarchical Navigable Small World (HNSW) Indexing",
        "content": """# Vector Indexing: HNSW and Graph Retrieval

## 1. HNSW Graph Architecture
Hierarchical Navigable Small World (HNSW) is a state-of-the-art graph-based indexing algorithm for Approximate Nearest Neighbor (ANN) search in high-dimensional vector spaces. It constructs a multi-layer graph where:
- Layer 0 (Bottom): Contains all vectors with short-range edges for high-precision local routing.
- Upper Layers: Subsample vectors exponentially, creating long-range skip-connections analogous to skip-lists.

Search begins at the topmost layer with greedy routing, skipping across large clusters until reaching a local minimum, then stepping down to the next finer layer until reaching Layer 0.

## 2. Key Hyperparameters in Qdrant and HNSW
1. M (Number of bi-directional links per node): Typically configured between 16 and 64. Higher M improves recall at the expense of memory consumption and index build latency.
2. ef_construct (Size of dynamic candidate list during graph building): Higher values allow discovering better nearest neighbors during index creation.
3. ef_search (Search-time beam size): Controls the search trade-off between latency (p95) and recall accuracy. Setting ef_search=64 to 128 usually yields >98% Recall@10.

## 3. Quantization Strategies: Scalar and Product Quantization
In multi-million vector deployments, storing uncompressed FP32 vectors causes memory bloat (1536 dimensions * 4 bytes = 6.14 KB per vector).
- Scalar Quantization (SQ): Compresses FP32 to INT8, yielding a 4x reduction in RAM with <1% loss in retrieval recall.
- Product Quantization (PQ): Decomposes vectors into sub-vectors and clusters them into codebooks, achieving up to 16x-32x memory compression."""
    },
    {
        "doc_id": "evals-and-ragas-metrics",
        "title": "RAG Evaluation Frameworks & LLMOps Metrics",
        "content": """# Evaluating Production RAG Systems

## 1. The Core Metrics Triad
A comprehensive evaluation harness separates retrieval quality from generation quality:

1. Retrieval Quality:
   - Recall@k: The proportion of ground-truth relevant passages present within the top-k retrieved results.
   - Mean Reciprocal Rank (MRR): Evaluates how early the first relevant passage appears in the ranked list (MRR = 1 / rank_first).
   - Normalized Discounted Cumulative Gain (nDCG@k): Rewards placing the most relevant passages at the very top.

2. Generation & Grounding Quality:
   - Faithfulness (Hallucination Rate): Measures what percentage of factual statements made by the LLM can be directly inferred from the context.
   - Answer Relevance: Measures whether the response directly addresses the user's intent without superfluous filler.
   - Citation Precision: Verifies that cited footnotes correspond to factual evidence in the linked chunk.

## 2. Operational & Latency Budgets
Production systems monitor:
- p50 / p95 Retrieval Latency: Target < 40ms for hybrid search + reranking.
- Time-to-First-Token (TTFT): Critical for streaming user interfaces.
- Cost per 1,000 queries ($): Quantified tracking of embedding and generation token costs."""
    },
    {
        "doc_id": "lora-qlora-finetuning",
        "title": "Parameter-Efficient Fine-Tuning: LoRA and QLoRA",
        "content": """# PEFT: Low-Rank Adaptation (LoRA) and QLoRA

## 1. Mathematical Formulation of LoRA
Low-Rank Adaptation freezes the pre-trained model weights W_0 in R^{d x k} and injects trainable rank decomposition matrices A and B:

W = W_0 + Delta W = W_0 + (B * A) * (alpha / r)

Where:
- B in R^{d x r} and A in R^{r x k} with low intrinsic rank r << min(d, k).
- alpha is a constant scaling factor.
- During training, W_0 remains frozen and does not receive gradient updates, reducing trainable parameters by up to 99.9%.

## 2. QLoRA 4-bit Quantization
QLoRA backpropagates gradients through a frozen, 4-bit quantized base model into 16-bit LoRA adapter weights. It introduces three core innovations:
1. NF4 (NormalFloat4): An information-theoretically optimal quantile quantization data type for normally distributed weights.
2. Double Quantization (DQ): Quantizes the quantization constants themselves, saving ~0.37 bits per parameter.
3. Paged Optimizers: Uses CUDA Unified Memory to automatically page optimizer state memory between GPU and CPU RAM during memory spikes."""
    }
]


class DocumentLoader:
    """Loads and seeds documents for the production RAG engine."""

    def __init__(self, chunker: ContextualChunker = None):
        self.chunker = chunker or ContextualChunker()

    def load_seed_documents(self) -> List[DocumentChunk]:
        """Loads default technical AI engineering documentation for immediate portfolio demo."""
        all_chunks: List[DocumentChunk] = []
        for doc in SAMPLE_KNOWLEDGE_DOCS:
            chunks = self.chunker.chunk_markdown(
                doc_id=doc["doc_id"],
                title=doc["title"],
                text=doc["content"],
            )
            all_chunks.extend(chunks)
        return all_chunks

    def load_markdown_file(self, file_path: str) -> List[DocumentChunk]:
        """Loads an arbitrary markdown file from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = path.read_text(encoding="utf-8")
        doc_id = path.stem
        title = path.stem.replace("-", " ").replace("_", " ").title()

        return self.chunker.chunk_markdown(doc_id=doc_id, title=title, text=text)
