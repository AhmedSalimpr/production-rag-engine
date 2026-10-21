# Production RAG Engine — Portfolio Project Plan

> **Task**: Build an enterprise-grade Production RAG microservice with hybrid search, re-ranking, grounded citations, automated evaluation harness, and a modern interactive UI.
> **Target Audience**: AI Engineering hiring managers (optimized for the 90-second technical scan).

---

## 🎯 Project Overview & Signals Proven

| Dimension | Implementation | Signal Proven to Interviewers |
|---|---|---|
| **Retrieval Engine** | Hybrid Dense + Sparse BM25 via Reciprocal Rank Fusion (RRF) | Overcomes vocabulary mismatch, handles technical jargon & IDs |
| **Vector Database** | Qdrant client (Docker/Cloud enterprise service + embedded fallback) | Enterprise microservice architecture, production schema & payload filters |
| **Re-ranking** | Cross-encoder / Cohere / API reranker on top-k candidates | 2-stage retrieval pipeline, reduces context noise & token cost |
| **Generation & Grounding** | Structured Pydantic response with inline citations & quote provenance | Hallucination mitigation, traceable enterprise answers |
| **Eval Harness (The Win)** | Automated Golden Set eval: Recall@k, MRR, Faithfulness, Citation Precision, Cost/Latency | "Eval is the new system design" — quantified benchmarks |
| **Interface & API** | FastAPI REST endpoints + Apple HIG Liquid Glass Web UI + Native SwiftUI App | Full-stack deployment readiness, multi-platform demonstration |

---

## 🏗️ Architecture & File Structure

```
c:/Users/boosb/Documents/AI projects/
├── docker-compose.yml              # Qdrant service + App containerization
├── requirements.txt                # Python dependencies
├── .env.example                    # Clean environment variable template
├── README.md                       # Product spec: architecture, eval numbers, trade-offs
├── DESIGN.md                       # Apple HIG & Liquid Glass tokens & UX specification
├── eval_results.json               # Persisted automated benchmark results
├── src/
│   ├── config.py                   # Pydantic Settings & environment validation
│   ├── ingestion/
│   │   ├── chunker.py              # Contextual / recursive chunking with header metadata
│   │   └── loader.py               # Technical AI knowledge base seeder & loader
│   ├── retrieval/
│   │   ├── dense.py                # Qdrant dense vector store manager
│   │   ├── sparse.py               # BM25 lexical retriever with small-corpus fallback
│   │   ├── hybrid.py               # RRF (Reciprocal Rank Fusion) blender
│   │   └── reranker.py             # Cross-encoder / heuristic reranking module
│   ├── generation/
│   │   ├── prompt.py               # Grounded system prompts & citation constraints
│   │   ├── generator.py            # LLM generation with Pydantic citation models
│   │   └── hallucination_guard.py  # Self-reflection / quote verification filter
│   ├── evals/
│   │   ├── golden_dataset.json     # Curated technical test questions with ground truth
│   │   ├── metrics.py              # Recall@k, MRR, Faithfulness, Precision, Latency
│   │   └── run_eval.py             # CLI runner outputting benchmark summary table
│   └── api/
│       └── app.py                  # FastAPI service (/query, /ingest, /eval, /health)
├── frontend/
│   ├── index.html                  # Apple HIG macOS window layout & split-view
│   ├── index.css                   # Liquid Glass design tokens (blur, specular highlight, SF font)
│   └── app.js                      # Reactive UI state, live chunk inspection & search
├── swiftui-client/
│   ├── Package.swift               # Swift 6 multi-platform package (macOS 15+, iOS 18+)
│   └── Sources/
│       ├── RAGStudioApp.swift      # Entry point with hidden title bar
│       ├── Models/RAGModels.swift  # Codable & Sendable RAG schemas
│       ├── Services/RAGAPIService.swift # Modern async/await @Observable network service
│       └── Views/
│           ├── ContentView.swift   # NavigationSplitView 3-column architecture
│           ├── QueryView.swift     # Search bar, suggestion chips, citations HUD
│           ├── InspectorView.swift # Chunk inspector & RRF rank visualizer
│           └── ScorecardView.swift # LLMOps metrics dashboard
└── tests/
    ├── test_retrieval.py           # Unit tests for BM25, vector search & RRF
    └── test_generation.py          # Unit tests for citation validation & faithfulness
```

---

## 📋 Phased Execution Status

- [x] **Phase 1: Core Setup & Ingestion Pipeline** — `requirements.txt`, `.env.example`, `src/config.py`, `src/ingestion/chunker.py`, `src/ingestion/loader.py`
- [x] **Phase 2: Hybrid Retrieval & Re-ranking** — `src/retrieval/dense.py` (Qdrant), `src/retrieval/sparse.py` (BM25), `src/retrieval/hybrid.py` (RRF), `src/retrieval/reranker.py`
- [x] **Phase 3: Grounded Generation & Guardrails** — `src/generation/prompt.py`, `src/generation/generator.py`, `src/generation/hallucination_guard.py`
- [x] **Phase 4: Automated Evaluation Harness** — `src/evals/golden_dataset.json`, `src/evals/metrics.py`, `src/evals/run_eval.py`
- [x] **Phase 5: FastAPI Microservice** — `src/api/app.py` with `/api/query`, `/api/ingest`, `/api/metrics`, `/api/eval/run`, `/api/health`
- [x] **Phase 6: Apple HIG Web UI & Design System** — `DESIGN.md`, `frontend/index.html`, `frontend/index.css`, `frontend/app.js`
- [x] **Phase 7: Native SwiftUI Client** — `swiftui-client/` with Swift 6, `@Observable`, `NavigationSplitView`
- [x] **Phase 8: Verification & Benchmark** — Pytest suite passing 7/7, automated benchmark completed with 100% recall & precision, browser session validated
