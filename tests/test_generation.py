"""Unit tests for hallucination detection and citation grounding."""

import pytest
from src.generation.hallucination_guard import HallucinationGuard
from src.generation.generator import GroundedGenerator


def test_hallucination_guard_verbatim_quote():
    guard = HallucinationGuard()
    chunks = [
        {"content": "Reciprocal Rank Fusion merges rankings with a smoothing constant of 60 to prevent outliers."}
    ]
    citations = [
        {"quote": "smoothing constant of 60 to prevent outliers", "citation_index": 1},
        {"quote": "invented quote that does not exist anywhere in text", "citation_index": 2},
    ]

    verified, precision = guard.verify_citations(citations, chunks)
    assert len(verified) == 2
    assert verified[0]["is_grounded"] is True
    assert verified[1]["is_grounded"] is False
    assert precision == 0.5


def test_faithfulness_metric():
    guard = HallucinationGuard()
    chunks = [
        {"content": "HNSW constructs a multi-layer graph where upper layers contain skip connections and layer zero contains all vectors."}
    ]
    grounded_answer = "HNSW builds a multi-layer graph structure where upper layers have skip connections."
    hallucinated_answer = "Quantum computing algorithms can teleport data across black holes using entanglement."

    high_faith = guard.evaluate_faithfulness(grounded_answer, chunks)
    low_faith = guard.evaluate_faithfulness(hallucinated_answer, chunks)

    assert high_faith > 0.6
    assert low_faith < 0.3


def test_offline_grounded_generator():
    generator = GroundedGenerator()
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "doc-01",
            "source_title": "RAG Manual",
            "section_title": "RRF",
            "content": "RRF score is computed by summing reciprocal ranks with a smoothing factor k=60.",
            "token_count": 16,
        }
    ]
    response = generator.generate("What is the smoothing factor in RRF?", chunks)
    assert response.query == "What is the smoothing factor in RRF?"
    assert len(response.citations) >= 1
    assert response.faithfulness_score > 0.5
    assert response.confidence_score > 0.5
