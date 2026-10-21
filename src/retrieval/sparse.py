"""Sparse lexical retrieval engine using BM25Okapi."""

import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.ingestion.chunker import DocumentChunk


class SparseRetriever:
    """Manages lexical search and BM25 scoring over chunk documents."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Lowercases and extracts alphanumeric token sequences."""
        clean_text = text.lower()
        tokens = re.findall(r"\b[a-z0-9_]+\b", clean_text)
        return tokens

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Builds the BM25 inverted index from document chunks."""
        self.chunks = list(chunks)
        self.tokenized_corpus = [self._tokenize(c.content) for c in self.chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def add_chunks(self, new_chunks: List[DocumentChunk]):
        """Appends new chunks and rebuilds BM25 inverted index."""
        existing_ids = {c.chunk_id for c in self.chunks}
        to_add = [c for c in new_chunks if c.chunk_id not in existing_ids]
        if to_add:
            self.chunks.extend(to_add)
            self.tokenized_corpus = [self._tokenize(c.content) for c in self.chunks]
            self.bm25 = BM25Okapi(self.tokenized_corpus)


    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Executes lexical search and returns top-k documents sorted by BM25 score."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = list(self.bm25.get_scores(query_tokens))
        # Handle small corpus edge cases where IDF = log(1.0) = 0
        for idx in range(len(scores)):
            if scores[idx] <= 0:
                tf = sum(self.tokenized_corpus[idx].count(t) for t in query_tokens)
                if tf > 0:
                    scores[idx] = float(tf) * 0.1

        scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in scored_indices[:top_k]:
            score = float(scores[idx])
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_title": chunk.source_title,
                    "section_title": chunk.section_title,
                    "content": chunk.content,
                    "score": score,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                }
            )
        return results
