"""Dense vector retrieval engine using Qdrant and OpenAI embeddings."""

import os
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
)
from src.config import settings
from src.ingestion.chunker import DocumentChunk


class DenseRetriever:
    """Manages dense vector storage and similarity search via Qdrant."""

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or settings.qdrant_collection
        self.client = self._init_qdrant_client()
        self._ensure_collection()

    def _init_qdrant_client(self) -> QdrantClient:
        """Connects to Qdrant server (Docker/Cloud) or falls back to local embedded storage."""
        if settings.qdrant_host:
            try:
                client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    api_key=settings.qdrant_api_key,
                    timeout=2.0,
                )
                # Verify connectivity
                client.get_collections()
                return client
            except Exception:
                pass  # Fallback to local embedded storage if remote daemon is offline

        os.makedirs(settings.qdrant_storage_path, exist_ok=True)
        try:
            return QdrantClient(path=settings.qdrant_storage_path)
        except Exception:
            return QdrantClient(":memory:")

    def _ensure_collection(self):
        """Creates the Qdrant collection if it does not already exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def compute_embedding(self, text: str) -> List[float]:
        """Generates embedding using OpenAI API or deterministic unit-norm vector fallback."""
        if settings.openai_api_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url, timeout=4.0)
                response = client.embeddings.create(
                    input=text,
                    model=settings.embedding_model,
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"[Warning] OpenAI embedding failed ({e}); using deterministic fallback.")

        # Deterministic pseudo-embedding for testing or offline execution
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(settings.embedding_dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def compute_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding generation with fallback."""
        if settings.openai_api_key and texts:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
                response = client.embeddings.create(
                    input=texts,
                    model=settings.embedding_model,
                )
                return [d.embedding for d in response.data]
            except Exception as e:
                print(f"[Warning] Batch embedding failed ({e}); using deterministic fallback.")

        return [self.compute_embedding(t) for t in texts]

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Indexes document chunks into Qdrant."""
        if not chunks:
            return

        texts = [c.content for c in chunks]
        embeddings = self.compute_batch_embeddings(texts)

        points = []
        for i, chunk in enumerate(chunks):
            # Numeric integer ID or UUID from chunk_id
            point_id = int(hashlib.md5(chunk.chunk_id.encode()).hexdigest()[:15], 16)
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_title": chunk.source_title,
                "section_title": chunk.section_title,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata,
            }
            points.append(PointStruct(id=point_id, vector=embeddings[i], payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Executes dense cosine similarity search."""
        query_vector = self.compute_embedding(query)

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
            )
            points = response.points
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )

        results = []
        for scored_point in points:
            results.append(
                {
                    "chunk_id": scored_point.payload.get("chunk_id"),
                    "doc_id": scored_point.payload.get("doc_id"),
                    "source_title": scored_point.payload.get("source_title"),
                    "section_title": scored_point.payload.get("section_title"),
                    "content": scored_point.payload.get("content"),
                    "score": float(scored_point.score),
                    "token_count": scored_point.payload.get("token_count", 0),
                    "metadata": scored_point.payload.get("metadata", {}),
                }
            )
        return results
