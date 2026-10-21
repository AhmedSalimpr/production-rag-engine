"""Application configuration using Pydantic Settings."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys & Gateway Base URL
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.tokenrouter.com/v1", alias="OPENAI_BASE_URL")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Models
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    # Qdrant Vector DB
    qdrant_host: Optional[str] = Field(default=None, alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="production_rag_knowledge_base", alias="QDRANT_COLLECTION")
    qdrant_storage_path: str = Field(default="./data/qdrant_storage", alias="QDRANT_STORAGE_PATH")

    # Retrieval parameters
    dense_top_k: int = Field(default=15, alias="DENSE_TOP_K")
    sparse_top_k: int = Field(default=15, alias="SPARSE_TOP_K")
    rrf_k: int = Field(default=60, alias="RRF_K")
    rerank_top_k: int = Field(default=5, alias="RERANK_TOP_K")

    # Server parameters
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=True, alias="DEBUG")


settings = Settings()
