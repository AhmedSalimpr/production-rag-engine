"""Contextual and recursive document chunking engine."""

import re
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_title: str
    section_title: str
    content: str
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextualChunker:
    """Chunks documents while preserving hierarchical section context and token budgets."""

    def __init__(self, target_tokens: int = 250, overlap_tokens: int = 40):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def _approx_token_count(self, text: str) -> int:
        """Fast approximate token count (avg 4 chars/token in English)."""
        words = text.split()
        return max(1, int(len(words) * 1.3))

    def chunk_markdown(self, doc_id: str, title: str, text: str) -> List[DocumentChunk]:
        """Splits markdown hierarchically by headings, retaining section breadcrumbs."""
        lines = text.splitlines()
        chunks: List[DocumentChunk] = []

        current_section = "Introduction"
        current_buffer: List[str] = []
        current_token_count = 0
        chunk_idx = 0

        heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$")

        for line in lines:
            heading_match = heading_pattern.match(line)
            if heading_match:
                # Flush existing buffer before switching section if sizable
                if current_buffer and current_token_count >= self.target_tokens // 2:
                    content_str = "\n".join(current_buffer).strip()
                    if content_str:
                        chunk_id = hashlib.sha256(f"{doc_id}-{chunk_idx}-{content_str[:50]}".encode()).hexdigest()[:16]
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                source_title=title,
                                section_title=current_section,
                                content=content_str,
                                token_count=current_token_count,
                                metadata={"chunk_index": chunk_idx},
                            )
                        )
                        chunk_idx += 1
                        current_buffer = []
                        current_token_count = 0

                current_section = heading_match.group(2).strip()
                current_buffer.append(line)
                current_token_count += self._approx_token_count(line)
                continue

            line_tokens = self._approx_token_count(line)

            # If adding this line exceeds target tokens, emit a chunk
            if current_token_count + line_tokens > self.target_tokens and current_buffer:
                content_str = "\n".join(current_buffer).strip()
                if content_str:
                    chunk_id = hashlib.sha256(f"{doc_id}-{chunk_idx}-{content_str[:50]}".encode()).hexdigest()[:16]
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            source_title=title,
                            section_title=current_section,
                            content=content_str,
                            token_count=current_token_count,
                            metadata={"chunk_index": chunk_idx},
                        )
                    )
                    chunk_idx += 1

                # Overlap: keep the last 2 lines for continuity
                overlap_lines = current_buffer[-2:] if len(current_buffer) >= 2 else []
                current_buffer = overlap_lines + [line]
                current_token_count = sum(self._approx_token_count(l) for l in current_buffer)
            else:
                current_buffer.append(line)
                current_token_count += line_tokens

        # Flush final remaining buffer
        if current_buffer:
            content_str = "\n".join(current_buffer).strip()
            if content_str:
                chunk_id = hashlib.sha256(f"{doc_id}-{chunk_idx}-{content_str[:50]}".encode()).hexdigest()[:16]
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        source_title=title,
                        section_title=current_section,
                        content=content_str,
                        token_count=current_token_count,
                        metadata={"chunk_index": chunk_idx},
                    )
                )

        return chunks
