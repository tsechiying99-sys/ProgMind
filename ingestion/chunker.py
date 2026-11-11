"""面向课程资料的分页感知切片器。"""

import hashlib
import re
from typing import List

from .models import KnowledgeChunk, ParsedPage


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])|\n+")


class KnowledgeChunker:
    def __init__(self, chunk_size: int = 700, overlap: int = 80) -> None:
        self.chunk_size = max(200, chunk_size)
        self.overlap = max(0, min(overlap, self.chunk_size // 3))

    def chunk_pages(
        self,
        pages: List[ParsedPage],
        *,
        document_id: str,
        filename: str,
        title: str,
    ) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        for page in pages:
            page_chunks = self._chunk_text(page.text)
            for page_chunk_index, content in enumerate(page_chunks):
                digest = hashlib.sha256(
                    f"{document_id}:{page.page_number}:{page_chunk_index}:{content}".encode("utf-8")
                ).hexdigest()
                metadata = {
                    "document_id": document_id,
                    "source_file": filename,
                    "title": title,
                    "page_start": page.page_number,
                    "page_end": page.page_number,
                    "strategy": page.strategy,
                    "ocr_used": page.strategy in {"ocr", "hybrid"},
                    "quality_score": round(page.quality_score, 4),
                    "chunk_index": len(chunks),
                }
                if page.ocr_confidence is not None:
                    metadata["ocr_confidence"] = round(page.ocr_confidence, 4)
                chunks.append(KnowledgeChunk(digest, content, metadata))

        total = len(chunks)
        for chunk in chunks:
            chunk.metadata["total_chunks"] = total
        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        units = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
        chunks: List[str] = []
        current = ""
        for unit in units:
            if len(unit) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                step = self.chunk_size - self.overlap
                chunks.extend(
                    unit[start:start + self.chunk_size].strip()
                    for start in range(0, len(unit), step)
                    if unit[start:start + self.chunk_size].strip()
                )
                continue

            separator = "\n" if current else ""
            if len(current) + len(separator) + len(unit) <= self.chunk_size:
                current = f"{current}{separator}{unit}"
                continue

            chunks.append(current)
            prefix = current[-self.overlap:].lstrip() if self.overlap else ""
            current = f"{prefix}\n{unit}".strip() if prefix else unit

        if current:
            chunks.append(current)
        return chunks
