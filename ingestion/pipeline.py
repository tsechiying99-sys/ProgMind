"""PDF 解析、清洗、质量门控和知识切片的统一入口。"""

import hashlib
import os
from pathlib import Path
from typing import Optional

from .chunker import KnowledgeChunker
from .models import DocumentQualityError, IngestionReport, IngestionResult
from .pdf_parser import PDFParser, PDFParserConfig
from .text_cleaner import remove_repeated_margins, text_quality


class DocumentIngestionPipeline:
    def __init__(
        self,
        parser: Optional[PDFParser] = None,
        chunker: Optional[KnowledgeChunker] = None,
        document_min_quality: float = 0.55,
    ) -> None:
        self.parser = parser or PDFParser()
        self.chunker = chunker or KnowledgeChunker()
        self.document_min_quality = document_min_quality

    @classmethod
    def from_environment(cls) -> "DocumentIngestionPipeline":
        config = PDFParserConfig(
            max_pages=int(os.getenv("KNOWLEDGE_MAX_PDF_PAGES", "200")),
            ocr_dpi=int(os.getenv("PDF_OCR_DPI", "220")),
            native_min_chars=int(os.getenv("PDF_NATIVE_MIN_CHARS", "40")),
            native_min_quality=float(os.getenv("PDF_NATIVE_MIN_QUALITY", "0.75")),
            ocr_min_confidence=float(os.getenv("PDF_OCR_MIN_CONFIDENCE", "0.55")),
            image_coverage_threshold=float(os.getenv("PDF_IMAGE_COVERAGE_THRESHOLD", "0.35")),
            max_render_pixels=int(os.getenv("PDF_MAX_RENDER_PIXELS", "25000000")),
        )
        return cls(
            parser=PDFParser(config=config),
            chunker=KnowledgeChunker(
                chunk_size=int(os.getenv("KNOWLEDGE_CHUNK_SIZE", "700")),
                overlap=int(os.getenv("KNOWLEDGE_CHUNK_OVERLAP", "80")),
            ),
            document_min_quality=float(os.getenv("PDF_DOCUMENT_MIN_QUALITY", "0.55")),
        )

    def ingest_pdf(self, content: bytes, filename: str) -> IngestionResult:
        document_id = hashlib.sha256(content).hexdigest()
        pages = remove_repeated_margins(self.parser.parse(content))

        # 清理页眉页脚后重新计算最终质量，空白页不参与文档均分。
        usable_pages = [page for page in pages if page.text.strip()]
        for page in usable_pages:
            page.quality_score = text_quality(page.text)
            if page.ocr_confidence is not None:
                page.quality_score = round(
                    page.quality_score * (0.7 + 0.3 * page.ocr_confidence), 4
                )

        if not usable_pages:
            raise DocumentQualityError("PDF 未解析出可用于检索的文本")

        quality_score = sum(page.quality_score for page in usable_pages) / len(usable_pages)
        if quality_score < self.document_min_quality:
            raise DocumentQualityError(
                f"PDF 文本质量分 {quality_score:.2f}，低于入库阈值 {self.document_min_quality:.2f}"
            )

        safe_filename = Path(filename or "document.pdf").name
        title = Path(safe_filename).stem or "PDF 文档"
        chunks = self.chunker.chunk_pages(
            usable_pages,
            document_id=document_id,
            filename=safe_filename,
            title=title,
        )
        if not chunks:
            raise DocumentQualityError("PDF 文本无法生成有效知识片段")

        warnings = [warning for page in pages for warning in page.warnings]
        empty_pages = len(pages) - len(usable_pages)
        if empty_pages:
            warnings.append(f"有 {empty_pages} 页未识别到有效文本，未写入知识库")

        report = IngestionReport(
            pages=len(pages),
            native_pages=sum(page.strategy == "native" for page in pages),
            ocr_pages=sum(page.strategy == "ocr" for page in pages),
            hybrid_pages=sum(page.strategy == "hybrid" for page in pages),
            quality_score=quality_score,
            warnings=warnings,
        )
        return IngestionResult(document_id, pages, chunks, report)
