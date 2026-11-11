"""文档摄取过程使用的数据结构与业务异常。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class DocumentIngestionError(ValueError):
    """文档无法完成摄取。"""


class PDFValidationError(DocumentIngestionError):
    """PDF 文件格式、安全性或页数校验失败。"""


class DocumentQualityError(DocumentIngestionError):
    """文档解析成功，但文本质量不足以进入知识库。"""


@dataclass
class ParsedPage:
    """一页 PDF 的最终文本和解析证据。"""

    page_number: int
    strategy: str
    text: str
    native_quality: float = 0.0
    ocr_confidence: Optional[float] = None
    image_coverage: float = 0.0
    quality_score: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class KnowledgeChunk:
    """可直接写入向量库的知识片段。"""

    chunk_id: str
    content: str
    metadata: Dict[str, Any]

    def to_record(self) -> Dict[str, Any]:
        return {
            "id": self.chunk_id,
            "content": self.content,
            "metadata": dict(self.metadata),
        }


@dataclass
class IngestionReport:
    """PDF 解析质量摘要，供接口和日志展示。"""

    pages: int
    native_pages: int
    ocr_pages: int
    hybrid_pages: int
    quality_score: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages": self.pages,
            "native_pages": self.native_pages,
            "ocr_pages": self.ocr_pages,
            "hybrid_pages": self.hybrid_pages,
            "quality_score": round(self.quality_score, 4),
            "warnings": list(self.warnings),
        }


@dataclass
class IngestionResult:
    """完成质量门控后的文档摄取结果。"""
    document_id: str
    pages: List[ParsedPage]
    chunks: List[KnowledgeChunk]
    report: IngestionReport
