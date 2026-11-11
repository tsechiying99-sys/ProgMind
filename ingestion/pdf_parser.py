"""基于 PyMuPDF 与 RapidOCR 的逐页 PDF 解析器。"""

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pymupdf

from .models import PDFValidationError, ParsedPage
from .text_cleaner import clean_text, text_quality

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PDFParserConfig:
    max_pages: int = 200
    ocr_dpi: int = 220
    native_min_chars: int = 40
    native_min_quality: float = 0.75
    ocr_min_confidence: float = 0.55
    image_coverage_threshold: float = 0.35
    max_render_pixels: int = 25_000_000


class PDFParser:
    """按页选择原生文本或 OCR，并保留选择依据。"""

    def __init__(
        self,
        config: Optional[PDFParserConfig] = None,
        ocr_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config or PDFParserConfig()
        self._ocr_factory = ocr_factory
        self._ocr_engine = None
        self._ocr_lock = threading.Lock()

    def parse(self, content: bytes) -> List[ParsedPage]:
        if not content or not content.lstrip()[:5] == b"%PDF-":
            raise PDFValidationError("文件内容不是有效的 PDF")

        try:
            document = pymupdf.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise PDFValidationError(f"PDF 打开失败: {exc}") from exc

        try:
            if document.needs_pass:
                raise PDFValidationError("暂不支持加密或需要密码的 PDF")
            if document.page_count <= 0:
                raise PDFValidationError("PDF 中没有可解析页面")
            if document.page_count > self.config.max_pages:
                raise PDFValidationError(
                    f"PDF 共 {document.page_count} 页，超过 {self.config.max_pages} 页限制"
                )
            return [self._parse_page(page, index + 1) for index, page in enumerate(document)]
        finally:
            document.close()

    def _parse_page(self, page: pymupdf.Page, page_number: int) -> ParsedPage:
        native_text = clean_text(page.get_text("text", sort=True))
        native_score = text_quality(native_text)
        native_chars = len("".join(native_text.split()))
        image_coverage = self._image_coverage(page)

        reliable_native = (
            native_chars >= self.config.native_min_chars
            and native_score >= self.config.native_min_quality
        )
        sparse_but_valid_native = (
            native_chars > 0
            and native_score >= self.config.native_min_quality
            and image_coverage < self.config.image_coverage_threshold
        )
        if reliable_native or sparse_but_valid_native:
            return ParsedPage(
                page_number=page_number,
                strategy="native",
                text=native_text,
                native_quality=native_score,
                image_coverage=image_coverage,
                quality_score=native_score,
            )

        ocr_text, ocr_confidence = self._run_ocr(page)
        ocr_score = text_quality(ocr_text)
        warnings: List[str] = []

        if native_text:
            # 混合页只选择质量更好的一路，禁止直接拼接导致重复知识。
            native_rank = native_score
            ocr_rank = ocr_score * (0.7 + 0.3 * ocr_confidence)
            if native_rank >= ocr_rank:
                chosen_text = native_text
                quality_score = native_score
                warnings.append(f"第 {page_number} 页 OCR 未优于原生文本，保留原生文本")
            else:
                chosen_text = ocr_text
                quality_score = ocr_rank
                warnings.append(f"第 {page_number} 页原生文本质量较低，采用 OCR 文本")
            strategy = "hybrid"
        else:
            chosen_text = ocr_text
            quality_score = ocr_score * (0.7 + 0.3 * ocr_confidence)
            strategy = "ocr"

        if not chosen_text:
            warnings.append(f"第 {page_number} 页未识别到有效文本")

        return ParsedPage(
            page_number=page_number,
            strategy=strategy,
            text=chosen_text,
            native_quality=native_score,
            ocr_confidence=round(ocr_confidence, 4),
            image_coverage=image_coverage,
            quality_score=round(quality_score, 4),
            warnings=warnings,
        )

    @staticmethod
    def _image_coverage(page: pymupdf.Page) -> float:
        """
        计算页面中图片块的面积之和，占整个页面面积的比例。
        """
        page_area = max(1.0, page.rect.width * page.rect.height)
        image_area = 0.0
        try:
            for block in page.get_text("blocks"):
                if len(block) > 6 and block[6] == 1:
                    rect = pymupdf.Rect(block[:4]) & page.rect
                    image_area += max(0.0, rect.width * rect.height)
        except Exception:
            return 0.0
        return round(min(1.0, image_area / page_area), 4)

    def _get_ocr_engine(self) -> Any:
        if self._ocr_engine is None:
            with self._ocr_lock:
                if self._ocr_engine is None:
                    if self._ocr_factory is not None:
                        self._ocr_engine = self._ocr_factory()
                    else:
                        from rapidocr import RapidOCR

                        self._ocr_engine = RapidOCR()
        return self._ocr_engine

    def _run_ocr(self, page: pymupdf.Page) -> Tuple[str, float]:
        """
        把一页PDF渲染成图片，交给OCR引擎识别，再将识别结果过滤、排序和清洗，返回页面文字及平均识别置信度。
        """
        #1.根据DPI计算页面放大倍数
        scale = self.config.ocr_dpi / 72.0
        #2.估算像素数量，必要时降低分辨率
        estimated_pixels = page.rect.width * page.rect.height * scale * scale
        if estimated_pixels > self.config.max_render_pixels:
            # 对异常大画布自动降低 DPI，避免单页渲染耗尽内存。
            scale *= math.sqrt(self.config.max_render_pixels / estimated_pixels)
        #3.将PDF页面渲染成RGB图片
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        #4.将像素字节转换成NumPy数组
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        #5.获取OCR引擎，并加锁执行
        engine = self._get_ocr_engine()
        with self._ocr_lock:
            output = engine(image)

        #6.提取OCR返回的文字、置信度和坐标
        texts = self._as_list(getattr(output, "txts", None))
        scores = [float(score) for score in self._as_list(getattr(output, "scores", None))]
        boxes = self._as_list(getattr(output, "boxes", None))

        # 兼容测试替身以及 RapidOCR 旧版的 (boxes, texts, scores) 返回值。
        if not texts and isinstance(output, (tuple, list)) and len(output) >= 3:
            boxes = self._as_list(output[0])
            texts = self._as_list(output[1])
            scores = self._as_list(output[2])
            scores = [float(score) for score in scores]

        #7.过滤低置信度和空文字
        lines = []
        for index, text in enumerate(texts):
            score = scores[index] if index < len(scores) else 1.0
            if score < self.config.ocr_min_confidence or not str(text).strip():
                continue
            box = boxes[index] if index < len(boxes) else None
            if box is not None:
                points = np.asarray(box, dtype=float)
                y = float(points[:, 1].min())
                x = float(points[:, 0].min())
            else:
                y, x = float(index), 0.0
            lines.append((y, x, str(text).strip(), score))

        lines.sort(key=lambda item: (round(item[0] / 10.0), item[1]))
        accepted_scores = [item[3] for item in lines]
        confidence = sum(accepted_scores) / len(accepted_scores) if accepted_scores else 0.0
        return clean_text("\n".join(item[2] for item in lines)), confidence

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        """将 RapidOCR 的列表或 NumPy 数组统一为列表，避免数组布尔判断报错。"""
        return [] if value is None else list(value)
