import asyncio
import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pymupdf
import numpy as np
from fastapi import HTTPException, UploadFile

from api import main
from ingestion.chunker import KnowledgeChunker
from ingestion.models import DocumentQualityError, PDFValidationError
from ingestion.pdf_parser import PDFParser, PDFParserConfig
from ingestion.pipeline import DocumentIngestionPipeline
from mcp.knowledge_base import KnowledgeBase


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def make_native_pdf(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(page.rect + (40, 40, -40, -40), text, fontsize=12)
    content = document.tobytes()
    document.close()
    return content


def make_mixed_pdf(native_text: str) -> bytes:
    document = pymupdf.open()
    native_page = document.new_page()
    native_page.insert_textbox(native_page.rect + (40, 40, -40, -40), native_text, fontsize=12)
    scan_page = document.new_page()
    scan_page.insert_image(scan_page.rect, stream=_ONE_PIXEL_PNG)
    content = document.tobytes()
    document.close()
    return content


def make_encrypted_pdf() -> bytes:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "protected")
    content = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    document.close()
    return content


class FakeOCREngine:
    def __init__(self, text="扫描页中的 Java 课程知识，包含变量、循环、方法和面向对象。", score=0.96):
        self.text = text
        self.score = score
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return SimpleNamespace(
            txts=[self.text],
            scores=[self.score],
            boxes=[[[0, 0], [100, 0], [100, 20], [0, 20]]],
        )


class PDFIngestionTests(unittest.TestCase):
    def setUp(self):
        self.config = PDFParserConfig(
            native_min_chars=20,
            native_min_quality=0.70,
            ocr_min_confidence=0.55,
        )

    def test_native_pdf_does_not_initialize_ocr(self):
        factory = Mock(side_effect=AssertionError("原生 PDF 不应初始化 OCR"))
        parser = PDFParser(self.config, ocr_factory=factory)

        pages = parser.parse(make_native_pdf("Java variables methods classes and exceptions. " * 4))

        self.assertEqual("native", pages[0].strategy)
        self.assertIn("Java", pages[0].text)
        factory.assert_not_called()

    def test_mixed_pdf_uses_ocr_only_for_scanned_page(self):
        engine = FakeOCREngine()
        parser = PDFParser(self.config, ocr_factory=lambda: engine)

        pages = parser.parse(make_mixed_pdf("Java native text methods classes. " * 4))

        self.assertEqual(["native", "ocr"], [page.strategy for page in pages])
        self.assertEqual(1, engine.calls)
        self.assertIn("扫描页", pages[1].text)
        self.assertGreater(pages[1].ocr_confidence, 0.9)

    def test_rapidocr_numpy_output_is_supported(self):
        engine = FakeOCREngine()
        engine.__class__ = type(
            "NumpyOutputEngine",
            (),
            {
                "__call__": lambda self, image: SimpleNamespace(
                    txts=np.asarray(["Java OCR content"]),
                    scores=np.asarray([0.97]),
                    boxes=np.asarray([[[0, 0], [100, 0], [100, 20], [0, 20]]]),
                )
            },
        )
        parser = PDFParser(self.config, ocr_factory=lambda: engine)

        pages = parser.parse(make_mixed_pdf(""))

        self.assertEqual("Java OCR content", pages[0].text)
        self.assertEqual("ocr", pages[0].strategy)

    def test_pipeline_builds_stable_ids_and_page_metadata(self):
        content = make_mixed_pdf("Java native text methods classes. " * 4)
        parser = PDFParser(self.config, ocr_factory=FakeOCREngine)
        pipeline = DocumentIngestionPipeline(
            parser=parser,
            chunker=KnowledgeChunker(chunk_size=250, overlap=30),
            document_min_quality=0.50,
        )

        first = pipeline.ingest_pdf(content, "Java课程.pdf")
        second = pipeline.ingest_pdf(content, "Java课程.pdf")

        self.assertEqual(
            [chunk.chunk_id for chunk in first.chunks],
            [chunk.chunk_id for chunk in second.chunks],
        )
        self.assertEqual(2, first.report.pages)
        self.assertEqual(1, first.report.native_pages)
        self.assertEqual(1, first.report.ocr_pages)
        self.assertEqual({1, 2}, {chunk.metadata["page_start"] for chunk in first.chunks})
        self.assertTrue(all(chunk.metadata["document_id"] == first.document_id for chunk in first.chunks))

    def test_invalid_pdf_is_rejected_before_ocr(self):
        parser = PDFParser(self.config, ocr_factory=FakeOCREngine)
        with self.assertRaises(PDFValidationError):
            parser.parse(b"not a pdf")

    def test_encrypted_pdf_is_rejected(self):
        parser = PDFParser(self.config, ocr_factory=FakeOCREngine)
        with self.assertRaisesRegex(PDFValidationError, "密码"):
            parser.parse(make_encrypted_pdf())

    def test_low_quality_ocr_text_is_not_admitted(self):
        parser = PDFParser(
            self.config,
            ocr_factory=lambda: FakeOCREngine(text="■■■■■■■■", score=0.99),
        )
        pipeline = DocumentIngestionPipeline(parser=parser, document_min_quality=0.55)
        with self.assertRaises(DocumentQualityError):
            pipeline.ingest_pdf(make_mixed_pdf(""), "garbage.pdf")


class KnowledgeChunkWriteTests(unittest.TestCase):
    def test_prechunked_records_keep_scalar_metadata(self):
        knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
        knowledge_base._encoder = Mock()
        knowledge_base._encoder.encode_documents.return_value = [[0.1, 0.2]]
        knowledge_base._collection = Mock()
        knowledge_base._rebuild_bm25_index = Mock()

        count = knowledge_base.add_chunks([{
            "id": "chunk-1",
            "content": "Java 变量知识",
            "metadata": {
                "page_start": 1,
                "ocr_used": True,
                "invalid_nested": {"x": 1},
                "invalid_none": None,
            },
        }])

        self.assertEqual(1, count)
        metadata = knowledge_base._collection.upsert.call_args.kwargs["metadatas"][0]
        self.assertEqual({"page_start": 1, "ocr_used": True}, metadata)
        knowledge_base._rebuild_bm25_index.assert_called_once()


class PDFUploadEndpointTests(unittest.TestCase):
    def test_pdf_upload_returns_ingestion_report(self):
        content = make_native_pdf("Java variables methods classes and exceptions. " * 4)
        config = PDFParserConfig(
            native_min_chars=20,
            native_min_quality=0.70,
            ocr_min_confidence=0.55,
        )
        pipeline = DocumentIngestionPipeline(
            parser=PDFParser(config, ocr_factory=FakeOCREngine),
            document_min_quality=0.50,
        )
        knowledge_base = Mock()
        knowledge_base.add_chunks.side_effect = lambda records: len(records)
        knowledge_base.doc_count = 12
        tool_manager = Mock()
        upload = UploadFile(filename="course.PDF", file=io.BytesIO(content))

        with (
            patch.object(main, "_ingestion_pipeline", pipeline),
            patch.object(main, "_knowledge_base", knowledge_base),
            patch.object(main, "_tool_manager", tool_manager),
        ):
            response = asyncio.run(main.upload_knowledge(upload))

        self.assertEqual(12, response["total_chunks"])
        self.assertEqual(1, response["native_pages"])
        self.assertIn("document_id", response)
        tool_manager.clear_cache.assert_called_once_with("search_question_bank")

    def test_unsupported_upload_is_rejected(self):
        upload = UploadFile(filename="course.docx", file=io.BytesIO(b"content"))
        with (
            patch.object(main, "_knowledge_base", Mock()),
            patch.object(main, "_tool_manager", Mock()),
        ):
            with self.assertRaises(HTTPException) as context:
                asyncio.run(main.upload_knowledge(upload))
        self.assertEqual(415, context.exception.status_code)


if __name__ == "__main__":
    unittest.main()
