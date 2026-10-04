"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 提供统一 search_handler，供聊天 RAG 预处理和题目检索工具复用

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional
from sentence_transformers import SentenceTransformer

import chromadb

import re
import threading

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class BGEEncoder:
    QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(self)-> None:
        self.model = SentenceTransformer(
            "BAAI/bge-small-zh-v1.5",
            device="cpu",
        )

    def encode_documents(self, texts:list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        text = self.QUERY_INSTRUCTION + query

        vectors = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        return vectors.tolist()

class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "learning_knowledge_base_bge_v1"

    BM25_STOP_WORDS = {
        "的", "了", "是", "在", "和", "与", "或",
        "为什么", "怎么", "如何", "什么", "哪些",
        "使用", "进行", "实现", "内容", "资料", "课程",
        "相关", "介绍", "说明",
    }

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # 优先连接独立 ChromaDB 服务
        self._use_server = False
        # 使用BAAI/bge-small-zh-v1.5模型
        self._encoder = BGEEncoder()

        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        self._collection = self._client.get_or_create_collection(
            name="learning_knowledge_base_bge_v1",
            metadata={
                "description": "ProgMind 中文课程知识库",
                "embedding_model": "BAAI/bge-small-zh-v1.5",
                "hnsw:space": "cosine",
            },
        )

        # 取得已有 Collection 后检查
        metadata = self._collection.metadata or {}
        if metadata.get("embedding_model") != "BAAI/bge-small-zh-v1.5":
            raise RuntimeError("Collection 的 Embedding 模型不匹配")
        if metadata.get("hnsw:space") != "cosine":
            raise RuntimeError("Collection 距离类型不是 cosine")

        # 如果知识库为空，提示导入文档
        if self._collection.count() == 0:
            logger.info("课程知识库为空，请通过 /knowledge 接口导入课程资料")

        self._bm25_lock = threading.RLock()
        self._bm25: BM25Okapi | None = None
        self._bm25_records: list[dict[str, Any]] = []

        self._rebuild_bm25_index()

        if metadata.get("hnsw:space") != "cosine":
            raise RuntimeError("Collection 距离类型不是 cosine")

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """用于中文和代码标识符分词"""
        #统一小写
        normalized = (text or "").strip().lower()

        if not normalized:
            return []

        #进行中文分词
        candidates = list(
            jieba.lcut(normalized, cut_all=False)
        )
        #额外提取英文、代码标识符和数字
        candidates.extend(
            re.findall(
                r"\*{0,2}[a-zA-Z_][a-zA-Z0-9_]*|\d+(?:\.\d+)?",
                normalized,
            )
        )

        tokens: list[str] = []

        for token in candidates:
            token = token.strip()
            if not token:
                continue
            # 删除标点符号以及单独的下划线
            if not re.search(
                    r"[\u4e00-\u9fffA-Za-z0-9]",
                    token,
            ):
                continue
            # 删除停用词
            if token in cls.BM25_STOP_WORDS:
                continue
            # 去重
            if token not in tokens:
                tokens.append(token)

        return tokens

    def _rebuild_bm25_index(self) -> None:
        """从 ChromaDB 已有文档重建内存 BM25 索引。"""
        snapshot = self._collection.get(
            include=["documents", "metadatas"],
        )

        ids = snapshot.get("ids") or []
        documents = snapshot.get("documents") or []
        metadatas = snapshot.get("metadatas") or []

        records: list[dict[str, Any]] = []
        corpus: list[list[str]] = []

        for doc_id, document, metadata in zip(
                ids,
                documents,
                metadatas,
        ):
            content = str(document or "").strip()

            if not content:
                continue

            tokens = self._tokenize(content)

            if not tokens:
                continue

            metadata = metadata or {}

            records.append({
                "id": doc_id,
                "title": metadata.get("title", ""),
                "content": content,
                "chunk": metadata.get("chunk_index", 0),
                "_token_set": frozenset(tokens),
            })
            corpus.append(tokens)

        bm25 = BM25Okapi(corpus) if corpus else None

        with self._bm25_lock:
            self._bm25 = bm25
            self._bm25_records = records

        logger.info(
            "BM25 索引已重建: %d 个文档片段",
            len(records),
        )

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(
                    f"{title}_{i}_{chunk}".encode("utf-8")
                ).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({"title": title, "chunk_index": i, "total_chunks": len(chunks)})

        if ids:
            embeddings = self._encoder.encode_documents(docs)
            self._collection.upsert(ids=ids, documents=docs, metadatas=metas ,embeddings=embeddings)
            self._rebuild_bm25_index()
            logger.info("知识库导入 %d 个文档片段，BM25 索引已刷新",len(ids),)

        return len(ids)

    def _dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query or top_k <= 0:
            return []

        document_count = self._collection.count()
        if document_count <= 0:
            return []
        try:
            min_score = float(
                os.getenv("KNOWLEDGE_MIN_SIMILARITY", "0.25")
            )
        except ValueError:
            min_score = 0.25

        # 多召回一些，过滤后最多返回 top_k。
        recall_k = min(max(top_k * 3, top_k), document_count)

        # 将问题向量化
        query_embedding = self._encoder.encode_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=recall_k,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results["documents"] and results["documents"][0]:
            result_ids = (results["ids"][0] if results.get("ids") else [])
            for doc_id, doc, meta, dist in zip(
                result_ids,
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = round(1.0 - float(dist), 4)
                if score < min_score:
                    continue
                items.append({
                    "id": doc_id,
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    score,
                    "dense_score": score,
                    "bm25_score": 0.0,
                    "chunk":    meta.get("chunk_index", 0),
                    "match_sources": ["dense"],
                })
                if len(items) >= top_k:
                    break

        return items

    def _bm25_search(
            self,
            query: str,
            top_k: int,
    ) -> list[dict[str, Any]]:
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        try:
            min_score = float(
                os.getenv("BM25_MIN_SCORE", "0.5")
            )
        except ValueError:
            min_score = 0.5

        try:
            min_idf = float(
                os.getenv("BM25_MIN_IDF", "1.0")
            )
        except ValueError:
            min_idf = 1.0

        with self._bm25_lock:
            bm25 = self._bm25
            records = list(self._bm25_records)

        if bm25 is None or not records:
            return []

        scores = bm25.get_scores(query_tokens)

        # BM25 可以独立补回向量检索容易漏掉的代码标识符，例如
        # UnboundLocalError、nonlocal、row_number、*args。普通自然语言词项
        # 仍需 Dense 召回提供语义佐证，避免“保存、调用”等词造成假阳性。
        exact_identifiers = {
            token.lower()
            for token in re.findall(
                r"\*{1,2}[A-Za-z_][A-Za-z0-9_]*"
                r"|[A-Za-z_][A-Za-z0-9_]*",
                query,
            )
        }

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        results: list[dict[str, Any]] = []

        for index in ranked_indexes:
            raw_score = float(scores[index])

            # 没有可靠关键词重合的结果不进入融合
            if raw_score < min_score:
                continue

            record = records[index]

            query_token_set = set(query_tokens)
            document_token_set = set(
                record.get("_token_set", ())
            )

            matched_tokens = (
                    query_token_set & document_token_set
            )

            if not matched_tokens:
                continue

            # 至少有一个高区分度词项真正出现在文档中
            max_matched_idf = max(
                float(bm25.idf.get(token, 0.0))
                for token in matched_tokens
            )

            if max_matched_idf < min_idf:
                continue

            matched_identifiers = matched_tokens & exact_identifiers

            public_record = {
                key: value
                for key, value in record.items()
                if key != "_token_set"
            }

            results.append({
                **public_record,
                "score": raw_score,
                "dense_score": 0.0,
                "bm25_score": round(raw_score, 4),
                "bm25_max_idf": round(max_matched_idf, 4),
                "matched_terms": sorted(matched_tokens),
                "matched_identifiers": sorted(matched_identifiers),
                "exact_identifier_match": bool(matched_identifiers),
                "match_sources": ["bm25"],
            })

            if len(results) >= top_k:
                break

        return results

    def search(
            self,
            query: str,
            top_k: int = 5,
            retrieval_mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()

        if not query or top_k <= 0:
            return []

        recall_k = max(top_k * 3, 10)

        dense_results = self._dense_search(
            query,
            recall_k,
        )

        if retrieval_mode == "dense":
            return dense_results[:top_k]

        bm25_results = self._bm25_search(
            query,
            recall_k,
        )

        rrf_k = 60
        dense_weight = float(
            os.getenv("HYBRID_DENSE_WEIGHT", "0.7")
        )
        bm25_weight = float(
            os.getenv("HYBRID_BM25_WEIGHT", "0.3")
        )

        merged: dict[str, dict[str, Any]] = {}

        def add_result(
                item: dict[str, Any],
                rank: int,
                source: str,
                weight: float,
        ) -> None:
            doc_id = str(item.get("id", ""))

            if not doc_id:
                doc_id = hashlib.md5(
                    (
                        str(item.get("title", "")) + str(item.get("chunk", 0)) + str(item.get("content", ""))
                    ).encode("utf-8")
                ).hexdigest()

            existing = merged.setdefault(
                doc_id,
                {
                    **item,
                    "rrf_raw": 0.0,
                    "dense_score": 0.0,
                    "bm25_score": 0.0,
                    "match_sources": [],
                },
            )

            existing["rrf_raw"] += weight / (rrf_k + rank)

            if source == "dense":
                existing["dense_score"] = float(
                    item.get("dense_score", item.get("score", 0.0))
                )
            else:
                existing["bm25_score"] = float(
                    item.get("bm25_score", item.get("score", 0.0))
                )
                existing["exact_identifier_match"] = bool(
                    existing.get("exact_identifier_match", False)
                    or item.get("exact_identifier_match", False)
                )

                matched_identifiers = set(
                    existing.get("matched_identifiers", [])
                )
                matched_identifiers.update(
                    item.get("matched_identifiers", [])
                )
                existing["matched_identifiers"] = sorted(
                    matched_identifiers
                )

            if source not in existing["match_sources"]:
                existing["match_sources"].append(source)

        for rank, item in enumerate(dense_results, start=1):
            add_result(
                item,
                rank,
                "dense",
                dense_weight,
            )

        for rank, item in enumerate(bm25_results, start=1):
            add_result(
                item,
                rank,
                "bm25",
                bm25_weight,
            )

        max_rrf = (
                          dense_weight + bm25_weight
                  ) / (rrf_k + 1)

        results = []

        for item in merged.values():
            sources = set(item.get("match_sources", []))

            # BM25 单路结果只允许精确代码标识符命中。普通自然语言词项
            # 必须同时通过 Dense 的语义阈值，防止宽泛词带入无关片段。
            if (
                sources == {"bm25"}
                and not item.get("exact_identifier_match", False)
            ):
                continue

            fusion_score = (
                item.pop("rrf_raw") / max_rrf
                if max_rrf > 0
                else 0.0
            )

            item["score"] = round(fusion_score, 4)
            item["fusion_score"] = round(fusion_score, 4)
            item["dense_score"] = round(
                float(item["dense_score"]),
                4,
            )
            item["bm25_score"] = round(
                float(item["bm25_score"]),
                4,
            )

            results.append(item)

        results.sort(
            key=lambda item: float(
                item.get("fusion_score", 0.0)
            ),
            reverse=True,
        )

        return results[:top_k]

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    # ── MCP 工具 handler ─────────────────────────────────────────────────────
    async def search_handler(
            self,
            params: Dict[str, Any],
            context: Any,
    ) -> List[Dict]:
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        retrieval_mode = params.get(
            "retrieval_mode",
            "hybrid",
        )

        return await asyncio.to_thread(
            self.search,
            query,
            top_k,
            retrieval_mode,
        )

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) > chunk_size:
                if current:
                    chunks.append(current)
                    current = ""

                for start in range(0, len(sent), chunk_size):
                    piece = sent[start:start + chunk_size]
                    if len(piece) == chunk_size:
                        chunks.append(piece)
                    else:
                        current = piece
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks
