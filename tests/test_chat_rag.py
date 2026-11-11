import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from api import main
from core.decision_models import CollaborationPlan


class FakeKnowledgeBase:
    def __init__(self, results=None):
        self.calls = []
        self.results = results or [{
            "title": "函数参数",
            "content": "形参在定义中，实参在调用中。",
            "score": 0.9,
            "fusion_score": 0.9,
            "chunk": 0,
        }]

    async def search_handler(self, params, context):
        self.calls.append(params)
        return list(self.results)


class FakeToolManager:
    def __init__(self):
        self.rerank_calls = []

    async def rewrite_query(self, query, n=3):
        return [query, "Python 形参与实参"]

    async def rerank_results(self, query, items, top_k):
        self.rerank_calls.append((query, list(items), top_k))
        return [
            {**item, "rerank_applied": True, "rerank_score": 0.95}
            for item in items[:top_k]
        ]


class FakeMemoryContext:
    recent_messages = []

    def to_prompt_text(self):
        return "学习上下文"


class FakeMemory:
    async def get_context(self, user_id, conv_id, query):
        return FakeMemoryContext()


class FakeNoRagDecision:
    intent = None
    urgency = None
    intent_confidence = 0.4
    urgency_confidence = 0.0
    source = "test"
    needs_clarification = True
    collaboration = CollaborationPlan()

    def should_use_rag(self, threshold):
        return False


class FakeDecisionEngine:
    async def decide(self, message, history=None, request_id=""):
        return FakeNoRagDecision()


class ChatRagTests(unittest.TestCase):
    def test_fast_rag_calls_knowledge_base_without_agent_tool(self):
        knowledge_base = FakeKnowledgeBase()
        with patch.object(main, "_knowledge_base", knowledge_base):
            results = asyncio.run(main._search_course_knowledge("函数参数", 3))

        self.assertEqual(1, len(knowledge_base.calls))
        self.assertEqual("hybrid", knowledge_base.calls[0]["retrieval_mode"])
        self.assertEqual(1, len(results))

    def test_enhanced_rag_rewrites_and_deduplicates_results(self):
        knowledge_base = FakeKnowledgeBase()
        tool_manager = FakeToolManager()
        with (
            patch.object(main, "_knowledge_base", knowledge_base),
            patch.object(main, "_tool_manager", tool_manager),
        ):
            results = asyncio.run(main._search_course_knowledge(
                "函数参数", 3, enhanced=True
            ))

        self.assertEqual(2, len(knowledge_base.calls))
        self.assertEqual(1, len(results))
        self.assertEqual(1, len(tool_manager.rerank_calls))

    def test_llm_rerank_receives_rrf_sorted_candidates(self):
        knowledge_base = FakeKnowledgeBase(results=[
            {
                "title": "低相关片段", "content": "低相关内容",
                "score": 0.2, "fusion_score": 0.2, "chunk": 0,
            },
            {
                "title": "高相关片段", "content": "高相关内容",
                "score": 0.9, "fusion_score": 0.9, "chunk": 1,
            },
        ])
        tool_manager = FakeToolManager()
        with (
            patch.object(main, "_knowledge_base", knowledge_base),
            patch.object(main, "_tool_manager", tool_manager),
        ):
            asyncio.run(main._search_course_knowledge("函数参数", 1))

        candidates = tool_manager.rerank_calls[0][1]
        self.assertEqual([0.9, 0.2], [item["fusion_score"] for item in candidates])

    def test_search_reports_actual_rerank_state(self):
        with (
            patch.object(main, "_tool_manager", object()),
            patch.object(
                main,
                "_search_course_knowledge",
                AsyncMock(return_value=[{
                    "content": "答案", "score": 0.96, "rerank_applied": True,
                }]),
            ),
        ):
            response = asyncio.run(main.search("函数参数", 3))

        self.assertTrue(response["reranked"])

    def test_clarification_decision_skips_rag(self):
        knowledge_base = FakeKnowledgeBase()
        with (
            patch.object(main, "_memory", FakeMemory()),
            patch.object(main, "_decision_engine", FakeDecisionEngine()),
            patch.object(main, "_knowledge_base", knowledge_base),
        ):
            request, used = asyncio.run(main._prepare_chat_request(
                main.ChatRequest(message="这个怎么办", user_id="u1", conv_id="c1"),
                "c1",
            ))

        self.assertFalse(used)
        self.assertFalse(request.needs_rag)
        self.assertTrue(request.needs_clarification)
        self.assertEqual([], knowledge_base.calls)


if __name__ == "__main__":
    unittest.main()
