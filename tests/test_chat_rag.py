import asyncio
import unittest
from unittest.mock import patch

from api import main


class FakeKnowledgeBase:
    def __init__(self):
        self.calls = []

    async def search_handler(self, params, context):
        self.calls.append(params)
        return [{
            "title": "函数参数",
            "content": "形参在定义中，实参在调用中。",
            "score": 0.9,
            "chunk": 0,
        }]


class FakeToolManager:
    async def rewrite_query(self, query, n=3):
        return [query, "Python 形参与实参"]


class FakeEvidenceGate:
    enabled = True

    def __init__(self):
        self.calls = []

    async def evaluate_all(self, query, passages, *, request_id=""):
        self.calls.append((query, list(passages), request_id))
        return []


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

    def should_use_rag(self, threshold):
        return False


class FakeDecisionEngine:
    async def decide(self, message, request_id=""):
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
        with (
            patch.object(main, "_knowledge_base", knowledge_base),
            patch.object(main, "_tool_manager", FakeToolManager()),
        ):
            results = asyncio.run(main._search_course_knowledge(
                "函数参数", 3, enhanced=True
            ))

        self.assertEqual(2, len(knowledge_base.calls))
        self.assertEqual(1, len(results))

    def test_evidence_shadow_does_not_filter_production_results(self):
        knowledge_base = FakeKnowledgeBase()
        evidence_gate = FakeEvidenceGate()
        with (
            patch.object(main, "_knowledge_base", knowledge_base),
            patch.object(main, "_evidence_gate", evidence_gate),
        ):
            results = asyncio.run(main._search_course_knowledge(
                "函数参数", 3, request_id="r1"
            ))

        self.assertEqual(1, len(results))
        self.assertEqual("形参在定义中，实参在调用中。", results[0]["content"])
        self.assertEqual("r1", evidence_gate.calls[0][2])

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
