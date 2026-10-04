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


if __name__ == "__main__":
    unittest.main()
