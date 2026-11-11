import asyncio
import unittest
from types import SimpleNamespace

from mcp.tool_manager import MCPToolManager, Tool
from tools import ToolDependencies, ToolExecutionContext, register_learning_tools
from tools.contracts import object_schema


class FakeKnowledgeBase:
    async def search_handler(self, params, context):
        return [{
            "title": "函数练习题",
            "content": "题目：说明形参与实参的区别。",
            "score": 0.95,
        }]


class FakeQuestionKnowledgeBase:
    async def search_handler(self, params, context):
        return [
            {
                "id": "a", "title": "函数练习题", "content": "背景型题目",
                "score": 0.9, "fusion_score": 0.9,
            },
            {
                "id": "b", "title": "参数习题", "content": "直接考查形参与实参",
                "score": 0.8, "fusion_score": 0.8,
            },
        ]


class FakeMemory:
    def __init__(self):
        self.records = []

    async def get_profile(self, user_id):
        return {
            "learning_progress": ["函数基础"],
            "weak_points": ["参数传递"],
            "recent_mistakes": ["混淆形参与实参"],
        }

    async def record_learning_result(self, user_id, conv_id, record):
        saved = {"user_id": user_id, "conv_id": conv_id, **record}
        self.records.append(saved)
        return {"saved": True, "record": saved}


class FakeMessages:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def create(self, **kwargs):
        text = self.outputs.pop(0)
        return SimpleNamespace(content=[{"type": "text", "text": text}])


class FakeLLMClient:
    def __init__(self, outputs):
        self.messages = FakeMessages(outputs)


def context():
    return ToolExecutionContext(
        request_id="r1",
        user_id="u1",
        conv_id="c1",
        message="函数参数",
        agent_type="quiz",
    )


class ToolManagerTests(unittest.TestCase):
    def test_registration_rejects_duplicate_names(self):
        manager = MCPToolManager("test-key")
        tool = Tool("demo", "demo", lambda params, ctx: {}, object_schema({}))
        manager.register(tool)
        with self.assertRaisesRegex(ValueError, "已注册"):
            manager.register(tool)

    def test_sync_and_async_handlers_share_the_same_call_path(self):
        manager = MCPToolManager("test-key")
        manager.register(Tool("sync", "sync", lambda p, c: {"value": 1}, object_schema({})))

        async def async_handler(params, ctx):
            return {"value": 2}

        manager.register(Tool("async", "async", async_handler, object_schema({})))
        self.assertEqual(1, asyncio.run(manager.call("sync", {})).data["value"])
        self.assertEqual(2, asyncio.run(manager.call("async", {})).data["value"])

    def test_registry_contains_exactly_five_agent_tools(self):
        memory = FakeMemory()
        manager = MCPToolManager("test-key")
        register_learning_tools(manager, ToolDependencies(FakeKnowledgeBase(), memory))
        expected = {
            "check_prerequisites", "build_hint_plan", "search_question_bank",
            "grade_answer", "record_learning_result",
        }
        self.assertEqual(expected, set(manager.registered_names))

        grade = asyncio.run(manager.call("grade_answer", {
            "student_answer": "形参在函数定义中，实参在调用时传入",
            "reference_answer": "形参与实参不同",
            "key_points": ["形参", "实参"],
        }, context()))
        self.assertTrue(grade.success)
        self.assertTrue(grade.data["correct"])

        saved = asyncio.run(manager.call("record_learning_result", {
            "topic": "函数参数",
            "question": "形参与实参有什么区别？",
            "student_answer": "不知道",
            "score": 10,
            "correct": False,
        }, context()))
        self.assertTrue(saved.success)
        self.assertEqual(1, len(memory.records))

    def test_recursive_validation_and_bounds(self):
        manager = MCPToolManager("test-key")
        register_learning_tools(manager, ToolDependencies(FakeKnowledgeBase(), FakeMemory()))
        boolean_integer = asyncio.run(manager.call(
            "search_question_bank", {"query": "函数", "top_k": True}
        ))
        out_of_range = asyncio.run(manager.call(
            "grade_answer", {
                "student_answer": "A", "reference_answer": "A", "pass_threshold": 2.0,
            }
        ))
        invalid_item = asyncio.run(manager.call(
            "grade_answer", {
                "student_answer": "A", "reference_answer": "A", "key_points": [123],
            }
        ))
        self.assertFalse(boolean_integer.success)
        self.assertFalse(out_of_range.success)
        self.assertFalse(invalid_item.success)

    def test_llm_rerank_changes_rrf_order_without_rewriting_content(self):
        manager = MCPToolManager("test-key")
        manager._client = FakeLLMClient([
            '{"ranking":['
            '{"index":1,"score":0.98,"reason":"直接回答问题"},'
            '{"index":0,"score":0.40,"reason":"仅提供背景"}'
            ']}'
        ])
        candidates = [
            {
                "id": "a", "title": "背景", "content": "背景原文",
                "score": 0.9, "fusion_score": 0.9,
            },
            {
                "id": "b", "title": "答案", "content": "答案原文",
                "score": 0.8, "fusion_score": 0.8,
            },
        ]

        result = asyncio.run(manager.rerank_results("目标问题", candidates, 2))

        self.assertEqual(["b", "a"], [item["id"] for item in result])
        self.assertEqual(["答案原文", "背景原文"], [item["content"] for item in result])
        self.assertTrue(all(item["rerank_applied"] for item in result))
        self.assertEqual([2, 1], [item["rrf_rank"] for item in result])
        self.assertEqual([0.98, 0.4], [item["score"] for item in result])
        self.assertEqual([0.8, 0.9], [item["rrf_score"] for item in result])

    def test_llm_rerank_failure_falls_back_to_rrf_order(self):
        manager = MCPToolManager("test-key")
        manager._client = FakeLLMClient(["not-json", "still-not-json"])
        candidates = [
            {"id": "a", "content": "A", "score": 0.9, "fusion_score": 0.9},
            {"id": "b", "content": "B", "score": 0.7, "fusion_score": 0.7},
        ]

        result = asyncio.run(manager.rerank_results("问题", candidates, 2))

        self.assertEqual(["a", "b"], [item["id"] for item in result])
        self.assertTrue(all(item["rerank_applied"] is False for item in result))

    def test_question_search_reranks_filtered_rrf_candidates(self):
        manager = MCPToolManager("test-key")
        manager._client = FakeLLMClient([
            '{"ranking":['
            '{"index":1,"score":0.96,"reason":"直接匹配参数问题"},'
            '{"index":0,"score":0.50,"reason":"相关但较宽泛"}'
            ']}'
        ])
        register_learning_tools(
            manager,
            ToolDependencies(FakeQuestionKnowledgeBase(), FakeMemory()),
        )

        result = asyncio.run(manager.call(
            "search_question_bank",
            {"query": "形参与实参", "top_k": 2},
            context(),
            use_cache=False,
        ))

        self.assertTrue(result.success)
        self.assertEqual(
            ["b", "a"],
            [item["id"] for item in result.data["questions"]],
        )


if __name__ == "__main__":
    unittest.main()
