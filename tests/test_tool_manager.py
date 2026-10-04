import asyncio
import unittest

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


if __name__ == "__main__":
    unittest.main()
