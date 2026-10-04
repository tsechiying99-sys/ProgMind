import asyncio
import os
import unittest
from unittest.mock import patch

from agents.agent_orchestrator import ExplainAgent, QaAgent, Request
from mcp.tool_manager import MCPToolManager
from tools.build_hint_plan import create_tool as create_hint_tool
from tools.grade_answer import create_tool as create_grade_tool


class ToolUseBlock:
    type = "tool_use"

    def __init__(self, name, tool_input, tool_id="toolu_1"):
        self.id, self.name, self.input = tool_id, name, tool_input


class TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeToolClient:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    class Messages:
        def __init__(self, owner):
            self.owner = owner

        async def create(self, **kwargs):
            self.owner.calls.append(kwargs)
            return self.owner.responses.pop(0)

    @property
    def messages(self):
        return self.Messages(self)


def response(*blocks):
    return type("Response", (), {"content": list(blocks), "stop_reason": "end_turn"})()


def make_manager():
    manager = MCPToolManager("test-key")
    manager.register(create_hint_tool())
    manager.register(create_grade_tool())
    return manager


class AgentToolTests(unittest.TestCase):
    def setUp(self):
        self.req = Request(
            message="这道函数参数题我不知道下一步怎么做",
            user_id="u1",
            conv_id="c1",
        )

    def test_model_receives_only_current_agent_allowlist(self):
        client = FakeToolClient([
            response(ToolUseBlock("build_hint_plan", {
                "question": "函数参数题", "attempted_step": "读过题目",
            })),
            response(TextBlock("先检查实参与形参的对应关系。")),
        ])
        agent = QaAgent(client, "test-model", tool_manager=make_manager())
        result = asyncio.run(agent.handle(self.req))

        self.assertTrue(result.success)
        self.assertEqual(["build_hint_plan"], result.tools_used)
        visible = {tool["name"] for tool in client.calls[0]["tools"]}
        self.assertEqual({"build_hint_plan"}, visible)
        self.assertNotIn("grade_answer", visible)
        self.assertIn("tool_result", str(client.calls[1]["messages"]))

    def test_fabricated_out_of_scope_tool_is_rejected_server_side(self):
        client = FakeToolClient([
            response(ToolUseBlock("grade_answer", {
                "student_answer": "形参",
                "reference_answer": "实参",
            })),
            response(TextBlock("我会继续按讲解角色回答。")),
        ])
        agent = ExplainAgent(client, "test-model", tool_manager=make_manager())
        result = asyncio.run(agent.handle(self.req))

        self.assertTrue(result.success)
        self.assertEqual([], result.tools_used)
        self.assertFalse(result.tool_traces[0]["success"])
        self.assertIn("白名单", result.tool_traces[0]["error"])

    def test_manager_rejects_unknown_and_invalid_parameters(self):
        manager = make_manager()
        unknown = asyncio.run(manager.call(
            "build_hint_plan", {"question": "函数参数题", "secret": "not-allowed"}
        ))
        invalid = asyncio.run(manager.call("build_hint_plan", {"question": 123}))
        self.assertFalse(unknown.success)
        self.assertIn("不允许参数", unknown.error)
        self.assertFalse(invalid.success)
        self.assertIn("类型错误", invalid.error)

    def test_tool_loop_stops_at_configured_limit(self):
        client = FakeToolClient([
            response(ToolUseBlock("build_hint_plan", {"question": "函数参数题"}, "toolu_1")),
            response(ToolUseBlock("build_hint_plan", {"question": "函数参数题"}, "toolu_2")),
        ])
        agent = QaAgent(client, "test-model", tool_manager=make_manager())
        with patch.dict(os.environ, {"AGENT_MAX_TOOL_ROUNDS": "1"}):
            result = asyncio.run(agent.handle(self.req))
        self.assertFalse(result.success)
        self.assertEqual(2, len(client.calls))
        self.assertEqual(1, len(result.tool_traces))

    def test_removed_context_tools_are_not_registered(self):
        manager = make_manager()
        self.assertNotIn("search_course_material", manager.registered_names)
        self.assertNotIn("get_learning_profile", manager.registered_names)


if __name__ == "__main__":
    unittest.main()
