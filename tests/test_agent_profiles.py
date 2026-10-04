import unittest

from agents.agent_orchestrator import AgentProfile, ExplainAgent, QaAgent, QuizAgent, Request
from mcp.tool_manager import MCPToolManager, Tool
from tools.contracts import object_schema


class FakeClient:
    class Messages:
        async def create(self, **kwargs):  # pragma: no cover
            raise AssertionError("profile tests must not call the model")

    messages = Messages()


def make_manager():
    manager = MCPToolManager("test-key")
    for name in {
        *ExplainAgent.profile.tool_scope,
        *QaAgent.profile.tool_scope,
        *QuizAgent.profile.tool_scope,
    }:
        manager.register(Tool(
            name=name,
            description=f"test {name}",
            handler=lambda params, context: {"ok": True},
            schema=object_schema({}),
        ))
    manager.register(Tool(
        name="undeclared_admin_tool",
        description="must remain hidden",
        handler=lambda params, context: {"ok": True},
        schema=object_schema({}),
    ))
    return manager


class AgentProfileTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.manager = make_manager()
        self.explain = ExplainAgent(self.client, "test-model", tool_manager=self.manager)
        self.qa = QaAgent(self.client, "test-model", tool_manager=self.manager)
        self.quiz = QuizAgent(self.client, "test-model", tool_manager=self.manager)

    def test_agents_have_distinct_structured_role_contracts(self):
        profiles = [agent.profile for agent in (self.explain, self.qa, self.quiz)]
        self.assertTrue(all(isinstance(profile, AgentProfile) for profile in profiles))
        self.assertEqual(3, len({profile.role for profile in profiles}))
        self.assertEqual(3, len({profile.workflow for profile in profiles}))
        self.assertTrue(all(profile.responsibilities for profile in profiles))
        self.assertTrue(all(profile.input_contract for profile in profiles))
        self.assertTrue(all(profile.output_contract for profile in profiles))
        self.assertTrue(all(profile.escalation_conditions for profile in profiles))

    def test_profiles_define_different_generation_behavior(self):
        self.assertLess(self.qa.profile.temperature, self.explain.profile.temperature)
        self.assertGreater(self.quiz.profile.temperature, self.explain.profile.temperature)
        self.assertLess(self.quiz.profile.max_tokens, self.explain.profile.max_tokens)

    def test_system_prompt_contains_structured_contract(self):
        prompt = self.explain._build_system_prompt(
            Request(message="请讲解函数参数", user_id="u1", conv_id="c1")
        )
        self.assertIn("[结构化角色契约]", prompt)
        self.assertIn("职责：", prompt)
        self.assertIn("处理流程：", prompt)
        self.assertIn("输出要求：", prompt)
        self.assertIn("工具权限：", prompt)
        self.assertIn("check_prerequisites", prompt)
        self.assertNotIn("search_course_material", prompt)

    def test_manager_tools_are_filtered_by_each_profile_scope(self):
        for agent in (self.explain, self.qa, self.quiz):
            self.assertEqual(set(agent.profile.tool_scope), set(agent.get_tools()))
            self.assertNotIn("undeclared_admin_tool", agent.get_tools())

        self.assertIn("check_prerequisites", self.explain.get_tools())
        self.assertNotIn("grade_answer", self.explain.get_tools())
        self.assertIn("build_hint_plan", self.qa.get_tools())
        self.assertEqual({"check_prerequisites"}, set(self.explain.get_tools()))
        self.assertEqual({"build_hint_plan"}, set(self.qa.get_tools()))
        self.assertEqual(
            {"search_question_bank", "grade_answer", "record_learning_result"},
            set(self.quiz.get_tools()),
        )
        self.assertIn("grade_answer", self.quiz.get_tools())
        self.assertIn("record_learning_result", self.quiz.get_tools())


if __name__ == "__main__":
    unittest.main()
