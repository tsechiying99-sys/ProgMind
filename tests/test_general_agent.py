import asyncio
import unittest

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentStats,
    AgentType,
    Request,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel


class StubAgent:
    def __init__(self, agent_type, *, success=True, content="通用回答"):
        self.agent_type = agent_type
        self.success = success
        self.content = content
        self.stats = AgentStats()

    async def handle(self, req):
        return AgentResponse(
            agent_type=self.agent_type,
            content=self.content,
            success=self.success,
        )


class GeneralAgentFallbackTests(unittest.TestCase):
    def make_orchestrator(self):
        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        orchestrator._pool = {
            AgentType.GENERAL: [StubAgent(AgentType.GENERAL)],
            AgentType.EXPLAIN: [
                StubAgent(AgentType.EXPLAIN, success=False, content="讲解失败")
            ],
        }
        return orchestrator

    def test_review_greeting_other_and_critical_use_general(self):
        orchestrator = self.make_orchestrator()
        for intent in (
            IntentCategory.REVIEW,
            IntentCategory.GREETING,
            IntentCategory.OTHER,
        ):
            self.assertEqual(
                AgentType.GENERAL,
                orchestrator._route(intent, UrgencyLevel.LOW),
            )
        self.assertEqual(
            AgentType.GENERAL,
            orchestrator._route(IntentCategory.EXPLAIN, UrgencyLevel.CRITICAL),
        )

    def test_failed_specialist_falls_back_to_general(self):
        orchestrator = self.make_orchestrator()
        request = Request(
            message="请解释函数",
            user_id="u1",
            conv_id="c1",
            intent=IntentCategory.EXPLAIN,
        )
        response = asyncio.run(orchestrator._execute(request, AgentType.EXPLAIN))

        self.assertTrue(response.success)
        self.assertEqual(AgentType.GENERAL, response.agent_type)
        self.assertEqual("通用回答", response.content)

    def test_clarification_request_uses_general_agent(self):
        orchestrator = self.make_orchestrator()
        request = Request(
            message="这个怎么做",
            user_id="u1",
            conv_id="c1",
            intent=IntentCategory.EXPLAIN,
            urgency=UrgencyLevel.LOW,
            needs_clarification=True,
        )
        result = asyncio.run(orchestrator.run(request))

        self.assertEqual(AgentType.GENERAL, result.agent_type)
        self.assertEqual("通用回答", result.response)


if __name__ == "__main__":
    unittest.main()
