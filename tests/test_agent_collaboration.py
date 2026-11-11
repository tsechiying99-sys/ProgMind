import asyncio
import unittest
from unittest.mock import AsyncMock

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentStats,
    AgentType,
    Request,
)
from core.decision_models import AgentTask, CollaborationMode, CollaborationPlan
from core.intent_recognizer import IntentCategory, UrgencyLevel


class RecordingAgent:
    def __init__(self, agent_type, content):
        self.agent_type = agent_type
        self.content = content
        self.requests = []
        self.stats = AgentStats()

    async def handle(self, req):
        self.requests.append(req)
        return AgentResponse(
            agent_type=self.agent_type,
            content=self.content,
            success=True,
        )


class AgentCollaborationTests(unittest.TestCase):
    def make_orchestrator(self):
        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        orchestrator._pool = {
            AgentType.GENERAL: [RecordingAgent(AgentType.GENERAL, "通用回答")],
            AgentType.EXPLAIN: [RecordingAgent(AgentType.EXPLAIN, "讲解结果")],
            AgentType.QA: [RecordingAgent(AgentType.QA, "诊断结果")],
            AgentType.QUIZ: [RecordingAgent(AgentType.QUIZ, "练习结果")],
        }
        return orchestrator

    @staticmethod
    def make_request(plan):
        return Request(
            message="请分两个阶段帮助我",
            user_id="u1",
            conv_id="c1",
            intent=IntentCategory.EXPLAIN,
            urgency=UrgencyLevel.LOW,
            collaboration_plan=plan,
        )

    def test_collaboration_uses_structured_plan_without_keyword_scan(self):
        orchestrator = self.make_orchestrator()
        request = self.make_request(CollaborationPlan(
            mode=CollaborationMode.PARALLEL,
            tasks=(
                AgentTask("explain", "说明核心概念"),
                AgentTask("qa", "检查理解障碍"),
            ),
            confidence=0.9,
        ))

        tasks = orchestrator._collaboration_tasks(request)

        self.assertEqual([AgentType.EXPLAIN, AgentType.QA], [item[0] for item in tasks])

    def test_low_confidence_plan_falls_back_to_single_agent(self):
        orchestrator = self.make_orchestrator()
        request = self.make_request(CollaborationPlan(
            mode=CollaborationMode.PARALLEL,
            tasks=(AgentTask("explain", "讲解"), AgentTask("quiz", "出题")),
            confidence=0.4,
        ))

        self.assertEqual([], orchestrator._collaboration_tasks(request))

    def test_parallel_results_are_synthesized_instead_of_concatenated(self):
        orchestrator = self.make_orchestrator()
        orchestrator._synthesize_collaboration = AsyncMock(return_value="统一整合答案")
        plan = CollaborationPlan(
            mode=CollaborationMode.PARALLEL,
            tasks=(AgentTask("explain", "讲解"), AgentTask("qa", "诊断")),
            confidence=0.9,
        )
        request = self.make_request(plan)

        result = asyncio.run(orchestrator.run(request))

        self.assertEqual("统一整合答案", result.response)
        self.assertNotIn("讲解结果\n\n诊断结果", result.response)
        orchestrator._synthesize_collaboration.assert_awaited_once()

    def test_sequential_task_receives_previous_result(self):
        orchestrator = self.make_orchestrator()
        orchestrator._synthesize_collaboration = AsyncMock(return_value="讲解后练习")
        plan = CollaborationPlan(
            mode=CollaborationMode.SEQUENTIAL,
            tasks=(
                AgentTask("explain", "先讲解"),
                AgentTask("quiz", "再出题", (0,)),
            ),
            confidence=0.95,
        )
        request = self.make_request(plan)

        asyncio.run(orchestrator.run(request))

        quiz_request = orchestrator._pool[AgentType.QUIZ][0].requests[0]
        self.assertIn("讲解结果", quiz_request.context)
        self.assertEqual("再出题", quiz_request.message)


if __name__ == "__main__":
    unittest.main()
