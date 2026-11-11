import asyncio
import unittest

from core.decision_engine import DecisionEngine, SemanticCollaborationPlanner
from core.decision_models import AgentTask, CollaborationMode, CollaborationPlan
from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel


class FakeRecognizer:
    def __init__(self, intent, confidence, urgency=UrgencyLevel.LOW):
        self.result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            entities={},
            reasoning="测试决策",
            latency_ms=1.0,
        )
        self.received_history = None

    async def recognize(self, message, history=None):
        self.received_history = history
        return self.result


class FakePlanner:
    def __init__(self, plan):
        self.result = plan
        self.calls = []

    async def plan(self, message, *, history, primary_intent):
        self.calls.append((message, history, primary_intent))
        return self.result


class DecisionEngineTests(unittest.TestCase):
    def test_semantic_plan_is_attached_and_receives_history(self):
        plan = CollaborationPlan(
            mode=CollaborationMode.SEQUENTIAL,
            tasks=(
                AgentTask("explain", "讲解继承"),
                AgentTask("quiz", "基于讲解出题", (0,)),
            ),
            confidence=0.92,
            reason="出题依赖讲解",
        )
        recognizer = FakeRecognizer(IntentCategory.EXPLAIN, 0.9)
        planner = FakePlanner(plan)
        engine = DecisionEngine(recognizer, planner)
        history = [{"role": "user", "content": "我在学继承"}]

        decision = asyncio.run(engine.decide("继续并测试我", history=history))

        self.assertEqual(plan, decision.collaboration)
        self.assertEqual(history, recognizer.received_history)
        self.assertEqual(history, planner.calls[0][1])
        self.assertEqual("intent_recognizer_and_semantic_planner", decision.source)

    def test_low_confidence_requires_clarification_and_skips_planner(self):
        planner = FakePlanner(CollaborationPlan())
        engine = DecisionEngine(
            FakeRecognizer(IntentCategory.QA, 0.4),
            planner,
            confidence_threshold=0.55,
        )
        decision = asyncio.run(engine.decide("这个怎么办"))

        self.assertTrue(decision.needs_clarification)
        self.assertFalse(decision.should_use_rag(0.55))
        self.assertEqual([], planner.calls)
        self.assertEqual(CollaborationMode.SINGLE, decision.collaboration.mode)

    def test_greeting_skips_rag(self):
        planner = FakePlanner(CollaborationPlan())
        engine = DecisionEngine(
            FakeRecognizer(IntentCategory.GREETING, 0.95),
            planner,
        )
        decision = asyncio.run(engine.decide("你好"))

        self.assertFalse(decision.needs_clarification)
        self.assertFalse(decision.should_use_rag(0.55))

    def test_invalid_general_collaboration_is_reduced_to_single(self):
        plan = SemanticCollaborationPlanner._parse_plan(
            {
                "mode": "parallel",
                "confidence": 0.9,
                "tasks": [
                    {"agent_type": "general", "instruction": "总结"},
                    {"agent_type": "quiz", "instruction": "出题"},
                ],
            },
            "总结后出题",
            IntentCategory.REVIEW,
        )

        self.assertEqual(CollaborationMode.SINGLE, plan.mode)
        self.assertEqual(1, len(plan.tasks))


if __name__ == "__main__":
    unittest.main()
