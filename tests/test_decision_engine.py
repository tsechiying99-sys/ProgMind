import asyncio
import unittest

from core.decision_engine import DecisionEngine
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

    async def recognize(self, message, history=None):
        return self.result


class FakeJevClient:
    enabled = True
    model = "jev-test"

    async def evaluate(self, *, state, questions):
        return {
            "model": self.model,
            "answers": {
                "intent": {
                    "choice": "quiz",
                    "confidence": 0.93,
                    "probabilities": {"quiz": 0.93, "explain": 0.07},
                },
                "urgency": {"score": 0.2, "confidence": 0.8},
                "needs_rag": {"noul": 0.7},
                "needs_clarification": {"noul": 0.1},
            },
        }


class DecisionEngineTests(unittest.TestCase):
    def test_jev_shadow_does_not_change_production_route(self):
        engine = DecisionEngine(
            FakeRecognizer(IntentCategory.EXPLAIN, 0.9),
            FakeJevClient(),
            shadow_enabled=True,
        )
        decision = asyncio.run(engine.decide("讲解函数", request_id="r1"))

        self.assertEqual(IntentCategory.EXPLAIN, decision.intent)
        self.assertEqual(IntentCategory.QUIZ, decision.jev_shadow.intent)
        self.assertEqual("existing_intent_recognizer", decision.source)

    def test_low_confidence_requires_clarification_and_blocks_rag(self):
        engine = DecisionEngine(
            FakeRecognizer(IntentCategory.QA, 0.4),
            confidence_threshold=0.55,
        )
        decision = asyncio.run(engine.decide("这个怎么办"))

        self.assertTrue(decision.needs_clarification)
        self.assertFalse(decision.should_use_rag(0.55))

    def test_greeting_skips_rag(self):
        engine = DecisionEngine(FakeRecognizer(IntentCategory.GREETING, 0.95))
        decision = asyncio.run(engine.decide("你好"))

        self.assertFalse(decision.needs_clarification)
        self.assertFalse(decision.should_use_rag(0.55))


if __name__ == "__main__":
    unittest.main()
