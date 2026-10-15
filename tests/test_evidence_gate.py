import asyncio
import unittest

from core.evidence_gate import JevEvidenceShadowGate


class FakeJevClient:
    enabled = True
    model = "jev-test"

    async def evaluate(self, *, state, questions):
        return {
            "answers": {
                "relevant": {"noul": 0.9},
                "answer_evidence": {"noul": 0.85},
                "contradiction": {"noul": 0.1},
                "prompt_injection": {"noul": 0.05},
            },
        }


class EvidenceGateTests(unittest.TestCase):
    def test_relevant_passage_is_labeled_as_evidence(self):
        gate = JevEvidenceShadowGate(FakeJevClient(), enabled=True)
        decisions = asyncio.run(gate.evaluate_all(
            "什么是形参",
            [{"title": "函数参数", "content": "形参在函数定义中出现。"}],
            request_id="r1",
        ))

        self.assertEqual(1, len(decisions))
        self.assertEqual("evidence", decisions[0].route)
        self.assertGreater(decisions[0].relevance, 0.8)

    def test_disabled_gate_does_not_call_jev(self):
        gate = JevEvidenceShadowGate(FakeJevClient(), enabled=False)
        decisions = asyncio.run(gate.evaluate_all(
            "什么是形参",
            [{"title": "函数参数", "content": "形参说明"}],
        ))

        self.assertEqual([], decisions)


if __name__ == "__main__":
    unittest.main()
