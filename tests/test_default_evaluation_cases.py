import unittest
from collections import Counter

from evaluation.default_cases import JAVA_20_TURN_DIALOG, JAVA_INTENT_CASE_DATA
from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES


class DefaultEvaluationCasesTests(unittest.TestCase):
    def test_java_intent_dataset_adds_exactly_one_hundred_unique_cases(self):
        self.assertEqual(100, len(JAVA_INTENT_CASE_DATA))
        self.assertEqual(
            100,
            len({message for message, _ in JAVA_INTENT_CASE_DATA}),
        )
        self.assertEqual(105, len(DEFAULT_INTENT_CASES))

    def test_java_intent_dataset_covers_every_supported_intent(self):
        counts = Counter(intent for _, intent in JAVA_INTENT_CASE_DATA)
        self.assertEqual({
            "explain": 20,
            "qa": 20,
            "quiz": 20,
            "review": 15,
            "greeting": 15,
            "other": 10,
        }, dict(counts))

    def test_java_dialog_has_twenty_structured_learning_turns(self):
        turns = JAVA_20_TURN_DIALOG["turns"]
        self.assertEqual(20, len(turns))
        self.assertTrue(all(turn.get("message") for turn in turns))
        self.assertTrue(all(turn.get("expected_intent") for turn in turns))
        self.assertTrue(all(turn.get("expected_agent_type") for turn in turns))
        self.assertTrue(all("reference_answer" in turn for turn in turns))
        self.assertIn(JAVA_20_TURN_DIALOG, DEFAULT_DIALOG_CASES)


if __name__ == "__main__":
    unittest.main()
