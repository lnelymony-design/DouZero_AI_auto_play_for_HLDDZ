import unittest

from inference.decision_policy import choose_recommendation


class DecisionPolicyTests(unittest.TestCase):
    def test_env_override_has_priority(self):
        result = choose_recommendation(
            [("A", 1.0, 0.8), ("K", 0.99, 0.1)],
            ess_ratio=0.9,
            enabled=True,
            min_ess_ratio=0.45,
            min_risk_gain=0.25,
            max_model_gap_fraction=0.25,
            max_model_rank=3,
            locked_action="K",
            locked_model_rank=2,
        )
        self.assertEqual(result.action, "K")
        self.assertEqual(result.source, "env_override")
        self.assertTrue(result.changed_from_model_top)

    def test_low_ess_keeps_douzero(self):
        result = choose_recommendation(
            [("A", 1.0, 0.9), ("K", 0.99, 0.1)],
            ess_ratio=0.2,
            enabled=True,
            min_ess_ratio=0.45,
            min_risk_gain=0.25,
            max_model_gap_fraction=0.25,
            max_model_rank=3,
        )
        self.assertEqual(result.action, "A")
        self.assertEqual(result.source, "douzero")

    def test_materially_safer_close_candidate_can_be_promoted(self):
        result = choose_recommendation(
            [("A", 1.00, 0.80), ("K", 0.98, 0.30), ("Q", 0.90, 0.20)],
            ess_ratio=0.8,
            enabled=True,
            min_ess_ratio=0.45,
            min_risk_gain=0.25,
            max_model_gap_fraction=0.25,
            max_model_rank=3,
        )
        self.assertEqual(result.action, "K")
        self.assertEqual(result.source, "belief_safer")
        self.assertEqual(result.model_rank, 2)

    def test_pass_is_not_overridden(self):
        result = choose_recommendation(
            [("Pass", 1.0, None), ("A", 0.99, 0.0)],
            ess_ratio=0.9,
            enabled=True,
            min_ess_ratio=0.45,
            min_risk_gain=0.25,
            max_model_gap_fraction=0.25,
            max_model_rank=3,
        )
        self.assertEqual(result.action, "Pass")

    def test_bomb_is_not_promoted_from_non_bomb(self):
        bombs = {"3333"}
        result = choose_recommendation(
            [("A", 1.00, 0.90), ("3333", 0.99, 0.00), ("K", 0.80, 0.70)],
            ess_ratio=0.9,
            enabled=True,
            min_ess_ratio=0.45,
            min_risk_gain=0.25,
            max_model_gap_fraction=0.25,
            max_model_rank=3,
            is_bomb_action=lambda action: action in bombs,
        )
        self.assertEqual(result.action, "A")
        self.assertEqual(result.source, "douzero")


if __name__ == "__main__":
    unittest.main()
