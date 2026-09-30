import unittest

from inference.risk_adjustment import adjust_candidates


class RiskAdjustmentTests(unittest.TestCase):
    def test_zero_weight_preserves_model_order(self):
        result = adjust_candidates(
            [
                ("A", 1.0, 0.95),
                ("K", 0.8, 0.05),
                ("Q", 0.2, 0.00),
            ],
            weight=0.0,
        )
        ranks = {item.model_rank: item.adjusted_rank for item in result}
        self.assertEqual(ranks, {1: 1, 2: 2, 3: 3})

    def test_close_model_candidate_can_win_when_much_safer(self):
        result = adjust_candidates(
            [
                ("A", 1.00, 1.00),
                ("K", 0.99, 0.00),
                ("Q", 0.00, 0.00),
            ],
            weight=0.30,
        )
        winner = next(item for item in result if item.adjusted_rank == 1)
        self.assertEqual(winner.action, "K")
        self.assertEqual(winner.model_rank, 2)

    def test_large_model_gap_still_dominates_with_default_weight(self):
        result = adjust_candidates(
            [
                ("A", 1.0, 1.0),
                ("K", 0.2, 0.0),
                ("Q", 0.0, 0.0),
            ],
            weight=0.30,
        )
        winner = next(item for item in result if item.adjusted_rank == 1)
        self.assertEqual(winner.action, "A")

    def test_pass_uses_neutral_safety_component(self):
        result = adjust_candidates(
            [
                ("Pass", 0.9, None),
                ("A", 0.8, 0.0),
                ("K", 0.0, 0.0),
            ],
            weight=0.30,
        )
        pass_item = next(item for item in result if item.action == "Pass")
        self.assertIsNone(pass_item.response_risk)
        self.assertGreater(pass_item.adjusted_score, 0.0)
        self.assertLess(pass_item.adjusted_score, 100.0)

    def test_worst_model_candidate_cannot_win_on_safety_alone(self):
        result = adjust_candidates(
            [
                ("model_top", 1.0, 1.0),
                ("middle", 0.5, 0.5),
                ("model_bottom", 0.0, 0.0),
            ],
            weight=99.0,
        )
        winner = next(item for item in result if item.adjusted_rank == 1)
        self.assertEqual(winner.action, "model_top")

    def test_weight_is_clamped_below_full_replacement(self):
        result = adjust_candidates(
            [
                ("A", 1.0, 1.0),
                ("K", 0.99, 0.0),
                ("Q", 0.0, 0.0),
            ],
            weight=5.0,
        )
        # The function accepts the request but caps the heuristic contribution
        # below 50%, so DouZero remains the majority signal.
        top = next(item for item in result if item.adjusted_rank == 1)
        self.assertEqual(top.action, "K")


if __name__ == "__main__":
    unittest.main()
