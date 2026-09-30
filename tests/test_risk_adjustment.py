import unittest

from inference.risk_adjustment import adjust_candidates, select_safer_alternative


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

    def test_safer_alternative_requires_material_risk_gain(self):
        alt = select_safer_alternative(
            [
                ("top", 1.00, 0.70),
                ("second", 0.99, 0.55),
                ("third", 0.90, 0.10),
            ],
            min_risk_gain=0.20,
            max_model_gap_fraction=0.35,
        )
        self.assertIsNone(alt)

    def test_safer_alternative_can_flag_close_low_risk_move(self):
        alt = select_safer_alternative(
            [
                ("top", 1.00, 0.80),
                ("second", 0.98, 0.35),
                ("third", 0.90, 0.20),
            ],
            min_risk_gain=0.20,
            max_model_gap_fraction=0.35,
        )
        self.assertIsNotNone(alt)
        self.assertEqual(alt.action, "second")
        self.assertEqual(alt.model_rank, 2)
        self.assertAlmostEqual(alt.risk_gain, 0.45)

    def test_safer_alternative_never_promotes_far_model_choice(self):
        alt = select_safer_alternative(
            [
                ("top", 1.00, 0.95),
                ("second", 0.80, 0.90),
                ("bottom", 0.00, 0.00),
            ],
            min_risk_gain=0.20,
            max_model_gap_fraction=0.35,
        )
        self.assertIsNone(alt)

    def test_safer_alternative_does_not_compare_pass_risk(self):
        alt = select_safer_alternative(
            [
                ("Pass", 1.00, None),
                ("A", 0.99, 0.00),
                ("K", 0.50, 0.00),
            ]
        )
        self.assertIsNone(alt)


if __name__ == "__main__":
    unittest.main()
