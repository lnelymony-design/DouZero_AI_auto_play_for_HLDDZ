import unittest

from inference.engine import HandInferenceEngine


class HandInferenceEngineTests(unittest.TestCase):
    def test_revealed_bottom_card_is_forced_to_landlord(self):
        # Farmer perspective: D is a revealed bottom card, so before landlord
        # plays it the landlord must still own the big joker.
        engine = HandInferenceEngine(
            my_position="landlord_up",
            my_hand_cards="33445566778899TJQ",
            three_landlord_cards="D2A",
            sample_count=500,
            random_seed=7,
        )
        result = engine.infer()
        self.assertEqual(
            result["players"]["landlord"]["cards"]["D"]["one_plus"], 1.0
        )

    def test_played_bottom_card_is_no_longer_remaining(self):
        engine = HandInferenceEngine(
            my_position="landlord_up",
            my_hand_cards="33445566778899TJQ",
            three_landlord_cards="D2A",
            sample_count=500,
            random_seed=7,
        )
        engine.observe("landlord", "D")
        result = engine.infer()
        self.assertEqual(
            result["players"]["landlord"]["cards"]["D"]["one_plus"], 0.0
        )

    def test_remaining_count_tracks_observed_plays(self):
        engine = HandInferenceEngine(
            my_position="landlord",
            my_hand_cards="333444555666777888DX",
            three_landlord_cards="DX8",
            sample_count=300,
            random_seed=1,
        )
        engine.observe("landlord_down", "AA")
        engine.observe("landlord_up", "K")
        result = engine.infer()
        self.assertEqual(result["players"]["landlord_down"]["remaining_count"], 15)
        self.assertEqual(result["players"]["landlord_up"]["remaining_count"], 16)

    def test_response_risk_is_bounded_and_rocket_cannot_be_beaten(self):
        engine = HandInferenceEngine(
            my_position="landlord_down",
            my_hand_cards="DAAKKKQQJJT544333",
            three_landlord_cards="A77",
            sample_count=220,
            random_seed=11,
        )
        engine.infer()

        single_risk = engine.response_risk("3", max_samples=100)
        self.assertIsNotNone(single_risk)
        self.assertGreaterEqual(single_risk, 0.0)
        self.assertLessEqual(single_risk, 1.0)

        self.assertEqual(engine.response_risk("DX", max_samples=100), 0.0)
        self.assertIsNone(engine.response_risk("Pass", max_samples=100))


if __name__ == "__main__":
    unittest.main()
