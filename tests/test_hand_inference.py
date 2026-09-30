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

    def test_repeated_inference_is_stable_for_same_public_state(self):
        engine = HandInferenceEngine(
            my_position="landlord_up",
            my_hand_cards="33445566778899TJQ",
            three_landlord_cards="D2A",
            sample_count=240,
        )
        first = engine.infer()
        second = engine.infer()

        self.assertEqual(first["players"], second["players"])
        self.assertEqual(first["effective_samples"], second["effective_samples"])

    def test_play_behavior_penalizes_splitting_a_pair(self):
        engine = HandInferenceEngine(
            my_position="landlord",
            my_hand_cards="333444555666777888DX",
            sample_count=220,
            random_seed=3,
        )
        clean = engine._play_behavior_factor(
            reconstructed_hand=list("K3456789TJQA2"),
            action="K",
            rival_action="Q",
        )
        split_pair = engine._play_behavior_factor(
            reconstructed_hand=list("KK3456789TJQA2"),
            action="K",
            rival_action="Q",
        )
        self.assertLess(split_pair, clean)

    def test_farmer_teammate_pass_is_weaker_evidence_than_enemy_pass(self):
        engine = HandInferenceEngine(
            my_position="landlord",
            my_hand_cards="333444555666777888DX",
            sample_count=220,
            random_seed=4,
        )
        friendly = engine._pass_penalty_for("landlord_up", "landlord_down")
        enemy = engine._pass_penalty_for("landlord_up", "landlord")
        self.assertGreater(friendly, enemy)

    def test_actual_play_evidence_is_reported(self):
        engine = HandInferenceEngine(
            my_position="landlord",
            my_hand_cards="333444555666777888DX",
            sample_count=240,
            random_seed=5,
        )
        engine.observe("landlord_down", "A")
        result = engine.infer()
        self.assertEqual(result["play_evidence_count"], 1)
        self.assertIn("effective_sample_ratio", result)
        self.assertGreaterEqual(result["effective_sample_ratio"], 0.0)
        self.assertLessEqual(result["effective_sample_ratio"], 1.0)

    def test_actual_single_play_reduces_posterior_for_split_pair(self):
        common = dict(
            my_position="landlord",
            my_hand_cards="333444555666777888DX",
            sample_count=1200,
            random_seed=23,
            behavior_temperature=0.75,
        )

        neutral = HandInferenceEngine(
            **common,
            play_behavior_strength=0.0,
        )
        neutral.observe("landlord_down", "K")
        neutral_result = neutral.infer()
        neutral_k = neutral_result["players"]["landlord_down"]["cards"]["K"]["one_plus"]

        behavioral = HandInferenceEngine(
            **common,
            play_behavior_strength=1.0,
        )
        behavioral.observe("landlord_down", "K")
        behavioral_result = behavioral.infer()
        behavioral_k = behavioral_result["players"]["landlord_down"]["cards"]["K"]["one_plus"]

        # The observed K is already public and removed from the current hand.
        # If the sampled hidden hand still contains another K, that means the
        # player split KK to lead a single K.  The behavior model should make
        # those particles less likely, not more likely.
        self.assertLess(behavioral_k, neutral_k)
        self.assertGreaterEqual(
            behavioral_result["effective_sample_ratio"],
            behavioral.min_effective_sample_ratio - 0.01,
        )

    def test_combo_risk_summary_matches_rank_statistics(self):
        engine = HandInferenceEngine(
            my_position="landlord_up",
            my_hand_cards="33445566778899TJQ",
            three_landlord_cards="D2A",
            sample_count=320,
            random_seed=31,
        )
        result = engine.infer()
        landlord = result["players"]["landlord"]
        risks = landlord["combo_risks"]

        self.assertEqual(risks["has_2"], landlord["cards"]["2"]["one_plus"])
        self.assertEqual(risks["pair_2"], landlord["cards"]["2"]["pair_plus"])
        self.assertEqual(risks["triple_A"], landlord["cards"]["A"]["triple_plus"])
        self.assertEqual(risks["any_bomb"], landlord["any_bomb"])
        self.assertEqual(risks["rocket"], landlord["rocket"])

    def test_adaptive_tempering_respects_ess_floor(self):
        engine = HandInferenceEngine(
            my_position="landlord",
            my_hand_cards="333444555666777888DX",
            sample_count=320,
            random_seed=9,
            min_effective_sample_ratio=0.30,
        )
        # A sequence of observed high-card responses creates enough behavior
        # evidence to exercise the tempering path without changing hard card
        # constraints.
        engine.observe("landlord_down", "7")
        engine.observe("landlord_up", "")
        engine.observe("landlord", "")
        engine.observe("landlord_down", "9")
        engine.observe("landlord_up", "")
        engine.observe("landlord", "")
        engine.observe("landlord_down", "J")
        result = engine.infer()

        self.assertGreaterEqual(
            result["effective_sample_ratio"],
            0.30 - 0.01,
        )
        self.assertLessEqual(
            result["behavior_temperature_used"],
            result["behavior_temperature_configured"],
        )


if __name__ == "__main__":
    unittest.main()
