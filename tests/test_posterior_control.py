import unittest

from inference.engine import HandInferenceEngine


class PosteriorControlTests(unittest.TestCase):
    def _engine(self, **kwargs):
        defaults = dict(
            my_position="landlord",
            my_hand_cards="333444555666777888DX",
            sample_count=320,
            random_seed=17,
            min_effective_sample_ratio=0.30,
            residual_behavior_strength=1.0,
            residual_behavior_temperature=0.55,
        )
        defaults.update(kwargs)
        return HandInferenceEngine(**defaults)

    def test_residual_weight_prefers_preserving_pair_when_leading_single(self):
        engine = self._engine()
        hand = list("KK3456789TJQA2")
        split_k = engine._residual_strategy_factor(
            hand,
            action="K",
            rival_action="",
            acting_player="landlord_down",
        )
        shed_single = engine._residual_strategy_factor(
            hand,
            action="3",
            rival_action="",
            acting_player="landlord_down",
        )
        self.assertLess(split_k, shed_single)
        self.assertGreaterEqual(split_k, engine.residual_behavior_floor)

    def test_emergency_enemy_near_out_softens_control_penalty(self):
        engine = self._engine()
        hand = list("KK3456789TJQA2")
        normal = engine._residual_strategy_factor(
            hand,
            action="K",
            rival_action="Q",
            rival_player="landlord",
            acting_player="landlord_down",
            remaining_before={"landlord": 8},
        )
        emergency = engine._residual_strategy_factor(
            hand,
            action="K",
            rival_action="Q",
            rival_player="landlord",
            acting_player="landlord_down",
            remaining_before={"landlord": 2},
        )
        self.assertGreaterEqual(emergency, normal)

    def test_second_stage_keeps_effective_sample_floor(self):
        engine = self._engine()
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
            engine.min_effective_sample_ratio - 0.01,
        )
        self.assertIn("residual_behavior_temperature_used", result)
        self.assertLessEqual(
            result["residual_behavior_temperature_used"],
            result["residual_behavior_temperature_configured"],
        )

    def test_behavior_shift_is_exposed_for_auditing(self):
        engine = self._engine()
        engine.observe("landlord_down", "K")
        result = engine.infer()
        player = result["players"]["landlord_down"]
        self.assertIn("top_behavior_shifts", player)
        self.assertTrue(player["top_behavior_shifts"])
        self.assertIn(
            "prior_one_plus",
            player["cards"]["K"],
        )
        self.assertIn(
            "behavior_delta",
            player["cards"]["K"],
        )

    def test_cache_only_world_export_never_triggers_infer(self):
        engine = self._engine()
        self.assertEqual(
            engine.posterior_worlds(
                max_worlds=8,
                allow_infer=False,
            ),
            [],
        )
        engine.infer()
        worlds = engine.posterior_worlds(
            max_worlds=8,
            allow_infer=False,
        )
        self.assertTrue(worlds)

    def test_posterior_worlds_are_joint_and_normalized(self):
        engine = self._engine()
        engine.observe("landlord_down", "7")
        result = engine.infer()
        worlds = engine.posterior_worlds(max_worlds=12)
        self.assertTrue(worlds)
        self.assertLessEqual(len(worlds), 12)
        self.assertAlmostEqual(
            sum(item["probability"] for item in worlds),
            1.0,
            places=6,
        )
        expected = result["players"]
        for item in worlds:
            for player, hand in item["hands"].items():
                self.assertEqual(
                    len(hand),
                    expected[player]["remaining_count"],
                )


if __name__ == "__main__":
    unittest.main()
