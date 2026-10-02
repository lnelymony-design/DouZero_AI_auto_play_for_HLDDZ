import unittest

from douzero.env.game_new import GameEnv
from inference.rollout import (
    PosteriorRolloutEvaluator,
    snapshot_public_env,
)


class FirstLegalAgent:
    def act(self, infoset):
        action = infoset.legal_actions[0]
        return action, 0.0, [(action, 0.0)]


class FakeInference:
    def posterior_worlds(self, max_worlds=8):
        return [
            {
                "hands": {
                    "landlord_down": "5",
                    "landlord_up": "6",
                },
                "probability": 1.0,
            }
        ]


class RolloutTests(unittest.TestCase):
    def test_cancelled_rollout_returns_structured_status(self):
        public = GameEnv(["landlord", None])
        public.card_play_init(
            {
                "landlord": [3, 4],
                "landlord_down": [5],
                "landlord_up": [6],
                "three_landlord_cards": [],
            }
        )
        evaluator = PosteriorRolloutEvaluator(
            agents={
                "landlord": FirstLegalAgent(),
                "landlord_down": FirstLegalAgent(),
                "landlord_up": FirstLegalAgent(),
            },
            max_worlds=1,
            min_worlds=1,
            max_steps=12,
            time_budget_seconds=1.0,
        )
        snapshot = snapshot_public_env(public, "landlord")
        result = evaluator.evaluate_snapshot(
            public_snapshot=snapshot,
            worlds=FakeInference().posterior_worlds(),
            candidates=["3", "4"],
            my_position="landlord",
            should_cancel=lambda: True,
        )
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["worlds_completed"], 0)
        self.assertEqual(result["candidates"], [])

    def test_worlds_below_minimum_are_explicitly_insufficient(self):
        public = GameEnv(["landlord", None])
        public.card_play_init(
            {
                "landlord": [3, 4],
                "landlord_down": [5],
                "landlord_up": [6],
                "three_landlord_cards": [],
            }
        )
        evaluator = PosteriorRolloutEvaluator(
            agents={
                "landlord": FirstLegalAgent(),
                "landlord_down": FirstLegalAgent(),
                "landlord_up": FirstLegalAgent(),
            },
            max_worlds=2,
            min_worlds=2,
            max_steps=12,
            time_budget_seconds=1.0,
        )
        snapshot = snapshot_public_env(public, "landlord")
        one_world = FakeInference().posterior_worlds()
        result = evaluator.evaluate_snapshot(
            public_snapshot=snapshot,
            worlds=one_world,
            candidates=["3", "4"],
            my_position="landlord",
        )
        self.assertEqual(result["status"], "insufficient_worlds")
        self.assertEqual(result["worlds_completed"], 1)
        self.assertIsNone(result["best_action"])

    def test_rollout_returns_whole_game_values(self):
        public = GameEnv(["landlord", None])
        public.card_play_init(
            {
                "landlord": [3, 4],
                "landlord_down": [5],
                "landlord_up": [6],
                "three_landlord_cards": [],
            }
        )
        evaluator = PosteriorRolloutEvaluator(
            agents={
                "landlord": FirstLegalAgent(),
                "landlord_down": FirstLegalAgent(),
                "landlord_up": FirstLegalAgent(),
            },
            max_worlds=1,
            min_worlds=1,
            max_steps=12,
            time_budget_seconds=1.0,
        )
        result = evaluator.evaluate(
            public_env=public,
            hand_inference=FakeInference(),
            candidates=["3", "4"],
            my_position="landlord",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["worlds_completed"], 1)
        self.assertEqual(len(result["candidates"]), 2)
        for item in result["candidates"]:
            self.assertGreaterEqual(item["rollout_value"], 0.0)
            self.assertLessEqual(item["rollout_value"], 1.0)
            self.assertGreaterEqual(item["control_share"], 0.0)
            self.assertLessEqual(item["control_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
