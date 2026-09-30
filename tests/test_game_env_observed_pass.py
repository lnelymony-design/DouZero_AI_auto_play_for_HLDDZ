import unittest

from douzero.env.game_new import GameEnv


class FakeAgent:
    def __init__(self):
        self.calls = 0

    def act(self, infoset):
        self.calls += 1
        return [3], 0.5, [[[3], 0.5]]


class GameEnvObservedPassTests(unittest.TestCase):
    def _env(self):
        agent = FakeAgent()
        env = GameEnv(["landlord", agent])
        env.card_play_init({
            "landlord": [3, 4],
            "landlord_down": [5, 6],
            "landlord_up": [7, 8],
            "three_landlord_cards": [3, 4, 5],
        })
        return env, agent

    def test_empty_list_is_explicit_observed_pass(self):
        env, agent = self._env()

        self.assertEqual(env.acting_player_position, "landlord")
        env.step("landlord", action=[], update=True)

        self.assertEqual(agent.calls, 0)
        self.assertEqual(env.card_play_action_seq[-1], ("landlord", []))
        self.assertEqual(env.acting_player_position, "landlord_down")

    def test_none_action_requests_model_without_updating(self):
        env, agent = self._env()

        message, actions = env.step("landlord", action=None, update=False)

        self.assertEqual(agent.calls, 1)
        self.assertEqual(env.acting_player_position, "landlord")
        self.assertEqual(env.card_play_action_seq, [])
        self.assertTrue(message["action"])


if __name__ == "__main__":
    unittest.main()
