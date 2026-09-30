"""Posterior-world rollout for whole-game recommendation evaluation.

This module evaluates candidate actions by continuing the game inside complete
hidden-card worlds sampled from HandInferenceEngine's joint posterior.  Each
simulated player only sees its own hand plus public information through the
existing DouZero infoset, so exact hidden cards are used to define reality, not
leaked into the policy observation.
"""

from copy import deepcopy
import time

from constants import Bombs, RealCard2EnvCard
from douzero.env.game_new import GameEnv

POSITIONS = ("landlord", "landlord_up", "landlord_down")


def _same_team(a, b):
    if a == "landlord" or b == "landlord":
        return a == b == "landlord"
    return True


class ExactWorldEnv(GameEnv):
    """GameEnv variant that tracks exact hands for all three simulated players."""

    def __init__(self):
        super().__init__([None, None])

    @classmethod
    def from_public_env(cls, public_env, exact_hands):
        env = cls()
        env.acting_player_position = public_env.acting_player_position
        env.three_landlord_cards = list(public_env.three_landlord_cards or [])
        env.game_over = False
        env.player_utility_dict = None
        env.last_move_dict = deepcopy(public_env.last_move_dict)
        env.played_cards = deepcopy(public_env.played_cards)
        env.last_move = deepcopy(public_env.last_move)
        env.last_two_moves = deepcopy(public_env.last_two_moves)
        env.card_play_action_seq = deepcopy(public_env.card_play_action_seq)
        env.bomb_num = public_env.bomb_num
        env.last_pid = public_env.last_pid
        env.bid_info = deepcopy(public_env.bid_info)
        env.multiply_info = deepcopy(public_env.multiply_info)
        env.bid_count = public_env.bid_count
        env.multiply_count = deepcopy(public_env.multiply_count)
        env.step_count = public_env.step_count

        for position in POSITIONS:
            env.info_sets[position].player_hand_cards = sorted(
                list(exact_hands[position])
            )

        env.game_infoset = env.get_infoset()
        return env

    def apply_exact_action(self, action):
        """Apply one legal action without invoking a policy model."""
        action = sorted(list(action))
        position = self.acting_player_position
        legal = self.game_infoset.legal_actions
        if action not in legal:
            return False

        if action:
            self.last_pid = position
        if action in Bombs:
            self.bomb_num += 1

        self.last_move_dict[position] = action.copy()
        self.card_play_action_seq.append((position, action.copy()))

        hand = self.info_sets[position].player_hand_cards
        for card in action:
            if card not in hand:
                return False
            hand.remove(card)
        hand.sort()

        self.played_cards[position] += action

        if (
            position == "landlord"
            and action
            and self.three_landlord_cards
        ):
            for card in action:
                if card in self.three_landlord_cards:
                    self.three_landlord_cards.remove(card)

        self.check_if_game_overed()
        if not self.game_over:
            self.get_acting_player_position()
            self.game_infoset = self.get_infoset()
        return True


class PosteriorRolloutEvaluator:
    def __init__(
        self,
        agents,
        max_worlds=8,
        min_worlds=3,
        max_steps=72,
        time_budget_seconds=8.0,
    ):
        self.agents = dict(agents)
        self.max_worlds = max(1, int(max_worlds))
        self.min_worlds = max(1, min(int(min_worlds), self.max_worlds))
        self.max_steps = max(6, int(max_steps))
        self.time_budget_seconds = max(1.0, float(time_budget_seconds))

    @staticmethod
    def _candidate_to_env(action):
        if action in (None, "", "Pass", "pass", "PASS"):
            return []
        return sorted(RealCard2EnvCard[c] for c in str(action))

    def _choose_policy_action(self, env):
        position = env.acting_player_position
        infoset = env.game_infoset
        legal = infoset.legal_actions

        # Finishing immediately dominates any non-finishing policy choice.
        hand_len = len(infoset.player_hand_cards)
        finish = [move for move in legal if len(move) == hand_len]
        if finish:
            return finish[0]

        agent = self.agents.get(position)
        if agent is None:
            return legal[0] if legal else []

        action, _, _ = agent.act(infoset)
        return action

    def _simulate(self, public_env, exact_hands, candidate, my_position):
        env = ExactWorldEnv.from_public_env(public_env, exact_hands)
        candidate_env = self._candidate_to_env(candidate)
        if not env.apply_exact_action(candidate_env):
            return None

        trick_last_player = my_position if candidate_env else env.last_pid
        pass_streak = 0
        team_tricks = 0
        enemy_tricks = 0
        steps = 0

        while not env.game_over and steps < self.max_steps:
            position = env.acting_player_position
            action = self._choose_policy_action(env)
            if action:
                trick_last_player = position
                pass_streak = 0
            else:
                pass_streak += 1

            if not env.apply_exact_action(action):
                return None
            steps += 1

            if pass_streak >= 2 and trick_last_player:
                if _same_team(trick_last_player, my_position):
                    team_tricks += 1
                else:
                    enemy_tricks += 1
                pass_streak = 0

        terminal = bool(env.game_over)
        if terminal:
            winner = env.get_winner()
            won = (
                winner == "landlord"
                if my_position == "landlord"
                else winner == "farmer"
            )
            win_value = 1.0 if won else 0.0
        else:
            # Rare horizon fallback.  This is deliberately not labelled as a
            # win probability; it only keeps an unfinished rollout sortable.
            my_team_left = 0
            enemy_left = 0
            for position in POSITIONS:
                count = len(env.info_sets[position].player_hand_cards)
                if _same_team(position, my_position):
                    my_team_left += count
                else:
                    enemy_left += count
            total = max(1, my_team_left + enemy_left)
            win_value = enemy_left / total

        total_tricks = team_tricks + enemy_tricks
        control_share = (
            team_tricks / total_tricks if total_tricks else 0.5
        )
        return {
            "value": win_value,
            "terminal": terminal,
            "steps": steps,
            "control_share": control_share,
        }

    def evaluate(self, public_env, hand_inference, candidates, my_position):
        worlds = hand_inference.posterior_worlds(self.max_worlds)
        if not worlds or not candidates:
            return None

        candidates = [str(action) for action in candidates]
        own_hand = list(
            public_env.info_sets[my_position].player_hand_cards
        )
        started = time.monotonic()
        accum = {
            action: {
                "weighted_value": 0.0,
                "weight": 0.0,
                "terminal_weight": 0.0,
                "weighted_steps": 0.0,
                "weighted_control": 0.0,
                "samples": 0,
            }
            for action in candidates
        }

        complete_worlds = 0
        for world in worlds:
            world_prob = float(world["probability"])
            exact_hands = {my_position: own_hand}
            for position, cards in world["hands"].items():
                exact_hands[position] = sorted(
                    RealCard2EnvCard[c] for c in cards
                )

            world_complete = True
            world_results = {}
            for action in candidates:
                result = self._simulate(
                    public_env,
                    exact_hands,
                    action,
                    my_position,
                )
                if result is None:
                    world_complete = False
                    break
                world_results[action] = result

            if not world_complete:
                continue

            complete_worlds += 1
            for action, result in world_results.items():
                bucket = accum[action]
                bucket["weighted_value"] += world_prob * result["value"]
                bucket["weight"] += world_prob
                bucket["terminal_weight"] += (
                    world_prob if result["terminal"] else 0.0
                )
                bucket["weighted_steps"] += world_prob * result["steps"]
                bucket["weighted_control"] += (
                    world_prob * result["control_share"]
                )
                bucket["samples"] += 1

            if (
                complete_worlds >= self.min_worlds
                and time.monotonic() - started >= self.time_budget_seconds
            ):
                break

        if complete_worlds < self.min_worlds:
            return None

        results = []
        for action in candidates:
            bucket = accum[action]
            weight = bucket["weight"]
            if weight <= 0:
                continue
            results.append({
                "action": action,
                "rollout_value": bucket["weighted_value"] / weight,
                "terminal_ratio": bucket["terminal_weight"] / weight,
                "expected_steps": bucket["weighted_steps"] / weight,
                "control_share": bucket["weighted_control"] / weight,
                "samples": bucket["samples"],
            })

        results.sort(
            key=lambda item: (
                item["rollout_value"],
                item["control_share"],
                -item["expected_steps"],
            ),
            reverse=True,
        )
        return {
            "worlds_completed": complete_worlds,
            "elapsed_seconds": time.monotonic() - started,
            "candidates": results,
            "best_action": results[0]["action"] if results else None,
        }
