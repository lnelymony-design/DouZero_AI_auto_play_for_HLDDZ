"""Posterior-world rollout for whole-game recommendation evaluation.

P0 design rules:
- The live recognition thread may export a JSON-serializable public snapshot.
- Rollout never receives or mutates the live GameEnv/HandInferenceEngine.
- All candidates are compared on the same set of fully completed worlds.
- Deadline/cancellation is checked between model steps and candidates.
- Non-terminal values remain heuristics; they are not labelled win rates.
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


def snapshot_public_env(public_env, my_position):
    """Detach the public state needed to reconstruct an ExactWorldEnv.

    The returned object contains only JSON-serializable public/own information.
    No InfoSet object, model, tensor or live mutable object is retained.
    """
    return {
        "acting_player_position": public_env.acting_player_position,
        "three_landlord_cards": list(public_env.three_landlord_cards or []),
        "last_move_dict": {
            key: list(value)
            for key, value in public_env.last_move_dict.items()
        },
        "played_cards": {
            key: list(value)
            for key, value in public_env.played_cards.items()
        },
        "last_move": list(public_env.last_move or []),
        "last_two_moves": [
            list(value)
            for value in (public_env.last_two_moves or [])
        ],
        "card_play_action_seq": [
            [position, list(action)]
            for position, action in public_env.card_play_action_seq
        ],
        "bomb_num": int(public_env.bomb_num),
        "last_pid": public_env.last_pid,
        "bid_info": deepcopy(public_env.bid_info),
        "multiply_info": deepcopy(public_env.multiply_info),
        "bid_count": int(public_env.bid_count),
        "multiply_count": {
            key: int(value)
            for key, value in public_env.multiply_count.items()
        },
        "step_count": int(public_env.step_count),
        "my_position": my_position,
        "my_hand": list(
            public_env.info_sets[my_position].player_hand_cards
        ),
    }


class ExactWorldEnv(GameEnv):
    """GameEnv variant that tracks exact hands for all simulated players."""

    def __init__(self):
        super().__init__([None, None])

    @classmethod
    def from_public_snapshot(cls, snapshot, exact_hands):
        env = cls()
        env.acting_player_position = snapshot["acting_player_position"]
        env.three_landlord_cards = list(
            snapshot.get("three_landlord_cards") or []
        )
        env.game_over = False
        env.player_utility_dict = None
        env.last_move_dict = {
            key: list(value)
            for key, value in snapshot["last_move_dict"].items()
        }
        env.played_cards = {
            key: list(value)
            for key, value in snapshot["played_cards"].items()
        }
        env.last_move = list(snapshot.get("last_move") or [])
        env.last_two_moves = [
            list(value)
            for value in snapshot.get("last_two_moves") or []
        ]
        env.card_play_action_seq = [
            (position, list(action))
            for position, action in snapshot["card_play_action_seq"]
        ]
        env.bomb_num = int(snapshot["bomb_num"])
        env.last_pid = snapshot["last_pid"]
        env.bid_info = deepcopy(snapshot["bid_info"])
        env.multiply_info = deepcopy(snapshot["multiply_info"])
        env.bid_count = int(snapshot["bid_count"])
        env.multiply_count = {
            key: int(value)
            for key, value in snapshot["multiply_count"].items()
        }
        env.step_count = int(snapshot["step_count"])

        for position in POSITIONS:
            env.info_sets[position].player_hand_cards = sorted(
                list(exact_hands[position])
            )

        env.game_infoset = env.get_infoset()
        return env

    @classmethod
    def from_public_env(cls, public_env, exact_hands):
        """Backward-compatible helper for existing tests/tools."""
        snapshot = snapshot_public_env(
            public_env,
            public_env.acting_player_position,
        )
        return cls.from_public_snapshot(snapshot, exact_hands)

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
        self.time_budget_seconds = max(0.25, float(time_budget_seconds))

    @staticmethod
    def _candidate_to_env(action):
        if action in (None, "", "Pass", "pass", "PASS"):
            return []
        return sorted(RealCard2EnvCard[c] for c in str(action))

    @staticmethod
    def _should_stop(should_cancel, deadline_monotonic):
        if should_cancel is not None and should_cancel():
            return "cancelled"
        if (
            deadline_monotonic is not None
            and time.monotonic() >= deadline_monotonic
        ):
            return "deadline"
        return None

    def _choose_policy_action(self, env):
        position = env.acting_player_position
        infoset = env.game_infoset
        legal = infoset.legal_actions

        hand_len = len(infoset.player_hand_cards)
        finish = [move for move in legal if len(move) == hand_len]
        if finish:
            return finish[0]

        agent = self.agents.get(position)
        if agent is None:
            return legal[0] if legal else []

        action, _, _ = agent.act(infoset)
        return action

    def _simulate(
        self,
        public_snapshot,
        exact_hands,
        candidate,
        my_position,
        should_cancel=None,
        deadline_monotonic=None,
    ):
        stopped = self._should_stop(
            should_cancel,
            deadline_monotonic,
        )
        if stopped:
            return {"status": stopped}

        env = ExactWorldEnv.from_public_snapshot(
            public_snapshot,
            exact_hands,
        )
        candidate_env = self._candidate_to_env(candidate)
        if not env.apply_exact_action(candidate_env):
            return {"status": "invalid_candidate"}

        trick_last_player = my_position if candidate_env else env.last_pid
        pass_streak = 0
        team_tricks = 0
        enemy_tricks = 0
        steps = 0

        while not env.game_over and steps < self.max_steps:
            stopped = self._should_stop(
                should_cancel,
                deadline_monotonic,
            )
            if stopped:
                return {"status": stopped}

            action = self._choose_policy_action(env)
            position = env.acting_player_position
            if action:
                trick_last_player = position
                pass_streak = 0
            else:
                pass_streak += 1

            if not env.apply_exact_action(action):
                return {"status": "invalid_policy_action"}
            steps += 1

            if pass_streak >= 2 and trick_last_player:
                if _same_team(trick_last_player, my_position):
                    team_tricks += 1
                else:
                    enemy_tricks += 1
                pass_streak = 0

        terminal = bool(env.game_over)
        heuristic = None
        if terminal:
            winner = env.get_winner()
            won = (
                winner == "landlord"
                if my_position == "landlord"
                else winner == "farmer"
            )
            value = 1.0 if won else 0.0
        else:
            my_team_left = 0
            enemy_left = 0
            for position in POSITIONS:
                count = len(env.info_sets[position].player_hand_cards)
                if _same_team(position, my_position):
                    my_team_left += count
                else:
                    enemy_left += count
            total = max(1, my_team_left + enemy_left)
            heuristic = enemy_left / total
            value = heuristic

        total_tricks = team_tricks + enemy_tricks
        control_share = (
            team_tricks / total_tricks if total_tricks else 0.5
        )
        return {
            "status": "ok",
            "value": value,
            "terminal": terminal,
            "heuristic": heuristic,
            "steps": steps,
            "control_share": control_share,
        }

    def evaluate_snapshot(
        self,
        public_snapshot,
        worlds,
        candidates,
        my_position,
        should_cancel=None,
        deadline_monotonic=None,
    ):
        started = time.monotonic()
        if deadline_monotonic is None:
            deadline_monotonic = started + self.time_budget_seconds

        worlds = list(worlds or [])[: self.max_worlds]
        candidates = [str(action) for action in (candidates or [])]
        if not worlds or not candidates:
            return {
                "status": "insufficient_input",
                "worlds_completed": 0,
                "elapsed_seconds": time.monotonic() - started,
                "candidates": [],
                "best_action": None,
            }

        own_hand = list(public_snapshot["my_hand"])
        accum = {
            action: {
                "weighted_value": 0.0,
                "weight": 0.0,
                "terminal_weight": 0.0,
                "terminal_wins": 0.0,
                "weighted_steps": 0.0,
                "weighted_control": 0.0,
                "samples": 0,
            }
            for action in candidates
        }

        complete_worlds = 0
        stop_reason = None

        for world in worlds:
            stop_reason = self._should_stop(
                should_cancel,
                deadline_monotonic,
            )
            if stop_reason:
                break

            world_prob = float(world["probability"])
            exact_hands = {my_position: own_hand}
            for position, cards in world["hands"].items():
                exact_hands[position] = sorted(
                    RealCard2EnvCard[c] for c in cards
                )

            world_results = {}
            world_complete = True
            for action in candidates:
                result = self._simulate(
                    public_snapshot,
                    exact_hands,
                    action,
                    my_position,
                    should_cancel=should_cancel,
                    deadline_monotonic=deadline_monotonic,
                )
                if result.get("status") != "ok":
                    stop_reason = result.get("status")
                    world_complete = False
                    break
                world_results[action] = result

            # Candidate comparison must use common completed worlds only.
            if not world_complete:
                break

            complete_worlds += 1
            for action, result in world_results.items():
                bucket = accum[action]
                bucket["weighted_value"] += world_prob * result["value"]
                bucket["weight"] += world_prob
                if result["terminal"]:
                    bucket["terminal_weight"] += world_prob
                    bucket["terminal_wins"] += (
                        world_prob * result["value"]
                    )
                bucket["weighted_steps"] += world_prob * result["steps"]
                bucket["weighted_control"] += (
                    world_prob * result["control_share"]
                )
                bucket["samples"] += 1

        results = []
        for action in candidates:
            bucket = accum[action]
            weight = bucket["weight"]
            if weight <= 0:
                continue

            terminal_weight = bucket["terminal_weight"]
            terminal_ratio = terminal_weight / weight
            terminal_win_rate = (
                bucket["terminal_wins"] / terminal_weight
                if terminal_weight > 0
                else None
            )
            results.append({
                "action": action,
                "rollout_value": bucket["weighted_value"] / weight,
                "terminal_ratio": terminal_ratio,
                "terminal_win_rate": terminal_win_rate,
                "heuristic_weight": 1.0 - terminal_ratio,
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

        if stop_reason == "cancelled":
            status = "cancelled"
        elif complete_worlds >= self.min_worlds:
            status = "ok" if stop_reason is None else "partial"
        elif complete_worlds > 0:
            status = "partial"
        else:
            status = (
                "deadline"
                if stop_reason == "deadline"
                else "insufficient_worlds"
            )

        return {
            "status": status,
            "stop_reason": stop_reason,
            "worlds_completed": complete_worlds,
            "worlds_requested": min(len(worlds), self.max_worlds),
            "elapsed_seconds": time.monotonic() - started,
            "candidates": results,
            "best_action": results[0]["action"] if results else None,
        }

    def evaluate(self, public_env, hand_inference, candidates, my_position):
        """Backward-compatible synchronous entry for tests/tools."""
        worlds = hand_inference.posterior_worlds(self.max_worlds)
        snapshot = snapshot_public_env(public_env, my_position)
        return self.evaluate_snapshot(
            snapshot,
            worlds,
            candidates,
            my_position,
        )
