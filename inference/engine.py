"""Monte-Carlo opponent hand inference.

This module never claims to know an opponent's exact cards.  It samples all
currently legal hidden-card allocations and treats a Pass as *evidence* rather
than proof.  If a sampled hand could have beaten the current move but passed,
that sample receives a configurable penalty.
"""
from collections import Counter, defaultdict
import math
import random

from constants import AllEnvCard, EnvCard2RealCard, RealCard2EnvCard
from douzero.env import move_detector as md
from .legality import can_beat

POSITIONS = ("landlord", "landlord_up", "landlord_down")
INITIAL_HAND_SIZES = {
    "landlord": 20,
    "landlord_up": 17,
    "landlord_down": 17,
}
CARD_ORDER = ("D", "X", "2", "A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3")
DECK_REAL = [EnvCard2RealCard[c] for c in AllEnvCard]


def _remove_counter(source, to_remove):
    result = Counter(source)
    result.subtract(Counter(to_remove))
    if any(v < 0 for v in result.values()):
        return None
    return result


def _expand(counter):
    cards = []
    for card, count in counter.items():
        cards.extend([card] * count)
    return cards


def _sorted_cards(cards):
    rank = {c: i for i, c in enumerate(CARD_ORDER)}
    return "".join(sorted(cards, key=lambda c: rank[c]))


class HandInferenceEngine:
    """Estimate the two opponents' remaining cards from public play history."""

    def __init__(
        self,
        my_position,
        my_hand_cards,
        three_landlord_cards="",
        sample_count=1600,
        pass_penalty=0.62,
        friendly_pass_penalty=0.86,
        play_behavior_floor=0.18,
        play_behavior_strength=1.0,
        random_seed=None,
    ):
        if my_position not in POSITIONS:
            raise ValueError(f"Unknown position: {my_position}")
        self.my_position = my_position
        self.initial_my_hand = list(my_hand_cards)
        self.three_landlord_cards = list(three_landlord_cards or "")
        self.sample_count = max(200, int(sample_count))
        self.pass_penalty = min(1.0, max(0.05, float(pass_penalty)))
        self.friendly_pass_penalty = min(
            1.0, max(self.pass_penalty, float(friendly_pass_penalty))
        )
        self.play_behavior_floor = min(
            0.95, max(0.05, float(play_behavior_floor))
        )
        self.play_behavior_strength = min(
            3.0, max(0.0, float(play_behavior_strength))
        )
        # Common random numbers make adjacent public states comparable and keep
        # the UI from flickering purely because a fresh Monte-Carlo stream was
        # drawn.  A caller-provided seed still overrides the default.
        self.random_seed = 20260930 if random_seed is None else random_seed
        self.rng = random.Random(self.random_seed)
        self.history = []
        self._latest_weighted_samples = []
        self._latest_total_weight = 0.0

        self.opponents = [p for p in POSITIONS if p != self.my_position]

    def reset(self):
        self.history = []
        self._latest_weighted_samples = []
        self._latest_total_weight = 0.0

    def observe(self, player, action):
        """Append one public action.

        action may be a card string (e.g. 'AA', '34567'), '', None, or 'Pass'.
        """
        if player not in POSITIONS:
            raise ValueError(f"Unknown position: {player}")
        if action in (None, "", "Pass", "pass", "PASS"):
            cards = ""
        else:
            cards = str(action)
            unknown = [c for c in cards if c not in RealCard2EnvCard]
            if unknown:
                raise ValueError(f"Unknown card symbols: {unknown}")
        self.history.append((player, cards))

    def _played_by_player(self):
        out = {p: [] for p in POSITIONS}
        for player, action in self.history:
            if action:
                out[player].extend(list(action))
        return out

    def _current_my_hand(self, played):
        remaining = _remove_counter(self.initial_my_hand, played[self.my_position])
        if remaining is None:
            raise ValueError("Observed own plays are inconsistent with initial hand.")
        return _expand(remaining)

    def _current_hidden_pool(self, played, current_my_hand):
        visible = list(current_my_hand)
        for p in POSITIONS:
            visible.extend(played[p])
        hidden = _remove_counter(DECK_REAL, visible)
        if hidden is None:
            raise ValueError("Public history is inconsistent with a 54-card deck.")
        return hidden

    def _forced_landlord_cards(self, played):
        # When we are a farmer, the three revealed bottom cards were definitely
        # in the landlord's initial hand.  Copies already played by landlord no
        # longer need to be forced into the remaining hand.
        if self.my_position == "landlord":
            return Counter()
        forced = Counter(self.three_landlord_cards)
        forced.subtract(Counter(played["landlord"]))
        return Counter({k: max(0, v) for k, v in forced.items() if v > 0})

    def _remaining_counts(self, played):
        return {
            p: INITIAL_HAND_SIZES[p] - len(played[p])
            for p in self.opponents
        }

    @staticmethod
    def _same_team(player_a, player_b):
        if not player_a or not player_b:
            return False
        return player_a != "landlord" and player_b != "landlord"

    def _pass_penalty_for(self, passer, rival_player):
        # Farmer-vs-farmer passes are much weaker evidence: letting a teammate's
        # winning card stand is normal cooperative play.  A pass against the
        # opposing camp is more informative.
        if self._same_team(passer, rival_player):
            return self.friendly_pass_penalty
        return self.pass_penalty

    def _build_behavior_contexts(self):
        """Build Pass and actual-play evidence from the public action history.

        Every context stores the cards that the same player plays *later*, so a
        sampled current hand can be rewound to the exact moment of the action.
        """
        pass_contexts = []
        play_contexts = []
        last_move = ""
        last_player = None
        pass_streak = 0

        for idx, (player, action) in enumerate(self.history):
            future_cards = []
            if player in self.opponents:
                for later_player, later_action in self.history[idx + 1:]:
                    if later_player == player and later_action:
                        future_cards.extend(list(later_action))

            if not action:
                if player in self.opponents and last_move:
                    pass_contexts.append(
                        (player, last_player, last_move, future_cards)
                    )
                pass_streak += 1
                if pass_streak >= 2:
                    last_move = ""
                    last_player = None
                    pass_streak = 0
            else:
                if player in self.opponents:
                    play_contexts.append(
                        (player, action, last_player, last_move, future_cards)
                    )
                last_move = action
                last_player = player
                pass_streak = 0

        return pass_contexts, play_contexts

    @staticmethod
    def _rank_response_width(move_type):
        if move_type == md.TYPE_1_SINGLE:
            return 1
        if move_type == md.TYPE_2_PAIR:
            return 2
        if move_type == md.TYPE_3_TRIPLE:
            return 3
        return None

    def _play_behavior_factor(self, reconstructed_hand, action, rival_action):
        """Heuristic likelihood factor for an observed non-Pass play.

        This is deliberately conservative and cheap enough to evaluate for every
        Monte-Carlo particle.  It does *not* pretend to model optimal play.
        Instead it uses two robust behavioral clues:

        1. breaking a pair/triple/bomb to play fewer copies is less natural;
        2. when responding with a single/pair/triple, using a much higher rank
           despite holding lower legal responses is less natural.

        Bombs/rocket are also mildly down-weighted when another response was
        available.  The returned factor is clamped, so one behavioral guess can
        never eliminate an otherwise legal hidden hand.
        """
        if not action:
            return 1.0

        hand_counter = Counter(reconstructed_hand)
        action_counter = Counter(action)
        raw = 1.0

        # Structure-splitting evidence.
        for card, used in action_counter.items():
            owned = hand_counter[card]
            if 0 < used < owned:
                extra = owned - used
                if used == 1:
                    raw *= {1: 0.80, 2: 0.64, 3: 0.45}.get(extra, 0.45)
                elif used == 2:
                    raw *= {1: 0.74, 2: 0.52}.get(extra, 0.52)
                elif used == 3:
                    raw *= 0.58

        action_env = sorted(RealCard2EnvCard[c] for c in action)
        action_type = md.get_move_type(action_env)
        action_kind = action_type.get("type")

        # If this was a same-type response, lower legal ranks that were also
        # available make the chosen high response somewhat less likely.
        if rival_action:
            rival_env = sorted(RealCard2EnvCard[c] for c in rival_action)
            rival_type = md.get_move_type(rival_env)
            width = self._rank_response_width(action_kind)
            if (
                width is not None
                and rival_type.get("type") == action_kind
                and "rank" in action_type
                and "rank" in rival_type
            ):
                rival_rank = rival_type["rank"]
                actual_rank = action_type["rank"]
                env_counts = Counter(
                    RealCard2EnvCard[c] for c in reconstructed_hand
                )
                lower_options = 0
                for rank, count in env_counts.items():
                    if rival_rank < rank < actual_rank and count >= width:
                        lower_options += 1
                if lower_options:
                    raw *= max(0.55, 0.88 ** lower_options)

            # Bomb/rocket conservation: if removing the observed bomb still
            # leaves any legal response, using the bomb was less forced.
            if action_kind in (md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB):
                remaining = Counter(
                    RealCard2EnvCard[c] for c in reconstructed_hand
                )
                remaining.subtract(Counter(action_env))
                rest = [
                    card
                    for card, count in remaining.items()
                    for _ in range(max(0, count))
                ]
                if rest and can_beat(rest, rival_env):
                    raw *= 0.58

        raw = max(self.play_behavior_floor, min(1.0, raw))
        if self.play_behavior_strength <= 0:
            return 1.0
        return max(
            self.play_behavior_floor,
            raw ** self.play_behavior_strength,
        )

    def _sample_current_hands(self, hidden_pool, remaining_counts, forced_landlord):
        pool = Counter(hidden_pool)
        hands = {p: [] for p in self.opponents}

        if "landlord" in self.opponents:
            for card, count in forced_landlord.items():
                if pool[card] < count:
                    return None
                hands["landlord"].extend([card] * count)
                pool[card] -= count

        slots = {
            p: remaining_counts[p] - len(hands[p])
            for p in self.opponents
        }
        if any(v < 0 for v in slots.values()):
            return None

        loose_cards = _expand(Counter({k: v for k, v in pool.items() if v > 0}))
        if sum(slots.values()) != len(loose_cards):
            return None

        self.rng.shuffle(loose_cards)
        cursor = 0
        for p in self.opponents:
            take = slots[p]
            hands[p].extend(loose_cards[cursor:cursor + take])
            cursor += take

        return hands

    def infer(self):
        # Rewind the Monte-Carlo stream for every public state.  The same state
        # is therefore deterministic, while new observations change the legal
        # pool/weights rather than introducing unrelated sampling noise.
        self.rng.seed(self.random_seed)
        played = self._played_by_player()
        current_my_hand = self._current_my_hand(played)
        hidden_pool = self._current_hidden_pool(played, current_my_hand)
        remaining_counts = self._remaining_counts(played)
        forced_landlord = self._forced_landlord_cards(played)
        pass_contexts, play_contexts = self._build_behavior_contexts()

        weighted_samples = []
        for _ in range(self.sample_count):
            hands = self._sample_current_hands(hidden_pool, remaining_counts, forced_landlord)
            if hands is None:
                continue

            weight = 1.0
            for player, rival_player, rival_action, future_cards in pass_contexts:
                reconstructed = list(hands[player]) + list(future_cards)
                hand_env = [RealCard2EnvCard[c] for c in reconstructed]
                rival_env = [RealCard2EnvCard[c] for c in rival_action]
                if can_beat(hand_env, rival_env):
                    weight *= self._pass_penalty_for(player, rival_player)

            for (
                player,
                action,
                rival_player,
                rival_action,
                future_cards,
            ) in play_contexts:
                reconstructed = (
                    list(hands[player])
                    + list(action)
                    + list(future_cards)
                )
                weight *= self._play_behavior_factor(
                    reconstructed,
                    action,
                    rival_action,
                )

            if weight > 0:
                weighted_samples.append((hands, weight))

        total_weight = sum(w for _, w in weighted_samples)
        if total_weight <= 0:
            self._latest_weighted_samples = []
            self._latest_total_weight = 0.0
            return {
                "players": {},
                "samples": 0,
                "effective_samples": 0.0,
                "warning": "No legal hidden-card samples remain.",
            }

        sum_sq = sum((w / total_weight) ** 2 for _, w in weighted_samples)
        effective_samples = 1.0 / sum_sq if sum_sq else 0.0

        self._latest_weighted_samples = weighted_samples
        self._latest_total_weight = total_weight

        result = {}
        for player in self.opponents:
            rank_stats = {
                c: {"one_plus": 0.0, "pair_plus": 0.0, "triple_plus": 0.0, "bomb": 0.0}
                for c in CARD_ORDER
            }
            any_bomb = 0.0
            rocket = 0.0
            hand_mass = defaultdict(float)

            for hands, weight in weighted_samples:
                p = weight / total_weight
                counts = Counter(hands[player])
                for card in CARD_ORDER:
                    n = counts[card]
                    if n >= 1:
                        rank_stats[card]["one_plus"] += p
                    if n >= 2:
                        rank_stats[card]["pair_plus"] += p
                    if n >= 3:
                        rank_stats[card]["triple_plus"] += p
                    if n >= 4:
                        rank_stats[card]["bomb"] += p

                has_rocket = counts["D"] >= 1 and counts["X"] >= 1
                has_bomb = has_rocket or any(counts[c] >= 4 for c in CARD_ORDER)
                if has_rocket:
                    rocket += p
                if has_bomb:
                    any_bomb += p

                hand_mass[_sorted_cards(hands[player])] += p

            top_hands = sorted(hand_mass.items(), key=lambda x: x[1], reverse=True)[:5]
            result[player] = {
                "remaining_count": remaining_counts[player],
                "cards": {
                    card: {k: round(v, 4) for k, v in stats.items()}
                    for card, stats in rank_stats.items()
                },
                "any_bomb": round(any_bomb, 4),
                "rocket": round(rocket, 4),
                "top_sampled_hands": [
                    {"hand": hand, "probability": round(prob, 4)}
                    for hand, prob in top_hands
                ],
            }

        return {
            "players": result,
            "samples": len(weighted_samples),
            "effective_samples": round(effective_samples, 1),
            "pass_evidence_count": len(pass_contexts),
            "play_evidence_count": len(play_contexts),
            "enemy_pass_evidence_count": sum(
                1
                for player, rival_player, _, _ in pass_contexts
                if not self._same_team(player, rival_player)
            ),
            "friendly_pass_evidence_count": sum(
                1
                for player, rival_player, _, _ in pass_contexts
                if self._same_team(player, rival_player)
            ),
            "pass_penalty": self.pass_penalty,
            "friendly_pass_penalty": self.friendly_pass_penalty,
            "play_behavior_floor": self.play_behavior_floor,
            "play_behavior_strength": self.play_behavior_strength,
            "effective_sample_ratio": round(
                effective_samples / len(weighted_samples), 3
            ) if weighted_samples else 0.0,
            "behavior_model": "heuristic_v2",
        }

    def response_risk(self, action, max_samples=320):
        """Estimate whether an enemy *can* legally beat a proposed action.

        This is a card-power risk, not a calibrated probability that the enemy
        will actually choose to respond.  For a farmer, only the landlord is an
        enemy; for the landlord, both farmers are enemies.
        """
        if action in (None, "", "Pass", "pass", "PASS"):
            return None

        cards = str(action)
        unknown = [c for c in cards if c not in RealCard2EnvCard]
        if unknown:
            raise ValueError(f"Unknown card symbols: {unknown}")

        if not self._latest_weighted_samples or self._latest_total_weight <= 0:
            self.infer()

        samples = self._latest_weighted_samples
        if not samples:
            return None

        max_samples = max(50, int(max_samples))
        if len(samples) > max_samples:
            step = len(samples) / max_samples
            selected = [samples[min(len(samples) - 1, int(i * step))]
                        for i in range(max_samples)]
        else:
            selected = samples

        enemies = (
            [p for p in self.opponents]
            if self.my_position == "landlord"
            else ["landlord"]
        )
        rival_env = [RealCard2EnvCard[c] for c in cards]

        risk_weight = 0.0
        total_weight = 0.0
        for hands, weight in selected:
            total_weight += weight
            can_enemy_beat = False
            for enemy in enemies:
                hand = hands.get(enemy)
                if hand is None:
                    continue
                hand_env = [RealCard2EnvCard[c] for c in hand]
                if can_beat(hand_env, rival_env):
                    can_enemy_beat = True
                    break
            if can_enemy_beat:
                risk_weight += weight

        if total_weight <= 0:
            return None
        return risk_weight / total_weight

    def response_risks(self, actions, max_samples=320):
        return {
            action: self.response_risk(action, max_samples=max_samples)
            for action in actions
        }

    @staticmethod
    def compact_summary(result, important_cards=("D", "X", "2", "A", "K")):
        parts = []
        display_map = {"D": "大王", "X": "小王", "T": "10"}
        for player, data in result.get("players", {}).items():
            card_parts = []
            for card in important_cards:
                prob = data["cards"][card]["one_plus"]
                card_parts.append(f"{display_map.get(card, card)}:{prob:.0%}")
            parts.append(
                f"{player}({data['remaining_count']}): "
                + " ".join(card_parts)
                + f" bomb:{data['any_bomb']:.0%}"
            )
        return " | ".join(parts)
