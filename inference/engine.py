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
        random_seed=None,
    ):
        if my_position not in POSITIONS:
            raise ValueError(f"Unknown position: {my_position}")
        self.my_position = my_position
        self.initial_my_hand = list(my_hand_cards)
        self.three_landlord_cards = list(three_landlord_cards or "")
        self.sample_count = max(200, int(sample_count))
        self.pass_penalty = min(1.0, max(0.05, float(pass_penalty)))
        self.rng = random.Random(random_seed)
        self.history = []

        self.opponents = [p for p in POSITIONS if p != self.my_position]

    def reset(self):
        self.history = []

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

    def _build_pass_contexts(self):
        """For each opponent Pass, store the rival move and later cards to restore.

        A sampled *current* hand can be reconstructed at an earlier Pass by
        adding cards that the same player played after that Pass.
        """
        contexts = []
        last_move = ""
        pass_streak = 0

        for idx, (player, action) in enumerate(self.history):
            if not action:
                if player in self.opponents and last_move:
                    future_cards = []
                    for later_player, later_action in self.history[idx + 1:]:
                        if later_player == player and later_action:
                            future_cards.extend(list(later_action))
                    contexts.append((player, last_move, future_cards))
                pass_streak += 1
                if pass_streak >= 2:
                    last_move = ""
                    pass_streak = 0
            else:
                last_move = action
                pass_streak = 0

        return contexts

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
        played = self._played_by_player()
        current_my_hand = self._current_my_hand(played)
        hidden_pool = self._current_hidden_pool(played, current_my_hand)
        remaining_counts = self._remaining_counts(played)
        forced_landlord = self._forced_landlord_cards(played)
        pass_contexts = self._build_pass_contexts()

        weighted_samples = []
        for _ in range(self.sample_count):
            hands = self._sample_current_hands(hidden_pool, remaining_counts, forced_landlord)
            if hands is None:
                continue

            weight = 1.0
            for player, rival_action, future_cards in pass_contexts:
                reconstructed = list(hands[player]) + list(future_cards)
                hand_env = [RealCard2EnvCard[c] for c in reconstructed]
                rival_env = [RealCard2EnvCard[c] for c in rival_action]
                if can_beat(hand_env, rival_env):
                    weight *= self.pass_penalty

            if weight > 0:
                weighted_samples.append((hands, weight))

        total_weight = sum(w for _, w in weighted_samples)
        if total_weight <= 0:
            return {
                "players": {},
                "samples": 0,
                "effective_samples": 0.0,
                "warning": "No legal hidden-card samples remain.",
            }

        sum_sq = sum((w / total_weight) ** 2 for _, w in weighted_samples)
        effective_samples = 1.0 / sum_sq if sum_sq else 0.0

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
            "pass_penalty": self.pass_penalty,
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
