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
from douzero.env.move_generator import MovesGener
from .legality import can_beat, _candidate_responses

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
        behavior_temperature=0.75,
        min_effective_sample_ratio=0.28,
        residual_behavior_floor=0.35,
        residual_behavior_strength=1.0,
        residual_behavior_temperature=0.55,
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
        self.behavior_temperature = min(
            1.0, max(0.0, float(behavior_temperature))
        )
        self.min_effective_sample_ratio = min(
            0.8, max(0.05, float(min_effective_sample_ratio))
        )
        # Second-stage posterior weighting.  The first behavior stage answers
        # "could this sampled hand plausibly have produced the observed play?".
        # This stage asks the narrower follow-up:
        # "after making that play, is the residual hand shape/control plan
        # plausible compared with nearby legal alternatives?"
        self.residual_behavior_floor = min(
            0.95, max(0.10, float(residual_behavior_floor))
        )
        self.residual_behavior_strength = min(
            3.0, max(0.0, float(residual_behavior_strength))
        )
        self.residual_behavior_temperature = min(
            1.0, max(0.0, float(residual_behavior_temperature))
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
        remaining_now = dict(INITIAL_HAND_SIZES)

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
                        (
                            player,
                            action,
                            last_player,
                            last_move,
                            future_cards,
                            dict(remaining_now),
                        )
                    )
                remaining_now[player] = max(
                    0, remaining_now[player] - len(action)
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

    @staticmethod
    def _run_bonus(flags, minimum):
        """Return a small bonus for long consecutive rank runs."""
        best = current = 0
        for present in flags:
            if present:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return max(0, best - minimum + 1)

    def _residual_hand_score(self, cards):
        """Cheap strategic-quality proxy for a residual hand.

        This is intentionally a weak likelihood feature, not a move evaluator.
        It rewards coherent groups, sequence potential and retained control
        cards while penalising isolated singles.  Only *relative* differences
        among nearby legal alternatives are used.
        """
        counts = Counter(cards)
        singles = sum(1 for n in counts.values() if n == 1)
        pairs = sum(1 for n in counts.values() if n == 2)
        triples = sum(1 for n in counts.values() if n == 3)
        bombs = sum(1 for n in counts.values() if n >= 4)
        rocket = int(counts["D"] > 0 and counts["X"] > 0)

        normal = ("3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
        straight_bonus = self._run_bonus(
            [counts[r] >= 1 for r in normal], 5
        )
        pair_run_bonus = self._run_bonus(
            [counts[r] >= 2 for r in normal], 3
        )

        control = (
            0.55 * counts["D"]
            + 0.42 * counts["X"]
            + 0.24 * counts["2"]
            + 0.08 * counts["A"]
        )

        return (
            -0.34 * singles
            + 0.10 * pairs
            + 0.24 * triples
            + 0.52 * bombs
            + 0.36 * rocket
            + 0.09 * straight_bonus
            + 0.12 * pair_run_bonus
            + control
        )

    @staticmethod
    def _remove_action_from_hand(hand, action):
        remaining = Counter(hand)
        remaining.subtract(Counter(action))
        if any(v < 0 for v in remaining.values()):
            return None
        return _expand(Counter({k: v for k, v in remaining.items() if v > 0}))

    def _nearby_legal_alternatives(self, reconstructed_hand, action, rival_action):
        """Return structurally comparable alternatives to the observed play."""
        action_env = sorted(RealCard2EnvCard[c] for c in action)
        action_info = md.get_move_type(action_env)
        action_type = action_info.get("type")
        action_len = len(action_env)
        hand_env = sorted(RealCard2EnvCard[c] for c in reconstructed_hand)

        if rival_action:
            rival_env = sorted(RealCard2EnvCard[c] for c in rival_action)
            candidates = _candidate_responses(hand_env, rival_env)
        else:
            candidates = MovesGener(hand_env).gen_moves()

        comparable = []
        seen = set()
        for move in candidates:
            move = sorted(move)
            if not move:
                continue
            info = md.get_move_type(move)
            if info.get("type") != action_type or len(move) != action_len:
                continue
            key = tuple(move)
            if key in seen:
                continue
            seen.add(key)
            comparable.append(move)
            if len(comparable) >= 24:
                break

        if tuple(action_env) not in seen:
            comparable.append(action_env)
        return comparable

    def _residual_strategy_factor(
        self,
        reconstructed_hand,
        action,
        rival_action,
        rival_player=None,
        acting_player=None,
        remaining_before=None,
    ):
        """Second likelihood weight based on the hand left after the play.

        Example: if a sampled world says a player had KK plus many clean
        alternatives, but they chose to lead a single K and leave an awkward K,
        that world becomes less likely.  Conversely a play that preserves a
        coherent pair/straight/control structure is more plausible.

        Emergency defence is treated more softly: when an enemy is nearly out
        of cards, spending control cards can be strategically reasonable.
        """
        if not action or self.residual_behavior_strength <= 0:
            return 1.0

        observed_remaining = self._remove_action_from_hand(
            reconstructed_hand, action
        )
        if observed_remaining is None:
            return self.residual_behavior_floor

        alternatives = self._nearby_legal_alternatives(
            reconstructed_hand, action, rival_action
        )
        if len(alternatives) <= 1:
            return 1.0

        observed_score = self._residual_hand_score(observed_remaining)
        scores = []
        for move in alternatives:
            move_real = [EnvCard2RealCard[c] for c in move]
            rest = self._remove_action_from_hand(reconstructed_hand, move_real)
            if rest is not None:
                scores.append(self._residual_hand_score(rest))
        if not scores:
            return 1.0

        best_score = max(scores)
        gap = max(0.0, best_score - observed_score)
        if gap <= 1e-9:
            return 1.0

        emergency = False
        if rival_player and remaining_before:
            rival_left = remaining_before.get(rival_player)
            if (
                rival_left is not None
                and rival_left <= 2
                and acting_player is not None
                and not self._same_team(rival_player, acting_player)
            ):
                emergency = True

        strength = self.residual_behavior_strength * (0.35 if emergency else 1.0)
        factor = math.exp(-strength * gap)
        return max(self.residual_behavior_floor, min(1.0, factor))

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

    @staticmethod
    def _tempered_weights(log_weights, exponent):
        if not log_weights:
            return []
        exponent = max(0.0, float(exponent))
        max_log = max(log_weights)
        return [
            math.exp((value - max_log) * exponent)
            for value in log_weights
        ]

    @staticmethod
    def _effective_sample_ratio_from_weights(weights):
        if not weights:
            return 0.0
        total = sum(weights)
        if total <= 0:
            return 0.0
        normalized_sq = sum((w / total) ** 2 for w in weights)
        if normalized_sq <= 0:
            return 0.0
        ess = 1.0 / normalized_sq
        return ess / len(weights)

    def _choose_behavior_temperature(self, log_weights):
        """Use the strongest behavior weighting that preserves a minimum ESS.

        The action model is heuristic rather than a calibrated human policy.
        Adaptive tempering prevents many correlated actions from collapsing the
        posterior onto a handful of particles and producing false certainty.
        """
        base = self.behavior_temperature
        if not log_weights or base <= 0:
            return 0.0

        base_weights = self._tempered_weights(log_weights, base)
        if (
            self._effective_sample_ratio_from_weights(base_weights)
            >= self.min_effective_sample_ratio
        ):
            return base

        # exponent=0 gives uniform weights (ESS ratio=1).  Binary-search the
        # largest exponent that still satisfies the requested ESS floor.
        low = 0.0
        high = base
        for _ in range(14):
            mid = (low + high) / 2.0
            weights = self._tempered_weights(log_weights, mid)
            ratio = self._effective_sample_ratio_from_weights(weights)
            if ratio >= self.min_effective_sample_ratio:
                low = mid
            else:
                high = mid
        return low

    def _choose_second_stage_temperature(
        self, base_weights, residual_log_weights, max_exponent
    ):
        """Apply as much residual-hand weighting as ESS safely allows."""
        if (
            not base_weights
            or not residual_log_weights
            or max_exponent <= 0
        ):
            return 0.0

        max_log = max(residual_log_weights)

        def combined(exponent):
            return [
                base * math.exp((logw - max_log) * exponent)
                for base, logw in zip(base_weights, residual_log_weights)
            ]

        if (
            self._effective_sample_ratio_from_weights(
                combined(max_exponent)
            )
            >= self.min_effective_sample_ratio
        ):
            return max_exponent

        low = 0.0
        high = max_exponent
        for _ in range(14):
            mid = (low + high) / 2.0
            if (
                self._effective_sample_ratio_from_weights(combined(mid))
                >= self.min_effective_sample_ratio
            ):
                low = mid
            else:
                high = mid
        return low

    def infer(self, should_cancel=None):
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

        sampled_hands = []
        log_weights = []
        residual_log_weights = []
        for sample_index in range(self.sample_count):
            if (
                should_cancel is not None
                and sample_index % 16 == 0
                and should_cancel()
            ):
                self._latest_weighted_samples = []
                self._latest_total_weight = 0.0
                return {
                    "players": {},
                    "samples": len(sampled_hands),
                    "effective_samples": 0.0,
                    "cancelled": True,
                    "warning": "Inference cancelled by newer public state.",
                }

            hands = self._sample_current_hands(hidden_pool, remaining_counts, forced_landlord)
            if hands is None:
                continue

            log_weight = 0.0
            for player, rival_player, rival_action, future_cards in pass_contexts:
                reconstructed = list(hands[player]) + list(future_cards)
                hand_env = [RealCard2EnvCard[c] for c in reconstructed]
                rival_env = [RealCard2EnvCard[c] for c in rival_action]
                if can_beat(hand_env, rival_env):
                    factor = self._pass_penalty_for(player, rival_player)
                    log_weight += math.log(max(1e-12, factor))

            residual_log_weight = 0.0
            for (
                player,
                action,
                rival_player,
                rival_action,
                future_cards,
                remaining_before,
            ) in play_contexts:
                reconstructed = (
                    list(hands[player])
                    + list(action)
                    + list(future_cards)
                )
                factor = self._play_behavior_factor(
                    reconstructed,
                    action,
                    rival_action,
                )
                log_weight += math.log(max(1e-12, factor))

                residual_factor = self._residual_strategy_factor(
                    reconstructed,
                    action,
                    rival_action,
                    rival_player=rival_player,
                    acting_player=player,
                    remaining_before=remaining_before,
                )
                residual_log_weight += math.log(
                    max(1e-12, residual_factor)
                )

            sampled_hands.append(hands)
            log_weights.append(log_weight)
            residual_log_weights.append(residual_log_weight)

        temperature_used = self._choose_behavior_temperature(log_weights)
        first_stage_weights = self._tempered_weights(
            log_weights, temperature_used
        )
        first_stage_ess_ratio = self._effective_sample_ratio_from_weights(
            first_stage_weights
        )

        residual_temperature_used = self._choose_second_stage_temperature(
            first_stage_weights,
            residual_log_weights,
            self.residual_behavior_temperature,
        )
        if residual_temperature_used > 0:
            max_residual = max(residual_log_weights)
            residual_weights = [
                math.exp(
                    (value - max_residual) * residual_temperature_used
                )
                for value in residual_log_weights
            ]
            weights = [
                base * extra
                for base, extra in zip(
                    first_stage_weights, residual_weights
                )
            ]
        else:
            weights = first_stage_weights

        weighted_samples = list(zip(sampled_hands, weights))

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
        prior_denominator = max(1, len(sampled_hands))
        for player in self.opponents:
            rank_stats = {
                c: {
                    "one_plus": 0.0,
                    "pair_plus": 0.0,
                    "triple_plus": 0.0,
                    "bomb": 0.0,
                    "prior_one_plus": 0.0,
                    "behavior_delta": 0.0,
                }
                for c in CARD_ORDER
            }

            # Uniform legal-allocation baseline before either behavior stage.
            for hands in sampled_hands:
                prior_counts = Counter(hands[player])
                for card in CARD_ORDER:
                    if prior_counts[card] >= 1:
                        rank_stats[card]["prior_one_plus"] += (
                            1.0 / prior_denominator
                        )

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

            for card in CARD_ORDER:
                rank_stats[card]["behavior_delta"] = (
                    rank_stats[card]["one_plus"]
                    - rank_stats[card]["prior_one_plus"]
                )

            top_hands = sorted(hand_mass.items(), key=lambda x: x[1], reverse=True)[:5]
            result[player] = {
                "remaining_count": remaining_counts[player],
                "cards": {
                    card: {k: round(v, 4) for k, v in stats.items()}
                    for card, stats in rank_stats.items()
                },
                "any_bomb": round(any_bomb, 4),
                "rocket": round(rocket, 4),
                "combo_risks": {
                    "has_2": round(rank_stats["2"]["one_plus"], 4),
                    "pair_2": round(rank_stats["2"]["pair_plus"], 4),
                    "triple_2": round(rank_stats["2"]["triple_plus"], 4),
                    "pair_A": round(rank_stats["A"]["pair_plus"], 4),
                    "triple_A": round(rank_stats["A"]["triple_plus"], 4),
                    "pair_K": round(rank_stats["K"]["pair_plus"], 4),
                    "any_bomb": round(any_bomb, 4),
                    "rocket": round(rocket, 4),
                },
                "top_sampled_hands": [
                    {"hand": hand, "probability": round(prob, 4)}
                    for hand, prob in top_hands
                ],
                "top_behavior_shifts": [
                    {
                        "card": card,
                        "prior": round(rank_stats[card]["prior_one_plus"], 4),
                        "posterior": round(rank_stats[card]["one_plus"], 4),
                        "delta": round(rank_stats[card]["behavior_delta"], 4),
                    }
                    for card in sorted(
                        CARD_ORDER,
                        key=lambda c: abs(
                            rank_stats[c]["behavior_delta"]
                        ),
                        reverse=True,
                    )[:5]
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
            "behavior_temperature_configured": self.behavior_temperature,
            "behavior_temperature_used": round(temperature_used, 4),
            "first_stage_effective_sample_ratio": round(
                first_stage_ess_ratio, 3
            ),
            "residual_behavior_floor": self.residual_behavior_floor,
            "residual_behavior_strength": self.residual_behavior_strength,
            "residual_behavior_temperature_configured": (
                self.residual_behavior_temperature
            ),
            "residual_behavior_temperature_used": round(
                residual_temperature_used, 4
            ),
            "residual_evidence_count": len(play_contexts),
            "min_effective_sample_ratio": self.min_effective_sample_ratio,
            "effective_sample_ratio": round(
                effective_samples / len(weighted_samples), 3
            ) if weighted_samples else 0.0,
            "behavior_model": "heuristic_v2_tempered",
        }

    def posterior_worlds(self, max_worlds=24, allow_infer=True):
        """Return deterministic weighted opponent-hand worlds for rollout.

        The returned probabilities sum to one across the selected worlds.
        Selection is stratified over posterior mass rather than taking only the
        highest-weight particles, which keeps lower-probability plausible worlds
        represented in downstream simulation.

        Async rollout must pass allow_infer=False so exporting a job can never
        trigger a second hidden Monte-Carlo inference on the live recognition
        path.
        """
        if not self._latest_weighted_samples or self._latest_total_weight <= 0:
            if not allow_infer:
                return []
            self.infer()

        samples = self._latest_weighted_samples
        total = self._latest_total_weight
        if not samples or total <= 0:
            return []

        max_worlds = max(1, min(int(max_worlds), len(samples)))
        cumulative = []
        running = 0.0
        for hands, weight in samples:
            running += weight / total
            cumulative.append((running, hands, weight / total))

        selected = []
        used = set()
        for i in range(max_worlds):
            target = (i + 0.5) / max_worlds
            for idx, (mass, hands, prob) in enumerate(cumulative):
                if mass >= target:
                    if idx not in used:
                        used.add(idx)
                        selected.append((hands, prob))
                    break

        if not selected:
            return []

        selected_total = sum(prob for _, prob in selected)
        return [
            {
                "hands": {
                    player: _sorted_cards(cards)
                    for player, cards in hands.items()
                },
                "probability": prob / selected_total,
            }
            for hands, prob in selected
        ]

    def response_profile(self, action, max_samples=320):
        """Describe how each opponent and the enemy side can answer an action.

        The top-level probabilities aggregate only true enemies:
        - landlord: both farmers are enemies;
        - farmer: only the landlord is an enemy.

        The players field always contains both physical opponents by logical
        position, including a farmer teammate. This lets the UI explain who can
        answer without treating a teammate response as enemy risk.

        For every scope, the three outcomes are mutually exclusive:
        ordinary_beat, bomb_only, and unbeatable.
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
            selected = [
                samples[min(len(samples) - 1, int(i * step))]
                for i in range(max_samples)
            ]
        else:
            selected = samples

        enemies = (
            list(self.opponents)
            if self.my_position == "landlord"
            else ["landlord"]
        )
        rival_env = [RealCard2EnvCard[c] for c in cards]
        bomb_types = {md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB}

        totals = {
            player: {
                "ordinary_weight": 0.0,
                "bomb_only_weight": 0.0,
                "unbeatable_weight": 0.0,
            }
            for player in self.opponents
        }
        aggregate = {
            "ordinary_weight": 0.0,
            "bomb_only_weight": 0.0,
            "unbeatable_weight": 0.0,
        }
        total_weight = 0.0

        def response_kind(hand):
            if hand is None:
                return "unbeatable"
            hand_env = [RealCard2EnvCard[c] for c in hand]
            responses = _candidate_responses(hand_env, rival_env)
            has_bomb = False
            for response in responses:
                response_type = md.get_move_type(response).get("type")
                if response_type in bomb_types:
                    has_bomb = True
                else:
                    return "ordinary"
            return "bomb_only" if has_bomb else "unbeatable"

        for hands, weight in selected:
            total_weight += weight
            sample_kinds = {}

            for player in self.opponents:
                kind = response_kind(hands.get(player))
                sample_kinds[player] = kind
                totals[player][f"{kind}_weight"] += weight

            enemy_kinds = [
                sample_kinds[player]
                for player in enemies
                if player in sample_kinds
            ]
            if "ordinary" in enemy_kinds:
                aggregate["ordinary_weight"] += weight
            elif "bomb_only" in enemy_kinds:
                aggregate["bomb_only_weight"] += weight
            else:
                aggregate["unbeatable_weight"] += weight

        if total_weight <= 0:
            return None

        def normalize(bucket):
            ordinary = bucket["ordinary_weight"] / total_weight
            bomb_only = bucket["bomb_only_weight"] / total_weight
            unbeatable = bucket["unbeatable_weight"] / total_weight
            can_beat_prob = ordinary + bomb_only
            return {
                "ordinary_beat": ordinary,
                "bomb_only": bomb_only,
                "unbeatable": unbeatable,
                "can_beat": can_beat_prob,
                "pressure": ordinary + 0.35 * bomb_only,
            }

        result = normalize(aggregate)
        result["players"] = {
            player: normalize(bucket)
            for player, bucket in totals.items()
        }
        return result

    def response_risk(self, action, max_samples=320):
        profile = self.response_profile(action, max_samples=max_samples)
        return None if profile is None else profile["can_beat"]

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
