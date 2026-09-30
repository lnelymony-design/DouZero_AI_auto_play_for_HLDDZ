"""Offline regression replay for the WeChat miniapp recognizer.

This tool never clicks the game.  It runs the same evidence strategy used by the
live worker over recorded MP4 files:

- local actions: stable hand delta, with cached legal table-play repair;
- opponent actions: remaining-count drop + exact visible-card-count match;
- Pass: explicit text first, turn-order inference as a recovery path;
- game over: zero count / disappeared count badge / disappeared local hand.

When tools/wechat_recording_truth.json contains a matching recording basename,
the replay also reports regression accuracy against settlement facts that were
manually verified from the videos.

Examples:
    python tools/replay_wechat_recording.py game1.mp4 game2.mp4
    python tools/replay_wechat_recording.py --sample-seconds 0.35 game1.mp4
"""
import argparse
import json
import os
from collections import Counter

import cv2

from helpers.WechatCardRecognizer import WechatCardRecognizer


RANK_TO_ENV = {str(i): i for i in range(3, 10)}
RANK_TO_ENV.update(
    {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "2": 17, "X": 20, "D": 30}
)


def legal(cards):
    if not cards:
        return False
    move = sorted(RANK_TO_ENV[c] for c in cards)
    n = len(move)
    counts = Counter(move)

    if n == 1:
        return True
    if n == 2:
        return move[0] == move[1] or move == [20, 30]
    if n == 3:
        return len(counts) == 1
    if n == 4:
        return len(counts) == 1 or (
            len(counts) == 2 and 3 in counts.values()
        )
    if n == 5 and len(counts) == 2:
        return sorted(counts.values()) == [2, 3]

    keys = sorted(counts)

    def continuous(values):
        return (
            len(values) > 1
            and values[-1] <= 14
            and all(b - a == 1 for a, b in zip(values, values[1:]))
        )

    if len(counts) == n and n >= 5 and continuous(keys):
        return True
    if (
        all(value == 2 for value in counts.values())
        and len(counts) >= 3
        and continuous(keys)
    ):
        return True
    if (
        all(value == 3 for value in counts.values())
        and len(counts) >= 2
        and continuous(keys)
    ):
        return True

    count_of_counts = Counter(counts.values())
    if (
        n == 6
        and count_of_counts[4] == 1
        and (count_of_counts[2] == 1 or count_of_counts[1] == 2)
    ):
        return True
    if (
        n == 8
        and (
            (count_of_counts[4] == 1 and count_of_counts[2] == 2)
            or count_of_counts[4] == 2
        )
    ):
        return True

    triples = sorted(
        rank for rank, value in counts.items() if value == 3
    )
    if len(triples) >= 2 and continuous(triples):
        singles = sum(1 for value in counts.values() if value == 1)
        pairs = sum(1 for value in counts.values() if value == 2)
        if len(triples) == singles + 2 * pairs:
            return True
        if len(triples) == pairs and len(counts) == len(triples) * 2:
            return True

    return False


def stable_factory():
    state = {}

    def stable(key, value, frames=2):
        previous, count = state.get(key, (None, 0))
        if value == previous:
            count += 1
        else:
            previous, count = value, 1
        state[key] = (previous, count)
        return value if count >= frames else None

    return stable


def hand_difference(before, after):
    before_counter = Counter(before)
    after_counter = Counter(after)
    if any(
        after_counter[card] > before_counter[card]
        for card in after_counter
    ):
        return None

    missing = before_counter - after_counter
    result = []
    for card in before:
        if missing[card] > 0:
            result.append(card)
            missing[card] -= 1
    return "".join(result)


def remove_cards_preserving_order(hand, cards):
    remove = Counter(cards)
    result = []
    for card in hand:
        if remove[card] > 0:
            remove[card] -= 1
        else:
            result.append(card)
    return "".join(result)


def multiset_subset(cards, pool):
    needed = Counter(cards)
    available = Counter(pool)
    return all(needed[card] <= available[card] for card in needed)


def load_truth(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload.get("recordings", {})


def recognize_settlement_hands(
    video_path,
    truth,
    recognizer,
    sample_seconds=0.35,
):
    """Read revealed end-of-round hands from several late video frames."""
    expected = truth.get("final_remaining", {})
    candidates = {"left": [], "right": [], "me": []}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps else 0.0

    start = max(0.0, duration - 8.0)
    t = start
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, image = cap.read()
        if not ok:
            break

        for side in ("left", "right"):
            count = expected.get(side)
            if count is None or int(count) <= 0:
                continue
            count = int(count)
            if side == "left":
                cards = recognizer.recognize_left_played(
                    image, expected_count=count
                )
            else:
                cards = recognizer.recognize_right_played(
                    image, expected_count=count
                )
            if len(cards) == count:
                candidates[side].append(cards)

        my_count = expected.get("me")
        if my_count is not None and int(my_count) > 0:
            cards = recognizer.recognize_my_hand(image)
            if len(cards) == int(my_count):
                candidates["me"].append(cards)

        t += max(0.20, sample_seconds)

    cap.release()

    result = {}
    for side, values in candidates.items():
        if not values:
            continue
        result[side] = Counter(values).most_common(1)[0][0]
    return result


def simulate(video_path, sample_seconds=0.35):
    recognizer = WechatCardRecognizer()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps * sample_seconds))
    frame_index = 0
    stable = stable_factory()

    pre_landlord_hand = None
    position_code = None
    initialized = False
    bottom_cards = None
    confirmed_hand = None
    expected_side = None

    side_cycle = {"me": "right", "right": "left", "left": "me"}
    first_actor = {0: "right", 1: "me", 2: "left"}

    tracked = {"left": None, "right": None}
    recent_opponent_play = {
        "left": {"cards": "", "time": -999.0},
        "right": {"cards": "", "time": -999.0},
    }
    recent_self_plays = []
    count_missing = {"left": 0, "right": 0}
    hand_missing = 0
    last_pass_sides = set()

    actions = []
    issues = []

    def add_pass(t, side, source="visual"):
        nonlocal expected_side
        actions.append((round(t, 2), side, "PASS", source))
        expected_side = side_cycle[side]

    def pending_self_change(raw_hand):
        if not confirmed_hand or not raw_hand:
            return False
        removed = hand_difference(confirmed_hand, raw_hand)
        return bool(removed)

    def sync_to(t, actor, stable_counts, raw_hand):
        nonlocal expected_side
        guard = 0
        while (
            expected_side is not None
            and expected_side != actor
            and guard < 3
        ):
            if expected_side != "me":
                visible_count = stable_counts.get(expected_side)
                old_count = tracked.get(expected_side)
                if (
                    visible_count is not None
                    and old_count is not None
                    and visible_count < old_count
                ):
                    return False
            elif pending_self_change(raw_hand):
                return False

            add_pass(t, expected_side, source="inferred")
            guard += 1
        return expected_side == actor

    while True:
        ok, image = cap.read()
        if not ok:
            break

        if frame_index % step:
            frame_index += 1
            continue

        t = frame_index / fps if fps else 0.0
        frame_index += 1

        raw_hand = recognizer.recognize_my_hand(image)
        init_hand = stable("init_hand", raw_hand, frames=5)
        live_hand = stable("live_hand", raw_hand, frames=3)

        raw_counts = {
            side: recognizer.recognize_remaining_count(
                image, side, expected=None
            )
            for side in ("left", "right")
        }
        for side in ("left", "right"):
            if raw_counts[side] is None:
                count_missing[side] += 1
            else:
                count_missing[side] = 0
        stable_counts = {
            side: stable(
                f"count_{side}", raw_counts[side], frames=4
            )
            for side in ("left", "right")
        }

        if initialized and not raw_hand:
            hand_missing += 1
        else:
            hand_missing = 0

        if not initialized:
            if init_hand and len(init_hand) == 17:
                pre_landlord_hand = init_hand

            landlord_side = recognizer.detect_landlord_side(image)
            position_map = {"right": 0, "me": 1, "left": 2}
            candidate_position = stable(
                "landlord_badge",
                position_map.get(landlord_side),
                frames=3,
            )

            if init_hand and len(init_hand) == 20:
                candidate_position = 1
            elif stable_counts["left"] == 20:
                candidate_position = 2
            elif stable_counts["right"] == 20:
                candidate_position = 0

            stable_position = stable(
                "position", candidate_position, frames=3
            )
            if stable_position is not None:
                position_code = stable_position

            top_bottom = stable(
                "bottom",
                recognizer.recognize_bottom_cards(image),
                frames=2,
            )
            if (
                position_code == 1
                and init_hand
                and len(init_hand) == 20
                and pre_landlord_hand
            ):
                added = hand_difference(init_hand, pre_landlord_hand)
                if added and len(added) == 3:
                    top_bottom = added

            if top_bottom and len(top_bottom) == 3:
                bottom_cards = top_bottom

            expected_initial = (
                20 if position_code == 1
                else 17 if position_code is not None
                else None
            )
            if (
                position_code is not None
                and init_hand
                and len(init_hand) == expected_initial
                and bottom_cards
            ):
                initialized = True
                confirmed_hand = init_hand
                expected_side = first_actor[position_code]
                landlord_physical = first_actor[position_code]
                tracked = {
                    "left": 20 if landlord_physical == "left" else 17,
                    "right": 20 if landlord_physical == "right" else 17,
                }
                actions.append(
                    (
                        round(t, 2),
                        "INIT",
                        position_code,
                        init_hand,
                        bottom_cards,
                        dict(tracked),
                    )
                )
            continue

        played_values = {
            "left": recognizer.recognize_left_played(image),
            "right": recognizer.recognize_right_played(image),
            "me": recognizer.recognize_my_played(image),
        }

        for side in ("left", "right"):
            cards = stable(
                f"play_{side}",
                played_values[side],
                frames=2,
            )
            if cards:
                recent_opponent_play[side] = {
                    "cards": cards,
                    "time": t,
                }

        self_visual = stable(
            "play_me", played_values["me"], frames=2
        )
        if self_visual and legal(self_visual):
            if (
                not recent_self_plays
                or recent_self_plays[-1]["cards"] != self_visual
            ):
                recent_self_plays.append(
                    {"cards": self_visual, "time": t}
                )
        recent_self_plays[:] = [
            item
            for item in recent_self_plays
            if t - item["time"] <= 6.0
        ]

        self_removed = None
        corrected_after = None
        self_final = False

        if (
            live_hand
            and confirmed_hand
            and live_hand != confirmed_hand
        ):
            removed = hand_difference(confirmed_hand, live_hand)
            if removed == "":
                confirmed_hand = live_hand
            elif removed and legal(removed):
                self_removed = removed
            elif removed:
                delta_counter = Counter(removed)
                candidates = []
                for item in recent_self_plays:
                    candidate = item["cards"]
                    if (
                        legal(candidate)
                        and multiset_subset(candidate, removed)
                        and multiset_subset(candidate, confirmed_hand)
                    ):
                        candidates.append(item)

                if candidates:
                    best = max(
                        candidates,
                        key=lambda item: (
                            len(item["cards"]), item["time"]
                        ),
                    )
                    self_removed = best["cards"]
                    corrected_after = remove_cards_preserving_order(
                        confirmed_hand, self_removed
                    )
                    issues.append(
                        (
                            round(t, 2),
                            "corrected-self",
                            removed,
                            "->",
                            self_removed,
                        )
                    )
                else:
                    issues.append(
                        (round(t, 2), "illegal-self", removed)
                    )

        if (
            not self_removed
            and confirmed_hand
            and expected_side == "me"
        ):
            raw_final = recognizer.recognize_my_played(
                image, expected_count=len(confirmed_hand)
            )
            final_cards = stable(
                "my_final_play", raw_final, frames=2
            )
            if (
                final_cards
                and Counter(final_cards) == Counter(confirmed_hand)
                and legal(final_cards)
            ):
                self_removed = final_cards
                self_final = True

        def opponent_candidate(side):
            old_count = tracked[side]
            new_count = stable_counts[side]
            if old_count is None or old_count <= 0:
                return None

            if new_count is not None and new_count < old_count:
                drop = old_count - new_count
                target = new_count
                source = "count"
            elif (
                new_count is None
                and count_missing[side] >= 4
            ):
                drop = old_count
                target = 0
                source = "badgegone"
            else:
                return None

            cached = recent_opponent_play[side]
            cards = (
                cached["cards"]
                if t - cached["time"] <= 3.0
                else ""
            )
            if not (
                cards
                and len(cards) == drop
                and legal(cards)
            ):
                if side == "left":
                    cards = recognizer.recognize_left_played(
                        image, expected_count=drop
                    )
                else:
                    cards = recognizer.recognize_right_played(
                        image, expected_count=drop
                    )

            if cards and len(cards) == drop and legal(cards):
                return cards, target, source
            return None

        for _ in range(3):
            if expected_side is None:
                break

            actor = None
            payload = None
            order = [
                expected_side,
                side_cycle[expected_side],
                side_cycle[side_cycle[expected_side]],
            ]

            for side in order:
                if side == "me" and self_removed:
                    actor = "me"
                    payload = self_removed
                    break
                if side in ("left", "right"):
                    candidate = opponent_candidate(side)
                    if candidate:
                        actor = side
                        payload = candidate
                        break

            if actor is None:
                break
            if not sync_to(
                t, actor, stable_counts, raw_hand
            ):
                break

            if actor == "me":
                actions.append(
                    (
                        round(t, 2),
                        "me",
                        payload,
                        0 if self_final else len(
                            corrected_after
                            if corrected_after is not None
                            else live_hand
                        ),
                    )
                )
                confirmed_hand = (
                    ""
                    if self_final
                    else corrected_after
                    if corrected_after is not None
                    else live_hand
                )
                self_removed = None
                corrected_after = None
                recent_self_plays.clear()
                expected_side = (
                    None if self_final else "right"
                )
            else:
                cards, new_count, source = payload
                actions.append(
                    (
                        round(t, 2),
                        actor,
                        cards,
                        new_count,
                        source,
                    )
                )
                tracked[actor] = new_count
                recent_opponent_play[actor] = {
                    "cards": "",
                    "time": -999.0,
                }
                expected_side = (
                    None
                    if new_count == 0
                    else side_cycle[actor]
                )

            if expected_side is None:
                break

        pass_sides = stable(
            "passes",
            frozenset(recognizer.detect_pass_sides(image)),
            frames=2,
        )
        if pass_sides is not None:
            current_passes = set(pass_sides)
            for _ in range(2):
                if (
                    expected_side in current_passes
                    and expected_side not in last_pass_sides
                ):
                    add_pass(
                        t, expected_side, source="visual"
                    )
                else:
                    break
            last_pass_sides = current_passes

    cap.release()

    summary = {
        "position_code": position_code,
        "bottom_cards": bottom_cards,
        "tracked_remaining": dict(tracked),
        "my_remaining": len(confirmed_hand or ""),
        "confirmed_my_hand": confirmed_hand or "",
    }
    return actions, issues, summary


def compare_truth(video_path, summary, truth, recognizer, sample_seconds):
    if not truth:
        return None

    expected_counts = truth.get("final_remaining", {})
    observed_counts = {
        "left": summary["tracked_remaining"].get("left"),
        "right": summary["tracked_remaining"].get("right"),
        "me": summary["my_remaining"],
    }

    count_hits = 0
    count_total = 0
    mismatches = []
    for side in ("left", "right", "me"):
        expected = expected_counts.get(side)
        if expected is None:
            continue
        count_total += 1
        if observed_counts.get(side) == expected:
            count_hits += 1
        else:
            mismatches.append(
                (
                    side,
                    observed_counts.get(side),
                    expected,
                )
            )

    revealed = recognize_settlement_hands(
        video_path,
        truth,
        recognizer,
        sample_seconds=sample_seconds,
    )
    expected_hands = truth.get("final_hands", {})
    hand_hits = 0
    hand_total = 0
    hand_mismatches = []
    for side in ("left", "right", "me"):
        expected = expected_hands.get(side)
        if expected is None:
            continue
        if expected == "":
            # Empty winner hand is already validated by the count.
            if expected_counts.get(side) == 0:
                hand_total += 1
                if observed_counts.get(side) == 0:
                    hand_hits += 1
            continue

        actual = revealed.get(side)
        if actual is None:
            hand_mismatches.append((side, None, expected))
            continue

        hand_total += 1
        if Counter(actual) == Counter(expected):
            hand_hits += 1
        else:
            hand_mismatches.append(
                (side, actual, expected)
            )

    return {
        "count_hits": count_hits,
        "count_total": count_total,
        "count_mismatches": mismatches,
        "hand_hits": hand_hits,
        "hand_total": hand_total,
        "hand_mismatches": hand_mismatches,
        "revealed_hands": revealed,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replay WeChat Dou Dizhu recordings through the "
            "recognition state machine."
        )
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="MP4 recordings to replay",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=0.35,
        help="Frame sampling interval; default matches live polling closely.",
    )
    parser.add_argument(
        "--truth-file",
        default=os.path.join(
            os.path.dirname(__file__),
            "wechat_recording_truth.json",
        ),
        help="Optional verified settlement truth JSON.",
    )
    args = parser.parse_args()

    truth_by_name = load_truth(args.truth_file)
    recognizer = WechatCardRecognizer()

    total_count_hits = 0
    total_count_cases = 0
    total_hand_hits = 0
    total_hand_cases = 0

    for video_path in args.videos:
        basename = os.path.basename(video_path)
        print("\n###", basename)

        actions, issues, summary = simulate(
            video_path,
            sample_seconds=max(0.20, args.sample_seconds),
        )
        for action in actions:
            print(action)

        print("SUMMARY", summary)

        if issues:
            print("ISSUES")
            for issue in issues:
                print(issue)

        truth = truth_by_name.get(basename)
        if truth:
            regression = compare_truth(
                video_path,
                summary,
                truth,
                recognizer,
                max(0.20, args.sample_seconds),
            )
            print(
                "REGRESSION remaining-count:",
                f"{regression['count_hits']}/"
                f"{regression['count_total']}",
            )
            for mismatch in regression["count_mismatches"]:
                print(
                    "  count mismatch:",
                    mismatch,
                )

            print(
                "REGRESSION settlement-hands:",
                f"{regression['hand_hits']}/"
                f"{regression['hand_total']}",
            )
            for mismatch in regression["hand_mismatches"]:
                print(
                    "  hand mismatch:",
                    mismatch,
                )

            total_count_hits += regression["count_hits"]
            total_count_cases += regression["count_total"]
            total_hand_hits += regression["hand_hits"]
            total_hand_cases += regression["hand_total"]

    if total_count_cases:
        print(
            "\n=== RECORDING REGRESSION TOTAL ==="
        )
        print(
            f"remaining-count: {total_count_hits}/"
            f"{total_count_cases}"
        )
        print(
            f"settlement-hands: {total_hand_hits}/"
            f"{total_hand_cases}"
        )


if __name__ == "__main__":
    main()
