#!/usr/bin/env python3
"""Offline replay validator for WeChat miniapp Dou Dizhu recordings.

Usage:
    python tools/wechat_video_replay.py "path/to/game.mp4"
    python tools/wechat_video_replay.py game1.mp4 game2.mp4 --json
    python tools/wechat_video_replay.py game.mp4 --baseline tools/wechat_video_baselines.json

The validator never clicks the game. It replays recorded frames through
WechatCardRecognizer and checks the same invariants used by the live worker:
- role/initial hand/bottom cards;
- local-player hand deltas;
- opponent count drops matched to visible play lengths;
- Pass reconstruction from turn order;
- non-increasing opponent remaining counts.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from helpers.WechatCardRecognizer import WechatCardRecognizer


POSITIONS = ("landlord_up", "landlord", "landlord_down")
SIDE_CYCLE = {"me": "right", "right": "left", "left": "me"}
LANDLORD_START_SIDE = {0: "right", 1: "me", 2: "left"}


class Stable:
    def __init__(self):
        self._state = {}

    def get(self, key, value, frames=2):
        previous, count = self._state.get(key, (object(), 0))
        if value == previous:
            count += 1
        else:
            previous, count = value, 1
        self._state[key] = (previous, count)
        return value if count >= frames else None


def hand_difference(before: str, after: str):
    before_counter = Counter(before)
    after_counter = Counter(after)
    if any(after_counter[card] > before_counter[card] for card in after_counter):
        return None

    missing = before_counter - after_counter
    result = []
    for card in before:
        if missing[card] > 0:
            result.append(card)
            missing[card] -= 1
    return "".join(result)


def display(cards: str):
    names = {"D": "大王", "X": "小王", "T": "10"}
    return " ".join(names.get(card, card) for card in cards)


@dataclass
class ReplayResult:
    file: str
    initialized: bool = False
    position: str | None = None
    initial_hand: str = ""
    bottom_cards: str = ""
    left_initial_count: int | None = None
    right_initial_count: int | None = None
    actions: list = field(default_factory=list)
    count_trace: dict = field(default_factory=lambda: {"left": [], "right": []})
    unresolved_drops: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    duration: float = 0.0

    def as_dict(self):
        return {
            "file": self.file,
            "initialized": self.initialized,
            "position": self.position,
            "initial_hand": self.initial_hand,
            "bottom_cards": self.bottom_cards,
            "left_initial_count": self.left_initial_count,
            "right_initial_count": self.right_initial_count,
            "actions": self.actions,
            "count_trace": self.count_trace,
            "unresolved_drops": self.unresolved_drops,
            "issues": self.issues,
            "duration": round(self.duration, 2),
        }


def replay(path: str, sample_seconds=0.35):
    recognizer = WechatCardRecognizer()
    stable = Stable()
    result = ReplayResult(file=os.path.basename(path))

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        result.issues.append("video_open_failed")
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    result.duration = total_frames / fps if fps else 0
    step = max(1, round(fps * sample_seconds))

    frame_index = 0
    initialized = False
    pre_landlord_hand = None
    position_code = None
    confirmed_my_hand = None
    expected_side = None
    tracked = {"left": None, "right": None}
    pre_counts = {"left": None, "right": None}
    recent_play = {
        "left": {"cards": "", "time": -999.0},
        "right": {"cards": "", "time": -999.0},
    }
    count_missing_frames = {"left": 0, "right": 0}
    self_hand_missing_frames = 0
    unresolved_since = {"left": None, "right": None}
    last_passes = set()

    def add_action(t, side, action, source, remaining=None):
        result.actions.append({
            "time": round(t, 2),
            "side": side,
            "action": action,
            "source": source,
            "remaining": remaining,
        })

    def infer_pass(t, side, source="inferred"):
        nonlocal expected_side
        add_action(t, side, "Pass", source)
        expected_side = SIDE_CYCLE[side]

    def sync_to_actor(t, actor, current_counts, raw_hand):
        nonlocal expected_side
        guard = 0
        while expected_side is not None and expected_side != actor and guard < 3:
            if expected_side == "me":
                if confirmed_my_hand and raw_hand:
                    removed = hand_difference(confirmed_my_hand, raw_hand)
                    if removed:
                        return False
            else:
                count = current_counts.get(expected_side)
                old = tracked.get(expected_side)
                if count is not None and old is not None and count < old:
                    return False
            infer_pass(t, expected_side)
            guard += 1
        return expected_side == actor

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step:
            frame_index += 1
            continue

        t = frame_index / fps
        frame_index += 1

        raw_hand = recognizer.recognize_my_hand(frame)
        init_hand = stable.get("init_hand", raw_hand, 5)
        live_hand = stable.get("live_hand", raw_hand, 3)

        raw_counts = {
            side: recognizer.recognize_remaining_count(frame, side, expected=None)
            for side in ("left", "right")
        }
        for side in ("left", "right"):
            if raw_counts[side] is None:
                count_missing_frames[side] += 1
            else:
                count_missing_frames[side] = 0

        if initialized and not raw_hand:
            self_hand_missing_frames += 1
        else:
            self_hand_missing_frames = 0
        counts = {
            side: stable.get(f"count_{side}", raw_counts[side], 4)
            for side in ("left", "right")
        }

        if not initialized:
            for side in ("left", "right"):
                if counts[side] is not None:
                    pre_counts[side] = counts[side]

            if init_hand and len(init_hand) == 17:
                pre_landlord_hand = init_hand

            badge = recognizer.detect_landlord_side(frame)
            pos_map = {"right": 0, "me": 1, "left": 2}
            badge_position = stable.get("badge_position", pos_map.get(badge), 3)

            inferred = badge_position
            if init_hand and len(init_hand) == 20:
                inferred = 1
            elif pre_counts["left"] == 20:
                inferred = 2
            elif pre_counts["right"] == 20:
                inferred = 0

            candidate_position = stable.get("position", inferred, 3)
            if candidate_position is not None:
                position_code = candidate_position

            top_bottom = stable.get(
                "bottom", recognizer.recognize_bottom_cards(frame), 2
            )
            bottom = top_bottom
            if (
                position_code == 1
                and init_hand
                and len(init_hand) == 20
                and pre_landlord_hand
            ):
                added = hand_difference(init_hand, pre_landlord_hand)
                if added and len(added) == 3:
                    bottom = added

            expected_count = (
                20 if position_code == 1 else 17
                if position_code is not None else None
            )
            if (
                position_code is not None
                and init_hand
                and len(init_hand) == expected_count
                and bottom
                and len(bottom) == 3
            ):
                initialized = True
                result.initialized = True
                result.position = POSITIONS[position_code]
                result.initial_hand = init_hand
                result.bottom_cards = bottom
                confirmed_my_hand = init_hand
                expected_side = LANDLORD_START_SIDE[position_code]
                landlord_side = expected_side
                tracked["left"] = 20 if landlord_side == "left" else 17
                tracked["right"] = 20 if landlord_side == "right" else 17
                result.left_initial_count = tracked["left"]
                result.right_initial_count = tracked["right"]
                result.count_trace["left"].append(tracked["left"])
                result.count_trace["right"].append(tracked["right"])
            continue

        # Always cache opponent play candidates.
        play_candidates = {
            "left": recognizer.recognize_left_played(frame),
            "right": recognizer.recognize_right_played(frame),
        }
        for side in ("left", "right"):
            cards = stable.get(f"play_{side}", play_candidates[side], 2)
            if cards:
                recent_play[side] = {"cards": cards, "time": t}

        # Local player: hand delta is authoritative.
        local_action = None
        local_final_out = False
        if live_hand and confirmed_my_hand and live_hand != confirmed_my_hand:
            removed = hand_difference(confirmed_my_hand, live_hand)
            if removed:
                local_action = removed
            elif removed == "":
                confirmed_my_hand = live_hand

        if (
            not local_action
            and confirmed_my_hand
            and self_hand_missing_frames >= 4
        ):
            final_cards = recognizer.recognize_my_played(
                frame, expected_count=len(confirmed_my_hand)
            )
            if final_cards and len(final_cards) == len(confirmed_my_hand):
                local_action = final_cards
                local_final_out = True

        if local_action and sync_to_actor(t, "me", counts, raw_hand):
            remaining = 0 if local_final_out else len(live_hand)
            add_action(t, "me", local_action, "hand_delta", remaining)
            confirmed_my_hand = "" if local_final_out else live_hand
            expected_side = None if local_final_out else "right"

        # Opponents: count drop + visible play length must agree.
        for side in ("left", "right"):
            new_count = counts[side]
            old_count = tracked[side]
            if old_count is None:
                continue
            if new_count is not None and new_count > old_count:
                result.issues.append(
                    f"{t:.1f}s {side} count increased {old_count}->{new_count}"
                )
                continue
            if new_count is not None and new_count == old_count:
                unresolved_since[side] = None
                continue

            source = None
            if new_count is not None and new_count < old_count:
                drop = old_count - new_count
                target_count = new_count
                source = "count+play"
            elif (
                new_count is None
                and count_missing_frames[side] >= 4
                and old_count > 0
            ):
                drop = old_count
                target_count = 0
                source = "badge_gone+all_out"
            else:
                continue

            info = recent_play[side]
            cards = info["cards"] if t - info["time"] <= 3.0 else ""
            if not cards or len(cards) != drop:
                cards = (
                    recognizer.recognize_left_played(frame, expected_count=drop)
                    if side == "left"
                    else recognizer.recognize_right_played(frame, expected_count=drop)
                )

            if cards and len(cards) == drop:
                if sync_to_actor(t, side, counts, raw_hand):
                    add_action(t, side, cards, source, target_count)
                    tracked[side] = target_count
                    result.count_trace[side].append(target_count)
                    recent_play[side] = {"cards": "", "time": -999.0}
                    unresolved_since[side] = None
                    expected_side = None if target_count == 0 else SIDE_CYCLE[side]
            else:
                if unresolved_since[side] is None:
                    unresolved_since[side] = t
                elif t - unresolved_since[side] >= 2.5:
                    marker = {
                        "time": round(t, 2),
                        "side": side,
                        "old": old_count,
                        "new": target_count,
                        "drop": drop,
                        "visible": cards,
                    }
                    if not result.unresolved_drops or result.unresolved_drops[-1] != marker:
                        result.unresolved_drops.append(marker)

        # Explicit Pass. Later hard evidence can still reconstruct a missed Pass.
        passes = stable.get(
            "passes", frozenset(recognizer.detect_pass_sides(frame)), 2
        )
        if passes is not None:
            pass_set = set(passes)
            if (
                expected_side in pass_set
                and expected_side not in last_passes
            ):
                infer_pass(t, expected_side, source="visual")
            last_passes = pass_set

    cap.release()

    if not result.initialized:
        result.issues.append("round_never_initialized")

    for side in ("left", "right"):
        trace = result.count_trace[side]
        if any(b > a for a, b in zip(trace, trace[1:])):
            result.issues.append(f"{side}_count_trace_not_monotonic")

    return result


def compare_baseline(result: ReplayResult, baseline: dict):
    expected = baseline.get(result.file)
    if not expected:
        return []

    errors = []
    checks = (
        ("position", result.position),
        ("initial_hand", result.initial_hand),
        ("bottom_cards", result.bottom_cards),
        ("left_initial_count", result.left_initial_count),
        ("right_initial_count", result.right_initial_count),
    )
    for key, actual in checks:
        if key in expected and expected[key] != actual:
            errors.append(
                f"baseline {key}: expected {expected[key]!r}, got {actual!r}"
            )
    return errors


def print_report(result: ReplayResult, baseline_errors):
    print(f"\n=== {result.file} ===")
    print(
        f"init={result.initialized} position={result.position} "
        f"hand={display(result.initial_hand)} "
        f"bottom={display(result.bottom_cards)}"
    )
    print(
        f"initial counts: left={result.left_initial_count} "
        f"right={result.right_initial_count}"
    )
    print(f"actions={len(result.actions)} unresolved_drops={len(result.unresolved_drops)}")
    for action in result.actions:
        action_text = (
            action["action"]
            if action["action"] == "Pass"
            else display(action["action"])
        )
        suffix = (
            f" remaining={action['remaining']}"
            if action.get("remaining") is not None
            else ""
        )
        print(
            f"{action['time']:6.2f}s {action['side']:>5} "
            f"{action_text:<28} [{action['source']}]{suffix}"
        )

    problems = list(result.issues) + list(baseline_errors)
    if result.unresolved_drops:
        problems.append(f"{len(result.unresolved_drops)} unresolved opponent count drops")

    if problems:
        print("RESULT: FAIL")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("RESULT: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument(
        "--baseline",
        default=str(Path(__file__).with_name("wechat_video_baselines.json")),
    )
    parser.add_argument("--sample-seconds", type=float, default=0.35)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    baseline = {}
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline, "r", encoding="utf-8") as f:
            baseline = json.load(f)

    reports = []
    exit_code = 0
    for video in args.videos:
        result = replay(video, sample_seconds=args.sample_seconds)
        baseline_errors = compare_baseline(result, baseline)
        reports.append({
            "result": result.as_dict(),
            "baseline_errors": baseline_errors,
        })

        if result.issues or result.unresolved_drops or baseline_errors:
            exit_code = 1

        if not args.json:
            print_report(result, baseline_errors)

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
