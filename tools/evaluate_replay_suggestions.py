"""Offline compare raw DouZero and risk-adjusted suggestions from replay JSON.

Input files come from:
    python tools/replay_wechat_recording.py --json-dir replay_json game1.mp4 ...

The tool does not claim that matching the recorded human play means a strategy is
better.  It reports decision divergence and response-risk differences for tuning.
"""

import argparse
from collections import Counter
import json
import os
from statistics import mean

from constants import AllEnvCard, RealCard2EnvCard
from douzero.env.game_new import GameEnv
from douzero.evaluation.deep_agent_new import DeepAgent
from inference import HandInferenceEngine, adjust_candidates


POSITIONS = ["landlord_up", "landlord", "landlord_down"]
MODEL_PATHS = {
    "landlord": "baselines/resnet/resnet_landlord.ckpt",
    "landlord_up": "baselines/resnet/resnet_landlord_up.ckpt",
    "landlord_down": "baselines/resnet/resnet_landlord_down.ckpt",
}


def same_action(a, b):
    a = "" if a in (None, "", "Pass", "PASS") else str(a)
    b = "" if b in (None, "", "Pass", "PASS") else str(b)
    return Counter(a) == Counter(b)


def player_for_side(position_code, side):
    if side == "me":
        return POSITIONS[position_code]
    if side == "right":
        return POSITIONS[(position_code + 1) % 3]
    if side == "left":
        return POSITIONS[(position_code + 2) % 3]
    raise ValueError(side)


def init_env(position_code, my_hand, bottom_cards):
    my_position = POSITIONS[position_code]
    my_env = sorted(RealCard2EnvCard[c] for c in my_hand)
    bottom_env = sorted(RealCard2EnvCard[c] for c in bottom_cards)

    other = list(AllEnvCard)
    for card in my_env:
        other.remove(card)

    right_index = (position_code + 1) % 3
    left_index = (position_code + 2) % 3
    data = {
        "three_landlord_cards": bottom_env,
        my_position: my_env,
        POSITIONS[right_index]: (
            other[0:17] if right_index != 1 else other[17:]
        ),
        POSITIONS[left_index]: (
            other[0:17] if right_index == 1 else other[17:]
        ),
    }

    agent = DeepAgent(my_position, MODEL_PATHS[my_position])
    env = GameEnv([my_position, agent])
    env.card_play_init(data)
    return env


def parse_replay(path):
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    init_event = next(
        (event for event in payload.get("actions", []) if len(event) > 1 and event[1] == "INIT"),
        None,
    )
    if init_event is None:
        raise ValueError("Replay has no INIT event")

    _, _, position_code, my_hand, bottom_cards, _ = init_event
    events = []
    found_init = False
    for event in payload.get("actions", []):
        if event is init_event:
            found_init = True
            continue
        if not found_init or len(event) < 3:
            continue

        side = event[1]
        if side not in ("me", "left", "right"):
            continue

        raw_action = event[2]
        action = "" if raw_action == "PASS" else raw_action
        events.append((side, action))

    return {
        "video": payload.get("video", os.path.basename(path)),
        "position_code": int(position_code),
        "my_hand": str(my_hand),
        "bottom_cards": str(bottom_cards),
        "events": events,
    }


def evaluate_one(
    replay,
    weights,
    sample_count=1000,
    min_ess=0.40,
    max_candidates=3,
):
    position_code = replay["position_code"]
    my_position = POSITIONS[position_code]

    env = init_env(
        position_code,
        replay["my_hand"],
        replay["bottom_cards"],
    )
    inference = HandInferenceEngine(
        my_position=my_position,
        my_hand_cards=replay["my_hand"],
        three_landlord_cards=replay["bottom_cards"],
        sample_count=sample_count,
    )

    stats = {
        weight: {
            "turns": 0,
            "enabled": 0,
            "changed": 0,
            "raw_match": 0,
            "adjusted_match": 0,
            "raw_risks": [],
            "adjusted_risks": [],
            "examples": [],
        }
        for weight in weights
    }

    for event_index, (side, actual_action) in enumerate(replay["events"]):
        player = player_for_side(position_code, side)

        if player == my_position:
            if env.acting_player_position != my_position:
                raise ValueError(
                    f"turn mismatch before event {event_index}: "
                    f"env={env.acting_player_position}, replay={player}"
                )

            inference_result = inference.infer()
            _, action_list = env.step(
                my_position,
                action=None,
                update=False,
            )
            candidate_pool = action_list[:max_candidates]
            triplets = []
            for action_text, score_text in candidate_pool:
                risk = (
                    None
                    if action_text == "Pass"
                    else inference.response_risk(action_text)
                )
                triplets.append((action_text, score_text, risk))

            raw_top = triplets[0] if triplets else None
            ess = inference_result.get("effective_sample_ratio", 0.0)

            for weight in weights:
                bucket = stats[weight]
                bucket["turns"] += 1
                if raw_top and same_action(raw_top[0], actual_action):
                    bucket["raw_match"] += 1
                if raw_top and isinstance(raw_top[2], (int, float)):
                    bucket["raw_risks"].append(float(raw_top[2]))

                if ess < min_ess or not triplets:
                    continue

                adjusted = adjust_candidates(triplets, weight=weight)
                winner = next(
                    (item for item in adjusted if item.adjusted_rank == 1),
                    None,
                )
                if winner is None:
                    continue

                bucket["enabled"] += 1
                if winner.model_rank != 1:
                    bucket["changed"] += 1
                if same_action(winner.action, actual_action):
                    bucket["adjusted_match"] += 1
                if isinstance(winner.response_risk, (int, float)):
                    bucket["adjusted_risks"].append(
                        float(winner.response_risk)
                    )

                if winner.model_rank != 1 and len(bucket["examples"]) < 12:
                    bucket["examples"].append(
                        {
                            "event": event_index,
                            "actual": actual_action or "Pass",
                            "raw": raw_top[0],
                            "raw_score": raw_top[1],
                            "raw_risk": raw_top[2],
                            "adjusted": winner.action,
                            "adjusted_model_rank": winner.model_rank,
                            "adjusted_score": winner.adjusted_score,
                            "adjusted_risk": winner.response_risk,
                            "ess": ess,
                        }
                    )

        # Apply the recorded action after evaluating the pre-action state.
        action_env = sorted(
            RealCard2EnvCard[c] for c in actual_action
        )
        if env.acting_player_position != player:
            raise ValueError(
                f"turn mismatch at event {event_index}: "
                f"env={env.acting_player_position}, replay={player}"
            )
        env.step(player, action=action_env, update=True)
        inference.observe(player, actual_action)

        if env.game_over:
            break

    return stats


def merge_stats(total, current):
    for weight, bucket in current.items():
        dest = total.setdefault(
            weight,
            {
                "turns": 0,
                "enabled": 0,
                "changed": 0,
                "raw_match": 0,
                "adjusted_match": 0,
                "raw_risks": [],
                "adjusted_risks": [],
                "examples": [],
            },
        )
        for key in (
            "turns",
            "enabled",
            "changed",
            "raw_match",
            "adjusted_match",
        ):
            dest[key] += bucket[key]
        dest["raw_risks"].extend(bucket["raw_risks"])
        dest["adjusted_risks"].extend(bucket["adjusted_risks"])
        if len(dest["examples"]) < 20:
            dest["examples"].extend(
                bucket["examples"][: 20 - len(dest["examples"])]
            )


def pct(n, d):
    return "-" if not d else f"{n / d:.1%}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replays", nargs="+", help="*.replay.json files")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.20, 0.30, 0.40],
        help="risk weights to compare",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--min-ess", type=float, default=0.40)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=3,
        help="Only risk-adjust this many top DouZero candidates.",
    )
    args = parser.parse_args()

    total = {}
    for path in args.replays:
        replay = parse_replay(path)
        print(f"REPLAY {replay['video']}")
        try:
            result = evaluate_one(
                replay,
                args.weights,
                sample_count=args.samples,
                min_ess=args.min_ess,
                max_candidates=max(1, args.max_candidates),
            )
        except Exception as exc:
            print(f"  SKIP: {exc}")
            continue
        merge_stats(total, result)

    print("\n=== RAW vs RISK-ADJUSTED ===")
    print(
        "注意：与录像实际出牌一致率只表示行为一致，不代表策略正确率或胜率。"
    )
    for weight in args.weights:
        b = total.get(weight)
        if not b:
            continue
        print(f"\nλ={weight:.2f}")
        print(f"  我的决策时点: {b['turns']}")
        print(f"  风险调整启用: {b['enabled']}")
        print(
            f"  改判: {b['changed']}/{b['enabled']} "
            f"({pct(b['changed'], b['enabled'])})"
        )
        print(
            f"  原建议与实际一致: {b['raw_match']}/{b['turns']} "
            f"({pct(b['raw_match'], b['turns'])})"
        )
        print(
            f"  调整建议与实际一致: {b['adjusted_match']}/{b['enabled']} "
            f"({pct(b['adjusted_match'], b['enabled'])})"
        )
        if b["raw_risks"]:
            print(
                f"  原建议平均可压: {mean(b['raw_risks']):.1%}"
            )
        if b["adjusted_risks"]:
            print(
                f"  调整建议平均可压: "
                f"{mean(b['adjusted_risks']):.1%}"
            )

        if b["examples"]:
            print("  改判样例:")
            for item in b["examples"][:8]:
                rr = (
                    "-"
                    if item["raw_risk"] is None
                    else f"{item['raw_risk']:.0%}"
                )
                ar = (
                    "-"
                    if item["adjusted_risk"] is None
                    else f"{item['adjusted_risk']:.0%}"
                )
                print(
                    f"    E{item['event']}: "
                    f"原 {item['raw']}({rr}) -> "
                    f"调 {item['adjusted']}({ar}, "
                    f"M{item['adjusted_model_rank']}, "
                    f"{item['adjusted_score']:.1f}); "
                    f"实际 {item['actual']}; "
                    f"ESS {item['ess']:.0%}"
                )


if __name__ == "__main__":
    main()
