"""Evaluate and tune hidden-hand inference from saved WeChat round audits.

Each live round audit contains:
- the user's initial hand and public action history,
- the final inference snapshot,
- a delayed settlement screenshot.

At settlement, WeChat reveals the remaining hands.  This tool re-reads those
revealed cards, scores the saved posterior, and can replay the same public
history across a small parameter grid.

Examples:
    python tools/evaluate_inference_audits.py
    python tools/evaluate_inference_audits.py --tune
    python tools/evaluate_inference_audits.py --tune --sample-count 600
"""
import argparse
import glob
import json
import math
import os
from collections import Counter
from itertools import product

import cv2

from config import Config
from helpers.WechatCardRecognizer import WechatCardRecognizer
from inference.engine import CARD_ORDER, HandInferenceEngine


SIDE_POSITION_MAP = {
    "landlord": {"left": "landlord_up", "right": "landlord_down"},
    "landlord_up": {"left": "landlord_down", "right": "landlord"},
    "landlord_down": {"left": "landlord", "right": "landlord_up"},
}


def _exact_count_probability(player_data, rank, actual_count):
    stats = player_data.get("cards", {}).get(rank, {})
    p1 = float(stats.get("one_plus", 0.0))
    p2 = float(stats.get("pair_plus", 0.0))
    p3 = float(stats.get("triple_plus", 0.0))
    p4 = float(stats.get("bomb", 0.0))

    if actual_count <= 0:
        return max(0.0, 1.0 - p1)
    if actual_count == 1:
        return max(0.0, p1 - p2)
    if actual_count == 2:
        return max(0.0, p2 - p3)
    if actual_count == 3:
        return max(0.0, p3 - p4)
    return max(0.0, p4)


def score_player(player_data, actual_hand):
    counts = Counter(actual_hand)
    brier_terms = []
    nll_terms = []

    for rank in CARD_ORDER:
        stats = player_data.get("cards", {}).get(rank, {})
        presence_p = float(stats.get("one_plus", 0.0))
        truth = 1.0 if counts[rank] > 0 else 0.0
        brier_terms.append((presence_p - truth) ** 2)

        exact_p = _exact_count_probability(
            player_data,
            rank,
            counts[rank],
        )
        nll_terms.append(-math.log(max(1e-6, exact_p)))

    return {
        "brier": sum(brier_terms) / len(brier_terms),
        "nll": sum(nll_terms) / len(nll_terms),
    }


def load_audits(audit_dir):
    recognizer = WechatCardRecognizer()
    audits = []

    for path in sorted(glob.glob(os.path.join(audit_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception as exc:
            print(f"跳过 {path}: JSON读取失败: {exc}")
            continue

        # v2 stores a burst of settlement frames; keep v1 compatibility.
        image_names = payload.get("screenshots") or []
        if not image_names and payload.get("screenshot"):
            image_names = [payload["screenshot"]]
        if not image_names:
            print(f"跳过 {path}: 没有结算截图")
            continue

        images = []
        for image_name in image_names:
            image_path = os.path.join(audit_dir, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                images.append((image_name, image))
        if not images:
            print(f"跳过 {path}: 所有结算截图都无法读取")
            continue

        my_position = payload.get("my_position")
        side_map = SIDE_POSITION_MAP.get(my_position, {})
        tracked = payload.get("tracked_remaining", {})
        actual_hands = {}
        actual_sources = {}

        for side in ("left", "right"):
            position = side_map.get(side)
            if not position:
                continue

            expected_count = tracked.get(side)
            if expected_count is None:
                continue
            expected_count = int(expected_count)

            if expected_count == 0:
                actual_hands[position] = ""
                actual_sources[position] = "winner_zero"
                continue

            candidates = []
            for image_name, image in images:
                if side == "left":
                    cards = recognizer.recognize_left_played(
                        image,
                        expected_count=expected_count,
                    )
                else:
                    cards = recognizer.recognize_right_played(
                        image,
                        expected_count=expected_count,
                    )
                if len(cards) == expected_count:
                    candidates.append((cards, image_name))

            if not candidates:
                print(
                    f"{os.path.basename(path)}: {side}在"
                    f"{len(images)}张结算截图中都未完整识别"
                    f"{expected_count}张，暂不用于评分"
                )
                continue

            # If several burst frames work, use the most frequently repeated
            # recognition.  This rejects one-frame animation/OCR glitches.
            frequency = Counter(cards for cards, _ in candidates)
            best_cards, _ = frequency.most_common(1)[0]
            source_name = next(
                name for cards, name in candidates if cards == best_cards
            )
            actual_hands[position] = best_cards
            actual_sources[position] = source_name

        payload["_path"] = path
        payload["_actual_hands"] = actual_hands
        payload["_actual_sources"] = actual_sources
        audits.append(payload)

    return audits


def score_snapshot(audit):
    inference = audit.get("inference") or {}
    players = inference.get("players") or {}
    scores = []

    for position, actual_hand in audit.get("_actual_hands", {}).items():
        player_data = players.get(position)
        if not player_data:
            continue
        score = score_player(player_data, actual_hand)
        score["position"] = position
        score["actual_hand"] = actual_hand
        scores.append(score)

    return scores


def aggregate(scores):
    if not scores:
        return None
    return {
        "brier": sum(item["brier"] for item in scores) / len(scores),
        "nll": sum(item["nll"] for item in scores) / len(scores),
        "sides": len(scores),
    }


def replay_audit(audit, params, sample_count):
    engine = HandInferenceEngine(
        my_position=audit["my_position"],
        my_hand_cards=audit["initial_my_hand"],
        three_landlord_cards=audit.get("three_landlord_cards", ""),
        sample_count=sample_count,
        pass_penalty=params["pass_penalty"],
        friendly_pass_penalty=params["friendly_pass_penalty"],
        play_behavior_strength=params["play_behavior_strength"],
        behavior_temperature=params["behavior_temperature"],
        min_effective_sample_ratio=params["min_effective_sample_ratio"],
    )

    for item in audit.get("public_history", []):
        action = item.get("action", "")
        if action == "Pass":
            action = ""
        engine.observe(item["player"], action)

    return engine.infer()


def score_params(audits, params, sample_count):
    scores = []
    for audit in audits:
        if not audit.get("_actual_hands"):
            continue
        result = replay_audit(audit, params, sample_count)
        players = result.get("players", {})
        for position, actual_hand in audit["_actual_hands"].items():
            if position in players:
                scores.append(
                    score_player(players[position], actual_hand)
                )
    return aggregate(scores)


def current_params():
    config = Config.load()
    return {
        "pass_penalty": config.inference_pass_penalty,
        "friendly_pass_penalty": config.inference_friendly_pass_penalty,
        "play_behavior_strength": config.inference_play_behavior_strength,
        "behavior_temperature": config.inference_behavior_temperature,
        "min_effective_sample_ratio": config.inference_min_effective_sample_ratio,
    }


def tune(audits, sample_count):
    base = current_params()
    grids = {
        "pass_penalty": [0.52, 0.62, 0.72],
        "friendly_pass_penalty": [0.82, 0.90],
        "play_behavior_strength": [0.6, 1.0, 1.4],
        "behavior_temperature": [0.55, 0.75, 0.95],
        "min_effective_sample_ratio": [0.24, 0.32],
    }

    results = []
    keys = list(grids)
    for values in product(*(grids[key] for key in keys)):
        params = dict(zip(keys, values))
        if params["friendly_pass_penalty"] < params["pass_penalty"]:
            continue
        score = score_params(audits, params, sample_count)
        if score is None:
            continue
        results.append((score["brier"], score["nll"], params, score))

    if not results:
        print("没有足够的结算真值用于调参。")
        return

    results.sort(key=lambda item: (item[0], item[1]))
    best = results[0]

    print("\n=== 当前配置 ===")
    print(json.dumps(base, ensure_ascii=False, indent=2))
    current_score = score_params(audits, base, sample_count)
    print("当前评分:", current_score)

    print("\n=== 网格搜索最佳（按Brier，其次NLL） ===")
    print(json.dumps(best[2], ensure_ascii=False, indent=2))
    print("最佳评分:", best[3])

    print("\n前5组:")
    for index, (_, _, params, score) in enumerate(results[:5], start=1):
        print(
            f"{index}. Brier={score['brier']:.4f} "
            f"NLL={score['nll']:.4f} "
            f"params={params}"
        )

    print(
        "\n说明：这只是基于已收集牌局的离线校准，"
        "不会自动改写config.json；样本少时不要过度拟合。"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-dir",
        default=os.path.join("screenshots", "inference_audits"),
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--sample-count", type=int, default=600)
    args = parser.parse_args()

    audits = load_audits(args.audit_dir)
    if not audits:
        print(f"没有找到可用审计：{args.audit_dir}")
        return

    all_scores = []
    print(f"读取到 {len(audits)} 个审计文件。")
    for audit in audits:
        scores = score_snapshot(audit)
        if not scores:
            continue
        print(f"\n{os.path.basename(audit['_path'])}")
        for item in scores:
            print(
                f"  {item['position']}: "
                f"真实={item['actual_hand'] or '空'} "
                f"Brier={item['brier']:.4f} "
                f"NLL={item['nll']:.4f}"
            )
        all_scores.extend(scores)

    overall = aggregate(all_scores)
    print("\n当前保存后验总体评分:", overall)

    if args.tune:
        tune(audits, max(200, args.sample_count))


if __name__ == "__main__":
    main()
