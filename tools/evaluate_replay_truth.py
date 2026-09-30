import argparse
import json
from statistics import mean

from constants import RealCard2EnvCard
from inference import HandInferenceEngine, adjust_candidates, select_safer_alternative
from inference.legality import can_beat
from tools.evaluate_replay_suggestions import (
    POSITIONS, init_env, parse_replay, player_for_side,
)


def load_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["recordings"]


def current_cards(events, index, side, final_cards):
    cards = list(final_cards or "")
    for event_side, action in events[index:]:
        if event_side == side and action:
            cards.extend(action)
    return cards


def enemy_sides(position_code):
    mine = POSITIONS[position_code]
    out = []
    for side in ("left", "right"):
        pos = player_for_side(position_code, side)
        if (mine == "landlord" and pos != "landlord") or (
            mine != "landlord" and pos == "landlord"
        ):
            out.append(side)
    return out


def actual_response(events, index, code, final_hands, action):
    if not action or action == "Pass":
        return None
    move = sorted(RealCard2EnvCard[c] for c in action)
    for side in enemy_sides(code):
        hand = current_cards(events, index, side, final_hands.get(side, ""))
        hand_env = sorted(RealCard2EnvCard[c] for c in hand)
        if can_beat(hand_env, move):
            return True
    return False


def actual_response_for_side(events, index, side, final_hands, action):
    if not action or action == "Pass":
        return None
    move = sorted(RealCard2EnvCard[c] for c in action)
    hand = current_cards(events, index, side, final_hands.get(side, ""))
    hand_env = sorted(RealCard2EnvCard[c] for c in hand)
    return can_beat(hand_env, move)


def evaluate(
    replay, truth, weights, samples, max_candidates, behavior_mode,
    safer_risk_gains, safer_gap_fractions,
):
    code = replay["position_code"]
    mine = POSITIONS[code]
    events = replay["events"]
    env = init_env(code, replay["my_hand"], replay["bottom_cards"])
    behavior_kwargs = {}
    if behavior_mode == "uniform":
        behavior_kwargs = {
            "pass_penalty": 1.0,
            "friendly_pass_penalty": 1.0,
            "play_behavior_strength": 0.0,
        }
    elif behavior_mode == "pass":
        behavior_kwargs = {
            "pass_penalty": 0.62,
            "friendly_pass_penalty": 0.86,
            "play_behavior_strength": 0.0,
        }
    inf = HandInferenceEngine(
        my_position=mine,
        my_hand_cards=replay["my_hand"],
        three_landlord_cards=replay["bottom_cards"],
        sample_count=samples,
        **behavior_kwargs,
    )
    candidates = []
    side_candidates = []
    decisions = []

    for idx, (side, actual) in enumerate(events):
        player = player_for_side(code, side)
        if player == mine:
            inf_result = inf.infer()
            _, actions = env.step(mine, action=None, update=False)
            triplets = []
            truth_map = {}
            profile_map = {}
            for action, score in actions[:max_candidates]:
                profile = (
                    None
                    if action == "Pass"
                    else inf.response_profile(action)
                )
                risk = None if profile is None else profile["can_beat"]
                truth_value = actual_response(
                    events, idx, code, truth["final_hands"], action
                )
                triplets.append((action, score, risk))
                profile_map[action] = profile
                truth_map[action] = truth_value
                if risk is not None:
                    candidates.append((float(risk), bool(truth_value)))
                if profile is not None:
                    for physical_side in ("left", "right"):
                        logical_player = player_for_side(
                            code, physical_side
                        )
                        side_profile = (
                            profile.get("players", {})
                            .get(logical_player)
                        )
                        if side_profile is None:
                            continue
                        side_truth = actual_response_for_side(
                            events,
                            idx,
                            physical_side,
                            truth["final_hands"],
                            action,
                        )
                        is_enemy = (
                            (mine == "landlord"
                             and logical_player != "landlord")
                            or
                            (mine != "landlord"
                             and logical_player == "landlord")
                        )
                        side_candidates.append(
                            (
                                physical_side,
                                logical_player,
                                "enemy" if is_enemy else "teammate",
                                float(side_profile["can_beat"]),
                                bool(side_truth),
                            )
                        )

            raw = triplets[0] if triplets else None
            row = {
                "raw_truth": None if raw is None else truth_map.get(raw[0]),
                "raw_pass": bool(raw and raw[0] == "Pass"),
                "ess": inf_result.get("effective_sample_ratio", 0.0),
                "adj": {},
                "triplets": [
                    {
                        "action": action,
                        "score": float(score),
                        "risk": risk,
                        "pressure": (
                            None
                            if profile_map.get(action) is None
                            else profile_map[action]["pressure"]
                        ),
                        "ordinary_beat": (
                            None
                            if profile_map.get(action) is None
                            else profile_map[action]["ordinary_beat"]
                        ),
                        "bomb_only": (
                            None
                            if profile_map.get(action) is None
                            else profile_map[action]["bomb_only"]
                        ),
                        "truth": truth_map.get(action),
                        "rank": rank,
                    }
                    for rank, (action, score, risk)
                    in enumerate(triplets, start=1)
                ],
            }
            for weight in weights:
                ranked = adjust_candidates(triplets, weight=weight)
                top = next(
                    (x for x in ranked if x.adjusted_rank == 1), None
                )
                row["adj"][weight] = None if top is None else {
                    "truth": truth_map.get(top.action),
                    "pass": top.action == "Pass",
                    "changed": top.model_rank != 1,
                }

            row["safer"] = {}
            for risk_gain in safer_risk_gains:
                for gap_fraction in safer_gap_fractions:
                    key = (risk_gain, gap_fraction)
                    alt = select_safer_alternative(
                        triplets,
                        min_risk_gain=risk_gain,
                        max_model_gap_fraction=gap_fraction,
                        max_model_rank=max_candidates,
                    )
                    row["safer"][key] = None if alt is None else {
                        "truth": truth_map.get(alt.action),
                        "raw_truth": truth_map.get(alt.raw_action),
                        "risk_gain": alt.risk_gain,
                        "model_gap": alt.model_gap,
                        "model_gap_fraction": alt.model_gap_fraction,
                        "model_rank": alt.model_rank,
                    }
            decisions.append(row)

        env.step(
            player,
            action=sorted(RealCard2EnvCard[c] for c in actual),
            update=True,
        )
        inf.observe(player, actual)
        if env.game_over:
            break

    return candidates, side_candidates, decisions


def rate(items):
    return None if not items else sum(items) / len(items)


def pct(value):
    return "-" if value is None else f"{value:.1%}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("replays", nargs="+")
    p.add_argument(
        "--truth-file", default="tools/wechat_recording_truth.json"
    )
    p.add_argument(
        "--weights", nargs="+", type=float,
        default=[0.42, 0.46, 0.50, 0.52, 0.54],
    )
    p.add_argument("--samples", type=int, default=1600)
    p.add_argument("--max-candidates", type=int, default=3)
    p.add_argument(
        "--safer-risk-gains", nargs="+", type=float,
        default=[0.15, 0.20, 0.25, 0.30],
    )
    p.add_argument(
        "--safer-gap-fractions", nargs="+", type=float,
        default=[0.20, 0.35, 0.50],
    )
    p.add_argument(
        "--behavior-mode",
        choices=["uniform", "pass", "full"],
        default="full",
    )
    args = p.parse_args()

    truths = load_truth(args.truth_file)
    all_candidates = []
    all_side_candidates = []
    all_decisions = []
    for path in args.replays:
        replay = parse_replay(path)
        truth = truths.get(replay["video"])
        if not truth:
            continue
        cand, side_cand, dec = evaluate(
            replay, truth, args.weights, args.samples,
            max(1, args.max_candidates), args.behavior_mode,
            args.safer_risk_gains, args.safer_gap_fractions,
        )
        all_candidates += cand
        all_side_candidates += side_cand
        all_decisions += dec
        print(
            f"REPLAY {replay['video']}: "
            f"decisions={len(dec)} candidates={len(cand)}"
        )

    if not all_candidates:
        print("NO DATA")
        return

    brier = mean((pred - float(truth)) ** 2 for pred, truth in all_candidates)
    predicted = mean(pred for pred, _ in all_candidates)
    observed = mean(float(truth) for _, truth in all_candidates)
    print(f"\n=== RESPONSE RISK TRUTH CHECK [{args.behavior_mode}] ===")
    print(f"candidate_n={len(all_candidates)}")
    print(f"Brier={brier:.4f}")
    print(f"predicted_mean={predicted:.1%}")
    print(f"actual_beatable={observed:.1%}")

    for lo, hi in ((0,.25),(.25,.5),(.5,.75),(.75,1.0001)):
        rows=[x for x in all_candidates if lo <= x[0] < hi]
        if rows:
            print(
                f"bin {lo:.0%}-{min(1,hi):.0%}: n={len(rows)} "
                f"pred={mean(x[0] for x in rows):.1%} "
                f"actual={mean(float(x[1]) for x in rows):.1%}"
            )

    def print_calibration(label, rows):
        if not rows:
            print(f"{label}: no data")
            return
        score = mean(
            (pred - float(truth)) ** 2
            for pred, truth in rows
        )
        print(
            f"{label}: n={len(rows)} "
            f"Brier={score:.4f} "
            f"pred={mean(pred for pred, _ in rows):.1%} "
            f"actual={mean(float(truth) for _, truth in rows):.1%}"
        )

    print("\n=== PHYSICAL-SIDE RESPONSE CALIBRATION ===")
    for physical_side in ("left", "right"):
        rows = [
            (pred, truth)
            for side, logical, relation, pred, truth
            in all_side_candidates
            if side == physical_side
        ]
        print_calibration(physical_side, rows)

    print("\n=== LOGICAL-POSITION RESPONSE CALIBRATION ===")
    for logical_player in POSITIONS:
        rows = [
            (pred, truth)
            for side, logical, relation, pred, truth
            in all_side_candidates
            if logical == logical_player
        ]
        print_calibration(logical_player, rows)

    print("\n=== RELATION RESPONSE CALIBRATION ===")
    for relation_name in ("enemy", "teammate"):
        rows = [
            (pred, truth)
            for side, logical, relation, pred, truth
            in all_side_candidates
            if relation == relation_name
        ]
        print_calibration(relation_name, rows)

    raw = [
        float(row["raw_truth"]) for row in all_decisions
        if row["raw_truth"] is not None
    ]
    raw_pass = sum(row["raw_pass"] for row in all_decisions)
    print("\n=== TOP ACTION TRUE BEATABILITY ===")
    print(
        f"raw: nonpass={len(raw)} pass={raw_pass} "
        f"beatable={pct(rate(raw))}"
    )
    for weight in args.weights:
        choices = [
            row["adj"].get(weight) for row in all_decisions
            if row["adj"].get(weight) is not None
        ]
        vals = [
            float(x["truth"]) for x in choices
            if x["truth"] is not None
        ]
        passes = sum(x["pass"] for x in choices)
        changed = sum(x["changed"] for x in choices)
        print(
            f"lambda={weight:.2f}: changed={changed}/{len(choices)} "
            f"nonpass={len(vals)} pass={passes} "
            f"beatable={pct(rate(vals))}"
        )

    print("\n=== TOP-CANDIDATE RISK SEPARATION ===")
    opportunities = []
    comparable = 0
    gains = []
    for decision_index, row in enumerate(all_decisions):
        items = row.get("triplets", [])
        if not items or items[0].get("risk") is None:
            continue
        raw_item = items[0]
        numeric = [
            x for x in items[1:]
            if x.get("risk") is not None
        ]
        if not numeric:
            continue
        comparable += 1
        best = min(numeric, key=lambda x: x["risk"])
        gain = raw_item["risk"] - best["risk"]
        gains.append(gain)

        scores = [x["score"] for x in items]
        span = max(scores) - min(scores) if scores else 0.0
        gap = raw_item["score"] - best["score"]
        gap_fraction = 0.0 if span <= 1e-12 else gap / span
        opportunities.append({
            "decision": decision_index,
            "raw": raw_item,
            "alt": best,
            "gain": gain,
            "gap_fraction": gap_fraction,
        })

    if gains:
        print(f"comparable decisions={comparable}")
        print(
            f"risk gain mean={mean(gains):.1%} "
            f"max={max(gains):.1%}"
        )
        for threshold in (0.01, 0.03, 0.05, 0.10, 0.15):
            count = sum(g >= threshold for g in gains)
            print(
                f"gain >= {threshold:.0%}: "
                f"{count}/{len(gains)}"
            )
        print("largest candidate separations:")
        for item in sorted(
            opportunities, key=lambda x: x["gain"], reverse=True
        )[:10]:
            raw_item = item["raw"]
            alt_item = item["alt"]
            print(
                f"  D{item['decision']}: "
                f"raw {raw_item['action']} risk={raw_item['risk']:.1%} "
                f"truth={raw_item['truth']} score={raw_item['score']:.3f} -> "
                f"M{alt_item['rank']} {alt_item['action']} "
                f"risk={alt_item['risk']:.1%} truth={alt_item['truth']} "
                f"score={alt_item['score']:.3f}; "
                f"gain={item['gain']:.1%}, "
                f"gap={item['gap_fraction']:.1%}"
            )
    else:
        print("no comparable non-pass top decisions")

    print("\n=== RESPONSE PRESSURE SEPARATION ===")
    pressure_gains = []
    pressure_rows = []
    for decision_index, row in enumerate(all_decisions):
        items = row.get("triplets", [])
        if not items or items[0].get("pressure") is None:
            continue
        raw_item = items[0]
        numeric = [
            x for x in items[1:]
            if x.get("pressure") is not None
        ]
        if not numeric:
            continue
        best = min(numeric, key=lambda x: x["pressure"])
        gain = raw_item["pressure"] - best["pressure"]
        pressure_gains.append(gain)

        scores = [x["score"] for x in items]
        span = max(scores) - min(scores) if scores else 0.0
        model_gap = raw_item["score"] - best["score"]
        gap_fraction = (
            0.0 if span <= 1e-12 else model_gap / span
        )
        pressure_rows.append({
            "decision": decision_index,
            "raw": raw_item,
            "alt": best,
            "gain": gain,
            "gap_fraction": gap_fraction,
        })

    if pressure_gains:
        print(
            f"pressure gain mean={mean(pressure_gains):.1%} "
            f"max={max(pressure_gains):.1%}"
        )
        for threshold in (0.01, 0.03, 0.05, 0.10, 0.15):
            count = sum(g >= threshold for g in pressure_gains)
            print(
                f"pressure gain >= {threshold:.0%}: "
                f"{count}/{len(pressure_gains)}"
            )
        print("largest pressure separations:")
        for item in sorted(
            pressure_rows, key=lambda x: x["gain"], reverse=True
        )[:10]:
            raw_item = item["raw"]
            alt_item = item["alt"]
            print(
                f"  D{item['decision']}: "
                f"raw {raw_item['action']} "
                f"pressure={raw_item['pressure']:.1%} "
                f"(ordinary={raw_item['ordinary_beat']:.1%}, "
                f"bomb={raw_item['bomb_only']:.1%}) "
                f"truth={raw_item['truth']} -> "
                f"M{alt_item['rank']} {alt_item['action']} "
                f"pressure={alt_item['pressure']:.1%} "
                f"(ordinary={alt_item['ordinary_beat']:.1%}, "
                f"bomb={alt_item['bomb_only']:.1%}) "
                f"truth={alt_item['truth']}; "
                f"gain={item['gain']:.1%}, "
                f"gap={item['gap_fraction']:.1%}"
            )

    print("\n=== SAFER ALTERNATIVE TRUTH GRID ===")
    print(
        "offered = 有多少决策点会显示更稳备选；"
        "raw/alt beatable = 这些同一决策点上原建议/备选的真实可压率。"
    )
    for risk_gain in args.safer_risk_gains:
        for gap_fraction in args.safer_gap_fractions:
            key = (risk_gain, gap_fraction)
            rows = [
                row["safer"].get(key) for row in all_decisions
                if row.get("safer", {}).get(key) is not None
            ]
            raw_vals = [
                float(x["raw_truth"]) for x in rows
                if x["raw_truth"] is not None
            ]
            alt_vals = [
                float(x["truth"]) for x in rows
                if x["truth"] is not None
            ]
            avg_gain = (
                None if not rows
                else mean(x["risk_gain"] for x in rows)
            )
            avg_gap = (
                None if not rows
                else mean(x["model_gap_fraction"] for x in rows)
            )
            print(
                f"risk>={risk_gain:.0%} gap<={gap_fraction:.0%}: "
                f"offered={len(rows)}/{len(all_decisions)} "
                f"raw={pct(rate(raw_vals))} "
                f"alt={pct(rate(alt_vals))} "
                f"pred_gain={pct(avg_gain)} "
                f"avg_gap={pct(avg_gap)}"
            )


if __name__ == "__main__":
    main()
