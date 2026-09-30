import argparse
import json
from statistics import mean

from constants import RealCard2EnvCard
from inference import HandInferenceEngine, adjust_candidates
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


def evaluate(
    replay, truth, weights, samples, max_candidates, behavior_mode
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
    decisions = []

    for idx, (side, actual) in enumerate(events):
        player = player_for_side(code, side)
        if player == mine:
            inf_result = inf.infer()
            _, actions = env.step(mine, action=None, update=False)
            triplets = []
            truth_map = {}
            for action, score in actions[:max_candidates]:
                risk = None if action == "Pass" else inf.response_risk(action)
                truth_value = actual_response(
                    events, idx, code, truth["final_hands"], action
                )
                triplets.append((action, score, risk))
                truth_map[action] = truth_value
                if risk is not None:
                    candidates.append((float(risk), bool(truth_value)))

            raw = triplets[0] if triplets else None
            row = {
                "raw_truth": None if raw is None else truth_map.get(raw[0]),
                "raw_pass": bool(raw and raw[0] == "Pass"),
                "ess": inf_result.get("effective_sample_ratio", 0.0),
                "adj": {},
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
            decisions.append(row)

        env.step(
            player,
            action=sorted(RealCard2EnvCard[c] for c in actual),
            update=True,
        )
        inf.observe(player, actual)
        if env.game_over:
            break

    return candidates, decisions


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
        "--behavior-mode",
        choices=["uniform", "pass", "full"],
        default="full",
    )
    args = p.parse_args()

    truths = load_truth(args.truth_file)
    all_candidates = []
    all_decisions = []
    for path in args.replays:
        replay = parse_replay(path)
        truth = truths.get(replay["video"])
        if not truth:
            continue
        cand, dec = evaluate(
            replay, truth, args.weights, args.samples,
            max(1, args.max_candidates), args.behavior_mode,
        )
        all_candidates += cand
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


if __name__ == "__main__":
    main()
