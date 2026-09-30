"""Summarize raw-vs-risk-adjusted suggestion audits.

This tool evaluates *behavior of the ranking layer*, not playing strength.
It reports how often the adjustment changes the top candidate, how often each
top candidate matches the user's actual next play, and how response risk changes.

Usage:
    python tools/summarize_suggestion_audits.py
    python tools/summarize_suggestion_audits.py screenshots/inference_audits
"""

import argparse
import glob
import json
import os
from statistics import mean


def _next_actual_action(payload, snapshot):
    history = payload.get("public_history", [])
    index = snapshot.get("history_length")
    if not isinstance(index, int) or index < 0 or index >= len(history):
        return None

    event = history[index]
    if event.get("player") != payload.get("my_position"):
        return None
    return event.get("action")


def summarize(paths):
    games = 0
    turns = 0
    enabled_turns = 0
    changed_turns = 0
    comparable_turns = 0
    raw_matches = 0
    adjusted_matches = 0
    raw_risks = []
    adjusted_risks = []
    changed_examples = []

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception as exc:
            print(f"SKIP {path}: {exc}")
            continue

        audits = payload.get("suggestion_audit", [])
        if not audits:
            continue
        games += 1

        for snapshot in audits:
            turns += 1
            raw = snapshot.get("raw_top") or {}
            adjusted = snapshot.get("adjusted_top")
            enabled = bool(snapshot.get("risk_adjustment_enabled")) and adjusted

            raw_risk = raw.get("response_risk")
            if isinstance(raw_risk, (int, float)):
                raw_risks.append(float(raw_risk))

            if not enabled:
                continue

            enabled_turns += 1
            changed = bool(snapshot.get("changed_top_action"))
            if changed:
                changed_turns += 1

            adjusted_risk = adjusted.get("response_risk")
            if isinstance(adjusted_risk, (int, float)):
                adjusted_risks.append(float(adjusted_risk))

            actual = _next_actual_action(payload, snapshot)
            if actual is not None:
                comparable_turns += 1
                if raw.get("action") == actual:
                    raw_matches += 1
                if adjusted.get("action") == actual:
                    adjusted_matches += 1

            if changed and len(changed_examples) < 20:
                changed_examples.append(
                    {
                        "file": os.path.basename(path),
                        "history_length": snapshot.get("history_length"),
                        "raw": raw.get("action"),
                        "raw_model_score": raw.get("model_score"),
                        "raw_risk": raw_risk,
                        "adjusted": adjusted.get("action"),
                        "adjusted_model_rank": adjusted.get("model_rank"),
                        "adjusted_risk": adjusted_risk,
                        "adjusted_score": adjusted.get("adjusted_score"),
                        "actual": actual,
                        "ess_ratio": snapshot.get("ess_ratio"),
                    }
                )

    def pct(num, den):
        return "-" if not den else f"{num / den:.1%}"

    print(f"审计牌局: {games}")
    print(f"建议时点: {turns}")
    print(f"风险调整启用: {enabled_turns}")
    print(f"改判次数: {changed_turns} ({pct(changed_turns, enabled_turns)})")
    print()
    print("与实际下一手的一致率（仅作行为对照，不代表策略正确率）")
    print(f"  原DouZero: {raw_matches}/{comparable_turns} ({pct(raw_matches, comparable_turns)})")
    print(f"  风险调整:  {adjusted_matches}/{comparable_turns} ({pct(adjusted_matches, comparable_turns)})")
    print()
    if raw_risks:
        print(f"原建议平均敌方可压: {mean(raw_risks):.1%}")
    if adjusted_risks:
        print(f"调整建议平均敌方可压: {mean(adjusted_risks):.1%}")

    if changed_examples:
        print("\n改判样例（最多20条）")
        for item in changed_examples:
            rr = "-" if item["raw_risk"] is None else f"{item['raw_risk']:.0%}"
            ar = "-" if item["adjusted_risk"] is None else f"{item['adjusted_risk']:.0%}"
            print(
                f"  {item['file']} @H{item['history_length']}: "
                f"原={item['raw']}({rr}) -> 调={item['adjusted']}({ar}, "
                f"M{item['adjusted_model_rank']}, 调分{item['adjusted_score']:.1f}); "
                f"实际={item['actual']}; ESS={item['ess_ratio']:.0%}"
                if isinstance(item["ess_ratio"], (int, float))
                else
                f"  {item['file']} @H{item['history_length']}: "
                f"原={item['raw']}({rr}) -> 调={item['adjusted']}({ar}); "
                f"实际={item['actual']}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default=os.path.join("screenshots", "inference_audits"),
        help="audit directory or one JSON audit file",
    )
    args = parser.parse_args()

    if os.path.isdir(args.path):
        paths = sorted(glob.glob(os.path.join(args.path, "*.json")))
    else:
        paths = [args.path]

    if not paths:
        print("没有找到审计 JSON。")
        return

    summarize(paths)


if __name__ == "__main__":
    main()
