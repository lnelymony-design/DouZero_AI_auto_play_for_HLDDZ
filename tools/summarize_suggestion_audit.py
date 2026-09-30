"""Summarize belief-aware suggestion decisions from WeChat audit JSON files."""

import argparse
import glob
import json
import os
from collections import Counter


def display_action(action):
    if not action or action == "Pass":
        return "不出"
    return (
        str(action)
        .replace("D", "大王 ")
        .replace("X", "小王 ")
        .replace("T", "10 ")
        .strip()
    )


def load_files(paths):
    if paths:
        files = []
        for path in paths:
            files.extend(glob.glob(path))
        return sorted(set(files))
    return sorted(glob.glob("screenshots/inference_audits/*.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        help="audit json file(s) or glob patterns; defaults to inference_audits/*.json",
    )
    args = parser.parse_args()

    files = load_files(args.paths)
    if not files:
        print("没有找到推牌审计 JSON。")
        return

    source_counts = Counter()
    total = 0
    changed = 0
    changes = []

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception as exc:
            print(f"跳过无法读取的文件: {path} ({exc})")
            continue

        for item in payload.get("suggestion_audit", []):
            total += 1
            source = item.get("decision_source", "legacy")
            source_counts[source] += 1

            raw = (item.get("raw_top") or {}).get("action")
            final = (item.get("adjusted_top") or {}).get("action", raw)
            is_changed = bool(item.get("changed_top_action", final != raw))
            if is_changed:
                changed += 1
                changes.append(
                    {
                        "file": os.path.basename(path),
                        "time": item.get("timestamp", "-"),
                        "source": source,
                        "raw": raw,
                        "final": final,
                        "ess": item.get("ess_ratio"),
                        "reason": item.get("decision_reason", ""),
                    }
                )

    print(f"审计文件: {len(files)}")
    print(f"建议次数: {total}")
    print(f"改动首选: {changed}")
    if total:
        print(f"改动比例: {changed / total:.1%}")
    print()
    print("来源统计:")
    labels = {
        "douzero": "保持 DouZero",
        "env_override": "直接出完/路径",
        "belief_safer": "推牌风险修正",
        "legacy": "旧版记录",
    }
    for source, count in source_counts.most_common():
        print(f"  {labels.get(source, source)}: {count}")

    if changes:
        print()
        print("首选发生变化的决策:")
        for item in changes:
            ess = item["ess"]
            ess_text = "-" if ess is None else f"{float(ess):.0%}"
            print(
                f"  {item['time']} | {labels.get(item['source'], item['source'])} | "
                f"{display_action(item['raw'])} -> {display_action(item['final'])} | "
                f"ESS {ess_text} | {item['reason']}"
            )


if __name__ == "__main__":
    main()
