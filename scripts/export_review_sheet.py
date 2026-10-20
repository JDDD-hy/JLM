"""Export the merged candidate pool as a human-review sheet (CSV + JSONL); never changes labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.data.validate import read_jsonl

COLUMNS = [
    "review_no",
    "id",
    "quote",
    "assistant_label",
    "assistant_reason",
    "suggested_scene_code",
    "suggested_scene_name",
    "legacy_scene",
    "source_type",
    "extraction",
    "source_group",
    "source_url",
    "token_count",
    "over_128_tokens",
    "duplicate_group",
    "near_duplicate_of",
    "origin_flag",
    # human columns, left blank
    "human_quality_status",
    "human_scene_code",
    "license_status",
    "reviewer",
    "human_note",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="data/reviewed/pool-20261019/merged_candidates.jsonl")
    parser.add_argument("--duplicates", default="data/reviewed/pool-20261019/duplicate_review.jsonl")
    parser.add_argument("--scene-codes", default="raw_data/curated/scene_codes.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = read_jsonl(args.pool)
    if result.issues:
        raise ValueError("Pool failed validation")
    codes = json.loads(Path(args.scene_codes).read_text(encoding="utf-8-sig"))
    near = defaultdict(list)
    for line in Path(args.duplicates).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        if pair.get("kind") == "exact" or pair["left_id"] == pair["right_id"]:
            continue
        near[pair["left_id"]].append(f"{pair['right_id']}({pair['similarity']:.0f})")
        near[pair["right_id"]].append(f"{pair['left_id']}({pair['similarity']:.0f})")
    rows = []
    for index, item in enumerate(result.records, start=1):
        scene = item.get("assistant_suggested_scene")
        if scene is None and isinstance(item.get("scene"), int):
            scene = item["scene"]
        reason = item.get("assistant_reason") or ""
        origin_flag = "AI生成嫌疑" if ("aigc" in reason.lower() or "AI 生成" in reason or "AI生成" in reason) else ""
        rows.append(
            {
                "review_no": index,
                "id": item["id"],
                "quote": item["quote"],
                "assistant_label": item.get("assistant_suggested_label") or "legacy_unlabeled",
                "assistant_reason": reason,
                "suggested_scene_code": scene if scene is not None else "",
                "suggested_scene_name": codes.get(str(scene), "") if scene is not None else "",
                "legacy_scene": item.get("legacy_scene_label") or "",
                "source_type": item.get("source_type", ""),
                "extraction": item.get("extraction_method") or "",
                "source_group": item.get("source_group", ""),
                "source_url": item.get("source_url", ""),
                "token_count": item.get("token_count", ""),
                "over_128_tokens": item.get("over_128_tokens", ""),
                "duplicate_group": item.get("source_group", ""),
                "near_duplicate_of": "; ".join(near.get(item["id"], [])),
                "origin_flag": origin_flag,
                "human_quality_status": "",
                "human_scene_code": "",
                "license_status": item.get("license_status", "pending_review"),
                "reviewer": "",
                "human_note": "",
            }
        )
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "review_sheet.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (target / "review_sheet.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (target / "review_sheet.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    legend = {
        "rows": len(rows),
        "pool_sha256": result.sha256,
        "human_quality_status_values": [
            "genuine_jiahao_style",
            "possible_jiahao_style",
            "meta_discussion",
            "ordinary_text",
            "spam",
            "uncertain",
        ],
        "license_status_values": ["pending_review", "approved_for_training", "rejected"],
        "scene_codes": codes,
        "notes": [
            "assistant_* 列为助手建议，不是人工标签。",
            "human_quality_status=genuine_jiahao_style 且 license_status=approved_for_training 才进入训练。",
            "duplicate_group 为来源帖 ID；同帖及同模板段落建议作为一组审核，避免跨划分泄漏。",
            "near_duplicate_of 为 RapidFuzz 相似度 >=92 的其他记录。",
            "origin_flag 标出评论区或原帖自称 AI 生成的来源。",
        ],
    }
    (target / "README.json").write_text(json.dumps(legend, ensure_ascii=False, indent=2), encoding="utf-8")
    write_simple(rows, target)


SIMPLE_COLUMNS = ["序号", "判定", "语录", "场景", "场景名", "提示", "备注", "来源组", "原帖", "id"]


def write_simple(rows: list[dict], target: Path) -> None:
    """One decision column; scene prefilled; flags folded into a single hint."""
    simple = []
    for row in rows:
        flags = []
        if row["near_duplicate_of"]:
            flags.append("近重复")
        if row["origin_flag"]:
            flags.append("AI嫌疑")
        if row["over_128_tokens"] is True:
            flags.append("超长")
        if row["extraction"]:
            flags.append("长帖拆段")
        if "@" in row["quote"]:
            flags.append("含@用户名需删")
        hint = "；".join(flags + ([row["assistant_reason"][:60]] if row["assistant_reason"] else []))
        simple.append(
            {
                "序号": row["review_no"],
                "判定": "",
                "语录": row["quote"],
                "场景": row["suggested_scene_code"],
                "场景名": row["suggested_scene_name"],
                "提示": hint,
                "备注": "",
                "来源组": row["source_group"],
                "原帖": row["source_url"],
                "id": row["id"],
            }
        )
    with (target / "review_simple.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(simple)
    guide = {
        "怎么填": "只填「判定」列：留空 = 通过（进训练）；x = 不要；? = 拿不准。其余列都已预填。",
        "场景": "已预填助手建议的场景代码；只有觉得不对时才改成 1-17 的其他数字，空白表示未知。",
        "备注": "可选。",
        "通过的含义": "通过 = 你确认它是真实嘉豪风格语录，且学校许可覆盖其训练使用。",
        "排序建议": "按「来源组」排序后审，同一篇长帖拆出的段落会排在一起。",
    }
    (target / "README_simple.json").write_text(json.dumps(guide, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
