"""Draft assistant annotations for '嘉豪的一天' series paragraphs; output still requires human review."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SLOTS = [
    ("书包", "possible_jiahao_style", "别人/我书包对比，用专业物品和括号注解自我抬高，就很……模板句", None),
    ("备赛资料", "possible_jiahao_style", "别人/我备赛对比模板句，自我抬高", None),
    ("行李箱", "possible_jiahao_style", "别人/我行李箱对比，用旅行装备和括号注解自我抬高，就很……模板句", None),
    ("赶不上", "possible_jiahao_style", "差点误机时蹲在柜台算规则被当成特种兵护送，夸张自述", None),
    ("直呼我全名", "possible_jiahao_style", "没人敢直呼全名、教授/院长也要先问豪哥，身份特殊化到荒诞", None),
    ("什么是嘉豪", "uncertain", "仅说忙于专业没空刷朋友圈，脱离系列语境风格特征弱", None),
    ("求教", "possible_jiahao_style", "第一次追人反衬以前都是异性围上来请教专业问题，自我特殊化", 13),
    ("离谱的", "possible_jiahao_style", "失恋后在暴雨中背专业内容数小时被当成疯子，夸张自述", 13),
    ("炸裂", "possible_jiahao_style", "回答问题讲二十分钟老师不敢打断，自我特殊化叙事", None),
    ("追人", "possible_jiahao_style", "第一次追人反衬以前都是异性围上来请教专业问题，自我特殊化", 13),
    ("打政策辩", "possible_jiahao_style", "第一次打政策辩反衬以前都是队伍抢着要我，自我特殊化", None),
    ("失恋", "possible_jiahao_style", "失恋后在暴雨中背专业内容数小时被当成疯子，现实与预期错位的夸张自述", 13),
    ("亏损", "possible_jiahao_style", "受挫后通宵复盘被当成理财疯子，夸张自述", None),
    ("输比赛", "possible_jiahao_style", "输比赛后对镜背稿六小时，夸张自述", None),
    ("同龄", "possible_jiahao_style", "同龄人/我对比，把娱乐替换为专业术语罗列，就很……模板句", None),
    ("朋友圈", "possible_jiahao_style", "别人/我朋友圈对比，以专业术语和凌晨配图自我抬高，模板句", None),
    ("半夜emo", "possible_jiahao_style", "朋友求助被从专业讲到凌晨四点从此绕路走，自我特殊化叙事", None),
    ("室友", "possible_jiahao_style", "室友求助被从专业讲到凌晨四点从此绕路走，自我特殊化叙事", None),
    ("谈恋爱", "possible_jiahao_style", "别人/我恋爱对比，把恋爱术语化重述，模板句", 13),
    ("别人失恋", "possible_jiahao_style", "别人/我失恋对比，把失恋处理成论文或答辩，模板句", 13),
    ("做pre", "possible_jiahao_style", "别人目录页我已定学术化标题，老师问是否要统治学科，自我特殊化叙事", None),
    ("收不住", "possible_jiahao_style", "回答问题讲二十分钟老师不敢打断，自我特殊化叙事", None),
    ("emo崩溃", "possible_jiahao_style", "受挫后暴雨中推导方程数小时晕倒被送医，夸张自述", 13),
    ("发问", "possible_jiahao_style", "第一次追人反衬以前都是学弟学妹围上来请教，自我特殊化", 13),
    ("没收住", "possible_jiahao_style", "回答问题讲二十分钟老师不敢打断，自我特殊化叙事", None),
]
SERIES_CAVEAT = "嘉豪的一天系列模板，同模板跨帖变体需人工分组防泄漏"


def draft(item: dict, scenes: dict[str, int], note: str) -> dict:
    head = item["quote"][:40]
    label, reason, scene_override = "uncertain", "系列模板之外的段落，需人工判断", None
    for key, slot_label, slot_reason, slot_scene in SLOTS:
        if key in head:
            label, reason, scene_override = slot_label, slot_reason, slot_scene
            break
    scene = scene_override or scenes[item["source_parent_id"]]
    full = f"{reason}；{SERIES_CAVEAT}" + (f"；{note}" if note else "")
    return {"id": item["id"], "label": label, "reason": full, "scene": scene}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True, help="Output of split_selected_paragraphs.py")
    parser.add_argument("--parent-scenes", required=True, help="JSON mapping parent record id -> scene int")
    parser.add_argument("--note", default="", help="Extra provenance caveat appended to every reason")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    scenes = json.loads(args.parent_scenes)
    rows = [
        draft(json.loads(line), scenes, args.note)
        for line in Path(args.selected).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    Path(args.output).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"annotated": len(rows), "labels": dict(Counter(r["label"] for r in rows))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
