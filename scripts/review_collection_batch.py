"""Materialize this batch's assistant suggestions, never human labels or permissions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.data.deduplicate import exact_deduplicate, near_duplicate_candidates
from src.data.validate import stable_record_id, write_jsonl
from src.evaluation.novelty import novelty_dicts

# Explicit semantic review of the actual 2026-10-19 bounded sample.
SPECIAL = {
    "jlm_331d91386cdffae0": ("spam", "抽奖/引导互动，不是语录"),
    "jlm_2a7a521e1d4f2968": ("spam", "正文只有话题标签，没有可用语录"),
    "jlm_b939c95f687f3bd9": ("ordinary_text", "古文名句拼接，不能据自我形象词汇当成嘉豪原创语录"),
    "jlm_2e25c1a4332d1b60": ("ordinary_text", "现成诗词引用，不是本次目标的独立互联网表达"),
    "jlm_6a69bc4f1cd37437": ("uncertain", "有反讽结构，但依赖典故和上下文，暂不收为短语录"),
    "jlm_943d468655349382": ("ordinary_text", "游戏主页求印象和自述，正文不是标题所称语录"),
    "jlm_0d8a9eabb29ed234": ("uncertain", "短双关，需要图片/上下文；不能仅凭物理一词判断风格"),
    "jlm_f2efc84914744fce": ("possible_jiahao_style", "将手的数量与分数线性对应，带有自我夸张；抽取首句供人工审核"),
}
ORDINARY = {
    "jlm_dc0ffc5ceb1159a3",
    "jlm_30a8bf45513b26ce",
    "jlm_9888350e318e81d0",
    "jlm_bee79c1b87158bfb",
    "jlm_db1ebcc2fc562a04",
    "jlm_b61039b668e5ecb9",
    "jlm_ed95aeb753ed6549",
    "jlm_74cb62a0fb4736a3",
    "jlm_5c58c596e6d20586",
}


def main() -> None:
    records = []
    batches = [Path(f"data/raw/live-20261019-batch0{i}") for i in (1, 2)]
    for batch in batches:
        with (batch / "extracted.jsonl").open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    unique, exact_pairs = exact_deduplicate(records)
    audit = []
    for record in unique:
        label, reason = SPECIAL.get(
            record["id"],
            (
                "ordinary_text" if record["id"] in ORDINARY else "meta_discussion",
                "普通评论/个人叙述或评价，缺少独立的目标语录结构"
                if record["id"] in ORDINARY
                else "主要讨论/评价嘉豪或古文梗，不是独立嘉豪语录",
            ),
        )
        audit.append(
            {
                "id": record["id"],
                "suggested_label": label,
                "reason": reason,
                "reviewer_type": "assistant",
                "requires_human_review": True,
                "quote": record["quote"],
                "source_url": record["source_url"],
            }
        )
    destination = Path("data/reviewed/live-20261019")
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(audit, destination / "assistant_triage.jsonl")
    source = next(record for record in unique if record["id"] == "jlm_f2efc84914744fce")
    quote = source["quote"].split("[笑哭R]", 1)[0]
    candidate = {
        **source,
        "id": stable_record_id(quote, None),
        "quote": quote,
        "parent_record_id": source["id"],
        "excerpt_start": 0,
        "excerpt_end": len(quote),
        "assistant_suggested_label": "possible_jiahao_style",
        "quality_status": "unreviewed",
        "reviewer_type": None,
        "reviewer": None,
        "license_evidence": None,
    }
    write_jsonl([candidate], destination / "candidates.jsonl")
    write_jsonl(
        [
            {
                "id": candidate["id"],
                "quote": quote,
                "source_url": candidate["source_url"],
                "quality_status": "uncertain",
                "license_status": "pending_review",
                "reviewer_type": None,
                "reviewer": None,
                "license_evidence": None,
            }
        ],
        destination / "review_form.jsonl",
    )
    with Path("raw_data/curated/train_text.jsonl").open(encoding="utf-8-sig") as handle:
        existing = [json.loads(line)["quote"] for line in handle if line.strip()]
    nearest = novelty_dicts([quote], existing)[0]
    near_pairs = near_duplicate_candidates(unique)
    write_jsonl((pair.__dict__ for pair in near_pairs), destination / "near_duplicate_review.jsonl")
    summary = {
        "source_posts": sum(
            len(json.loads((batch / "report.json").read_text(encoding="utf-8"))["posts"]) for batch in batches
        ),
        "extracted_fragments": len(records),
        "unique_fragments": len(unique),
        "exact_duplicates_between_batches": len(exact_pairs),
        "near_duplicate_pairs": len(near_pairs),
        "assistant_label_counts": dict(Counter(item["suggested_label"] for item in audit)),
        "shortlisted_candidates": 1,
        "human_confirmed": 0,
        "training_approved": 0,
        "original_training_rows_unchanged": len(existing),
        "candidate_novelty": nearest,
        "mcp_status": "configured_but_connection_refused",
        "actual_transport": "existing_skill_cli",
    }
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = (
        "# 2026-10-19 小规模采集审核包\n\n"
        f"读取 {summary['source_posts']} 篇公开笔记，提取 {len(records)} 段正文/评论。\n\n"
        "以下只有 1 条助手初筛候选，未经人类确认，也没有训练使用许可。\n\n"
        f"> {quote}\n\n"
        f"[原帖]({candidate['source_url']})\n\n"
        "判断依据：将手的数量与成绩线性对应，形成自我夸张与荒诞归因。"
        "仍需判断这是否只是普通炫耀，以及脱离原图是否成立。\n\n"
        f"训练语料最近邻相似度：{nearest['similarity']:.2f}；完全重复：{nearest['exact_match']}。\n\n"
        "其余片段的排除/存疑理由见 assistant_triage.jsonl；原始响应已版本化保存并剔除访问令牌/用户档案字段。\n\n"
        "正式人工表单：review_form.jsonl。原来的 96 条语料保持不变。\n"
    )
    (destination / "REVIEW.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
