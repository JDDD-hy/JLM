"""Dialogue v2 splits: fewer template replies, more real independent replies, emoji-spam normalized.

Validation/test are whole leakage groups re-drawn from the v1 held-out groups; the remaining v1
held-out groups move to train. Replies keep their original text in metadata (original_quote).
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.training.data_gate import PILOT_SCOPE

TAG = re.compile(r"\[[^\[\]]{1,8}R\]")


def normalize_reply(text: str, max_tags: int = 2) -> str:
    text = re.sub(r"(\[[^\[\]]{1,8}R\])(?:\s*\1)+", r"\1", text)
    count = 0

    def keep(match: re.Match) -> str:
        nonlocal count
        count += 1
        return match.group(0) if count <= max_tags else ""

    return TAG.sub(keep, text).strip()


def is_series(record: dict) -> bool:
    return bool(record.get("source_parent_id")) or "paragraph" in str(record.get("extraction_method") or "")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pick_groups(records: list[dict], target: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_group[record["leakage_group"]].append(record)
    groups = sorted(by_group)
    rng.shuffle(groups)
    chosen, rest, size = [], [], 0
    for group in groups:
        if size < target:
            chosen.extend(by_group[group])
            size += len(by_group[group])
        else:
            rest.extend(by_group[group])
    return chosen, rest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quote-split-dir", default="data/splits-pilot-20261020")
    parser.add_argument("--work-dir", default="data/dialogue-pilot-20261020/work")
    parser.add_argument("--output-dir", default="data/splits-dialogue-v2-20261020")
    parser.add_argument("--eval-size", type=int, default=40)
    parser.add_argument("--max-series", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target = Path(args.output_dir)
    if target.exists():
        raise SystemExit(f"{target} exists; splits are frozen")
    meta = {row["id"]: row for row in read_jsonl(Path(args.quote_split_dir) / "metadata_index.jsonl")}
    prompts = {}
    for path in sorted(glob.glob(str(Path(args.work_dir) / "prompts*.jsonl"))):
        for row in read_jsonl(path):
            prompts[row["id"]] = row["prompts"]
    usable = [meta[i] for i in prompts if i in meta]
    rng = random.Random(args.seed)
    old = {name: [r for r in usable if r["split"] == name] for name in ("train", "validation", "test")}
    val, val_rest = pick_groups(old["validation"], args.eval_size, rng)
    test, test_rest = pick_groups(old["test"], args.eval_size, rng)
    independent_train = [r for r in old["train"] + val_rest + test_rest if not is_series(r)]
    series_train = [r for r in old["train"] if is_series(r)]
    series_train = rng.sample(series_train, min(args.max_series, len(series_train)))
    splits = {"train": independent_train + series_train, "validation": val, "test": test}
    target.mkdir(parents=True)
    index = []
    stats = {}
    for name, records in splits.items():
        count = 2 if name == "train" else 1
        rows = []
        for record in records:
            reply = normalize_reply(record["quote"])
            for k, prompt in enumerate(prompts[record["id"]][:count]):
                pair_id = f"{record['id']}#p{k}"
                rows.append({"id": pair_id, "prompt": prompt, "text": reply})
                index.append(
                    {
                        **record,
                        "id": pair_id,
                        "reply_id": record["id"],
                        "quote": reply,
                        "original_quote": record["quote"],
                        "reply_normalized": reply != record["quote"],
                        "prompt": prompt,
                        "prompt_origin": "synthetic_assistant",
                        "review_scope": PILOT_SCOPE,
                        "split": name,
                    }
                )
        rng.shuffle(rows)
        with (target / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[name] = {
            "pairs": len(rows),
            "replies": len(records),
            "series_replies": sum(is_series(r) for r in records),
            "replies_with_jiuhen": sum("就很" in r["quote"] for r in records),
        }
    groups = {name: {r["leakage_group"] for r in records} for name, records in splits.items()}
    assert not (groups["train"] & groups["validation"]) and not (groups["train"] & groups["test"])
    with (target / "metadata_index.jsonl").open("w", encoding="utf-8") as handle:
        for item in index:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {
        "scope": PILOT_SCOPE,
        "splits": stats,
        "normalized_replies": dict(Counter(i["reply_normalized"] for i in index if i["id"].endswith("#p0"))),
    }
    (target / "statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
