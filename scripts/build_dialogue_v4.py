"""Dialogue v4 = v3 splits + 16 external quotes (github:zczjyq/jiahao-skill, safety-filtered) + new styled dialogues.

External quotes carry reply_origin=external_github_unlicensed so they can be dropped in one step if needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from src.training.data_gate import PILOT_SCOPE


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-dir", default="data/splits-dialogue-v3-20261020")
    parser.add_argument("--work-dir", default="data/dialogue-v4-20261020/work")
    parser.add_argument("--output-dir", default="data/splits-dialogue-v4-20261020")
    parser.add_argument("--validation-share", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target, work = Path(args.output_dir), Path(args.work_dir)
    if target.exists():
        raise SystemExit(f"{target} exists; splits are frozen")
    shutil.copytree(args.v3_dir, target)
    (target / "statistics.json").unlink(missing_ok=True)
    rows: dict[str, list[dict]] = {"train": [], "validation": []}
    meta: list[dict] = []

    for item in read_jsonl(work / "external_pairs.jsonl"):
        for k, prompt in enumerate(item["prompts"]):
            pair_id = f"{item['id']}#p{k}"
            rows["train"].append({"id": pair_id, "prompt": prompt, "text": item["reply"]})
            meta.append(
                {
                    "id": pair_id,
                    "quote": item["reply"],
                    "prompt": prompt,
                    "source": item["source"],
                    "prompt_origin": "synthetic_assistant",
                    "reply_origin": "external_github_unlicensed",
                    "review_scope": PILOT_SCOPE,
                    "leakage_group": f"external:{item['id']}",
                    "split": "train",
                }
            )

    synthetic = []
    for name in ("synthA.jsonl", "synthB.jsonl"):
        path = work / name
        if path.exists():
            synthetic.extend(read_jsonl(path))
    rng = random.Random(args.seed)
    rng.shuffle(synthetic)
    cut = round(len(synthetic) * args.validation_share)
    for split, items in (("validation", synthetic[:cut]), ("train", synthetic[cut:])):
        for item in items:
            digest = hashlib.sha256(f"{item['prompt']}\n{item['reply']}".encode()).hexdigest()[:16]
            pair_id = f"syn4_{digest}"
            rows[split].append({"id": pair_id, "prompt": item["prompt"], "text": item["reply"]})
            meta.append(
                {
                    "id": pair_id,
                    "quote": item["reply"],
                    "prompt": item["prompt"],
                    "topic": item.get("topic") or item.get("direction"),
                    "prompt_origin": "synthetic_assistant",
                    "reply_origin": "synthetic_assistant_jiahao_style_v4",
                    "review_scope": PILOT_SCOPE,
                    "leakage_group": f"synthetic:{pair_id}",
                    "split": split,
                }
            )

    for split, items in rows.items():
        append_jsonl(target / f"{split}.jsonl", items)
    append_jsonl(target / "metadata_index.jsonl", meta)
    counts = {
        s: sum(1 for _ in (target / f"{s}.jsonl").open(encoding="utf-8")) for s in ("train", "validation", "test")
    }
    stats = {
        "base": args.v3_dir,
        "added": {
            "external_pairs": sum(m["reply_origin"].startswith("external") for m in meta),
            "synthetic_v4": dict(Counter(m["split"] for m in meta if m["id"].startswith("syn4_"))),
        },
        "rows": counts,
    }
    (target / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
