"""Build dialogue pilot splits: synthetic user prompts paired with real collected replies.

Split membership comes from the frozen quote split, so a held-out reply never appears in train
under another prompt. Replies are unchanged; prompts are marked prompt_origin=synthetic_assistant.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from src.training.data_gate import PILOT_SCOPE


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quote-split-dir", default="data/splits-pilot-20261020-r2")
    parser.add_argument("--work-dir", default="data/dialogue-pilot-20261020/work")
    parser.add_argument("--output-dir", default="data/splits-dialogue-pilot-20261020")
    parser.add_argument("--train-prompts-per-reply", type=int, default=2)
    args = parser.parse_args()
    target = Path(args.output_dir)
    if target.exists():
        raise SystemExit(f"{target} exists; dialogue splits are frozen")
    quote_dir = Path(args.quote_split_dir)
    meta = {row["id"]: row for row in read_jsonl(quote_dir / "metadata_index.jsonl")}
    split_of = {}
    for split in ("train", "validation", "test"):
        for row in read_jsonl(quote_dir / f"{split}.jsonl"):
            split_of[row["id"]] = split
    work = Path(args.work_dir)
    replies = {}
    for chunk in sorted(glob.glob(str(work / "chunk*.jsonl"))):
        for row in read_jsonl(chunk):
            replies[row["id"]] = row["reply"]
    prompts = {}
    for path in sorted(glob.glob(str(work / "prompts*.jsonl"))):
        for row in read_jsonl(path):
            prompts[row["id"]] = row["prompts"]
    missing = sorted(set(split_of) - set(prompts))
    if missing:
        raise SystemExit(f"{len(missing)} replies lack prompts, e.g. {missing[:3]}")
    rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    index = []
    for quote_id, split in split_of.items():
        reply = replies[quote_id]
        if meta[quote_id]["quote"] != reply:
            raise SystemExit(f"Reply text drifted from metadata for {quote_id}")
        count = args.train_prompts_per_reply if split == "train" else 1
        for k, prompt in enumerate(prompts[quote_id][:count]):
            pair_id = f"{quote_id}#p{k}"
            rows[split].append({"id": pair_id, "prompt": prompt, "text": reply})
            index.append(
                {
                    **meta[quote_id],
                    "id": pair_id,
                    "reply_id": quote_id,
                    "prompt": prompt,
                    "prompt_origin": "synthetic_assistant",
                    "review_scope": PILOT_SCOPE,
                    "split": split,
                }
            )
    target.mkdir(parents=True)
    for split, items in rows.items():
        with (target / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (target / "metadata_index.jsonl").open("w", encoding="utf-8") as handle:
        for item in index:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    stats = {
        "scope": PILOT_SCOPE,
        "prompt_origin": "synthetic_assistant",
        "reply_origin": "collected_unreviewed",
        "rows": {split: len(items) for split, items in rows.items()},
        "distinct_replies": dict(Counter(split_of.values())),
    }
    (target / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
