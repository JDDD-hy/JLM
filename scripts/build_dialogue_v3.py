"""Dialogue v3: judge-filtered real-quote pairs + topic-anchored synthetic 嘉豪 exchanges.

Real pairs keep their v2 split; synthetic exchanges get their own seeded split and unique leakage groups.
Synthetic replies are marked reply_origin=synthetic_assistant_jiahao_style.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from src.training.data_gate import PILOT_SCOPE


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", default="data/splits-dialogue-v2-20261020")
    parser.add_argument("--work-dir", default="data/dialogue-v3-20261020/work")
    parser.add_argument("--output-dir", default="data/splits-dialogue-v3-20261020")
    parser.add_argument("--min-relevance", type=float, default=0.7)
    parser.add_argument("--synthetic-eval", type=int, default=30, help="Synthetic rows for each of validation/test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target = Path(args.output_dir)
    if target.exists():
        raise SystemExit(f"{target} exists; splits are frozen")
    work = Path(args.work_dir)
    relevance = {r["id"]: r["relevance"] for r in read_jsonl(work / "v2_pair_relevance.jsonl")}
    v2_meta = {r["id"]: r for r in read_jsonl(Path(args.v2_dir) / "metadata_index.jsonl")}
    rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    index: list[dict] = []
    dropped = Counter()
    for split in rows:
        for row in read_jsonl(Path(args.v2_dir) / f"{split}.jsonl"):
            if relevance[row["id"]] < args.min_relevance:
                dropped[split] += 1
                continue
            rows[split].append(row)
            index.append({**v2_meta[row["id"]], "judge_relevance": relevance[row["id"]], "reply_origin": "collected"})
    synthetic = []
    for path in sorted(glob.glob(str(work / "synth*.jsonl"))):
        synthetic.extend(read_jsonl(path))
    rng = random.Random(args.seed)
    rng.shuffle(synthetic)
    n = args.synthetic_eval
    synth_splits = {"validation": synthetic[:n], "test": synthetic[n : 2 * n], "train": synthetic[2 * n :]}
    for split, items in synth_splits.items():
        for item in items:
            digest = hashlib.sha256(f"{item['prompt']}\n{item['reply']}".encode()).hexdigest()[:16]
            pair_id = f"syn_{digest}"
            rows[split].append({"id": pair_id, "prompt": item["prompt"], "text": item["reply"]})
            index.append(
                {
                    "id": pair_id,
                    "quote": item["reply"],
                    "prompt": item["prompt"],
                    "topic": item.get("topic"),
                    "prompt_origin": "synthetic_assistant",
                    "reply_origin": "synthetic_assistant_jiahao_style",
                    "review_scope": PILOT_SCOPE,
                    "leakage_group": f"synthetic:{pair_id}",
                    "split": split,
                }
            )
    target.mkdir(parents=True)
    for split, items in rows.items():
        rng.shuffle(items)
        with (target / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (target / "metadata_index.jsonl").open("w", encoding="utf-8") as handle:
        for item in index:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    stats = {
        "scope": PILOT_SCOPE,
        "min_relevance": args.min_relevance,
        "dropped_real_pairs": dict(dropped),
        "rows": {split: len(items) for split, items in rows.items()},
        "synthetic_rows": {split: len(items) for split, items in synth_splits.items()},
    }
    (target / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
