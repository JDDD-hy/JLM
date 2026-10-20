"""Build a shuffled blind-evaluation sheet (model generations mixed with held-out real quotes)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

TEMPLATE_MARKERS = {
    "就很": re.compile(r"就很"),
    "书包里": re.compile(r"书包里"),
    "直呼我全名": re.compile(r"直呼我全名"),
    "追人": re.compile(r"怎么追人"),
    "同龄人": re.compile(r"同龄人"),
    "朋友圈发": re.compile(r"朋友圈发"),
    "讲到凌晨": re.compile(r"讲到凌晨"),
    "没敢打断": re.compile(r"没敢打断"),
}


def template_rates(texts: list[str]) -> dict[str, float]:
    total = max(len(texts), 1)
    rates = {name: round(sum(bool(p.search(t)) for t in texts) / total, 3) for name, p in TEMPLATE_MARKERS.items()}
    rates["any_marker"] = round(sum(any(p.search(t) for p in TEMPLATE_MARKERS.values()) for t in texts) / total, 3)
    return rates


def read_texts(path: str) -> list[str]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line)["text"] for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, help="evaluation JSON containing 'generations'")
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--per-side", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    seen, generations = set(), []
    for text in report["generations"]:
        cleaned = text.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            generations.append(cleaned)
    rng = random.Random(args.seed)
    real = read_texts(args.test_file)
    picked_gen = rng.sample(generations, min(args.per_side, len(generations)))
    picked_real = rng.sample(real, min(args.per_side, len(real)))
    items = [("model", text) for text in picked_gen] + [("real", text) for text in picked_real]
    rng.shuffle(items)
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "盲评表.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["序号", "语录", "像嘉豪语录(1/0)", "只是套模板(1/0)", "备注"])
        for index, (_, text) in enumerate(items, start=1):
            writer.writerow([index, text, "", "", ""])
    key = [{"序号": index, "source": source} for index, (source, _) in enumerate(items, start=1)]
    (target / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    stats = {
        "generated_unique": len(generations),
        "generated_total": len(report["generations"]),
        "template_rates": {
            "generations": template_rates(generations),
            "train": template_rates(read_texts(args.train_file)),
            "test": template_rates(real),
        },
    }
    (target / "template_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
