"""Score a filled blind-evaluation sheet against its answer key."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rate(rows: list[dict], column: str) -> dict:
    marked = [row[column].strip() for row in rows if row[column].strip() in {"0", "1"}]
    return {"filled": len(marked), "yes_rate": round(marked.count("1") / len(marked), 3) if marked else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", default="盲评表-试点R3-20261020.csv")
    parser.add_argument("--key", default="outputs/pilot-20261020/blind-eval-r3/answer_key.json")
    args = parser.parse_args()
    key = {str(item["序号"]): item["source"] for item in json.loads(Path(args.key).read_text(encoding="utf-8"))}
    with Path(args.sheet).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for source in ("model", "real"):
        subset = [row for row in rows if key[row["序号"]] == source]
        result[source] = {
            "rows": len(subset),
            "像嘉豪语录": rate(subset, "像嘉豪语录(1/0)"),
            "只是套模板": rate(subset, "只是套模板(1/0)"),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
