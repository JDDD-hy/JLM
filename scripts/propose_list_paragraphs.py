"""Propose numbered-list paragraphs from saved list-style notes as selection input; never fetches online."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STOP = re.compile(r"^(#|参考文献|\[\d+\]|aigc|（投资有风险|\(投资有风险)")
NUMBER = re.compile(r"^\s*(\d+)\s*[.、．，]\s*")


def split_items(text: str) -> list[str]:
    """Return verbatim slices of each numbered item (inner whitespace preserved)."""
    items: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if STOP.match(stripped):
            if items:
                break
            continue
        match = NUMBER.match(line)
        if match:
            items.append([line[match.end():]])
        elif items and re.fullmatch(r"[\W_]+", stripped):
            break  # separator line ends the numbered list
        elif items:
            items[-1].append(line)
    out = []
    for parts in items:
        joined = "\n".join(parts).strip()
        if len(joined) >= 8:
            out.append(joined)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--parent-ids", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parents = {}
    with (Path(args.batch) / "extracted.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                parents[item["id"]] = item
    rows = []
    for pid in args.parent_ids:
        quote = parents[pid]["quote"]
        for text in split_items(quote):
            if text not in quote:
                raise ValueError("Proposed paragraph is not verbatim")
            rows.append({"parent_id": pid, "text": text})
    Path(args.output).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for r in rows:
        print(r["parent_id"][-6:], r["text"][:60].replace("\n", " | "))
    print(json.dumps({"proposed": len(rows)}))


if __name__ == "__main__":
    main()
