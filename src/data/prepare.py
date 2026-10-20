from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.data.deduplicate import DuplicatePair, near_duplicate_candidates
from src.data.statistics import token_length_statistics
from src.data.validate import write_jsonl


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_leakage_groups(
    records: list[dict[str, Any]], near_threshold: float = 92.0
) -> tuple[dict[str, str], list[DuplicatePair]]:
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Record IDs must be unique before grouping")
    union = _UnionFind(ids)
    by_source: dict[str, list[str]] = defaultdict(list)
    for record in records:
        source_group = record.get("source_group") or record.get("source_id")
        sources = list(record.get("source_groups", []))
        if source_group:
            sources.append(source_group)
        for source in sources:
            by_source[str(source)].append(str(record["id"]))
    for group_ids in by_source.values():
        for record_id in group_ids[1:]:
            union.union(group_ids[0], record_id)
    near_pairs = near_duplicate_candidates(records, threshold=near_threshold)
    for pair in near_pairs:
        union.union(pair.left_id, pair.right_id)
    return {record_id: union.find(record_id) for record_id in ids}, near_pairs


def split_grouped_records(
    records: list[dict[str, Any]],
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    near_threshold: float = 92.0,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], list[DuplicatePair]]:
    ratios = ratios or {"train": 0.8, "validation": 0.1, "test": 0.1}
    if set(ratios) != {"train", "validation", "test"} or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("ratios must contain train/validation/test and sum to 1")
    groups, pairs = build_leakage_groups(records, near_threshold=near_threshold)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[groups[str(record["id"])]].append(record)
    units = list(grouped.values())
    if records and len(units) < sum(value > 0 for value in ratios.values()):
        raise ValueError("Too few independent source/duplicate groups for non-empty train/validation/test")
    random.Random(seed).shuffle(units)
    targets = {name: len(records) * ratio for name, ratio in ratios.items()}
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in ratios}
    ordered_units = sorted(units, key=len, reverse=True)
    for position, unit in enumerate(ordered_units):
        empty = [name for name in ratios if ratios[name] > 0 and not splits[name]]
        available = empty if len(ordered_units) - position <= len(empty) else list(ratios)
        split = max(available, key=lambda name: targets[name] - len(splits[name]))
        splits[split].extend(unit)
    return splits, groups, pairs


DIALOGUE_USER_TAG = "评论区有人说："
DIALOGUE_REPLY_TAG = "我回复："


def dialogue_prefix(prompt: str, history: list[tuple[str, str]] | None = None) -> str:
    """Text before the reply; shared by training and chat so the boundary tokenizes identically."""
    lines = []
    for user, reply in history or []:
        lines += [DIALOGUE_USER_TAG + user, DIALOGUE_REPLY_TAG + reply]
    lines += [DIALOGUE_USER_TAG + prompt, DIALOGUE_REPLY_TAG]
    return "\n".join(lines)


def encode_dialogue(tokenizer: Any, prompt: str, reply: str, max_length: int) -> dict[str, list[int]]:
    """EOS boundary + prompt framing (loss-masked) + reply + EOS (trained)."""
    if not prompt.strip() or not reply.strip():
        raise ValueError("Dialogue prompt and reply must be non-empty")
    tokenizer_special_tokens(tokenizer)
    prefix = [int(tokenizer.eos_token_id)] + tokenizer.encode(dialogue_prefix(prompt), add_special_tokens=False)
    budget = max_length - len(prefix) - 1
    if budget < 4:
        raise ValueError("Prompt leaves too little room for the reply")
    reply_ids = tokenizer.encode(reply, add_special_tokens=False, truncation=True, max_length=budget)
    input_ids = prefix + list(reply_ids) + [int(tokenizer.eos_token_id)]
    labels = [-100] * len(prefix) + list(reply_ids) + [int(tokenizer.eos_token_id)]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


CHAT_END_TOKEN = "<|im_end|>"


def chat_prefix_ids(
    tokenizer: Any, prompt: str, history: list[tuple[str, str]] | None = None, system_prompt: str | None = None
) -> list[int]:
    """Qwen chat-template prefix up to the assistant turn, thinking disabled."""
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    for user, reply in history or []:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


def encode_dialogue_chat(
    tokenizer: Any, prompt: str, reply: str, max_length: int, system_prompt: str | None = None
) -> dict[str, list[int]]:
    """Chat-template prefix (loss-masked) + reply + <|im_end|> (trained)."""
    if not prompt.strip() or not reply.strip():
        raise ValueError("Dialogue prompt and reply must be non-empty")
    tokenizer_special_tokens(tokenizer)
    end_id = tokenizer.convert_tokens_to_ids(CHAT_END_TOKEN)
    if end_id is None or end_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer lacks <|im_end|>; chat_template format needs a Qwen chat tokenizer")
    prefix = chat_prefix_ids(tokenizer, prompt, None, system_prompt)
    budget = max_length - len(prefix) - 1
    if budget < 4:
        raise ValueError("Prompt leaves too little room for the reply")
    reply_ids = tokenizer.encode(reply, add_special_tokens=False, truncation=True, max_length=budget)
    input_ids = prefix + list(reply_ids) + [int(end_id)]
    labels = [-100] * len(prefix) + list(reply_ids) + [int(end_id)]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def tokenizer_special_tokens(tokenizer: Any) -> dict[str, int | None]:
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token; cannot create independent quote boundaries")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


def encode_quote(tokenizer: Any, quote: str, max_length: int) -> dict[str, list[int]]:
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("Cannot encode an empty quote")
    tokenizer_special_tokens(tokenizer)
    prefix: list[int] = []
    # Qwen Base has no BOS: EOS also marks the previous document boundary.
    # Training on this boundary makes the first quote token learnable and agrees with generation.
    start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    prefix = [int(start_id)]
    content_budget = max_length - len(prefix) - 1
    if content_budget < 1:
        raise ValueError("max_length is too small for BOS/EOS boundaries")
    content = tokenizer.encode(quote, add_special_tokens=False, truncation=True, max_length=content_budget)
    input_ids = prefix + list(content) + [int(tokenizer.eos_token_id)]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": input_ids.copy()}


def prepare_splits(
    records: list[dict[str, Any]],
    tokenizer: Any,
    output_dir: str | Path,
    ratios: dict[str, float],
    seed: int,
    max_length: int,
    near_threshold: float,
) -> dict[str, Any]:
    splits, groups, near_pairs = split_grouped_records(records, ratios, seed, near_threshold)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "records": sorted(records, key=lambda row: row["id"]),
                "seed": seed,
                "ratios": ratios,
                "near_threshold": near_threshold,
                "max_length": max_length,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest = destination / "split_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
        raise ValueError("Frozen split manifest differs. Use a new versioned splits_dir; do not recycle the test set.")
    report: dict[str, Any] = {"seed": seed, "near_threshold": near_threshold, "splits": {}}
    for split_name, split_records in splits.items():
        exported = [{"id": record["id"], "text": record["quote"]} for record in split_records]
        write_jsonl(exported, destination / f"{split_name}.jsonl")
        lengths = [len(encode_quote(tokenizer, record["quote"], max_length)["input_ids"]) for record in split_records]
        report["splits"][split_name] = token_length_statistics(lengths)
        report["splits"][split_name]["total_tokens"] = sum(lengths)
        source_counts: dict[str, int] = defaultdict(int)
        for record in split_records:
            source_counts[str(record.get("source_platform"))] += 1
        report["splits"][split_name]["source_distribution"] = dict(source_counts)
    write_jsonl(
        (
            {**record, "leakage_group": groups[str(record["id"])], "split": split_name}
            for split_name, values in splits.items()
            for record in values
        ),
        destination / "metadata_index.jsonl",
    )
    write_jsonl((pair.__dict__ for pair in near_pairs), destination / "near_duplicate_review.jsonl")
    (destination / "statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest.write_text(json.dumps({"fingerprint": fingerprint, "seed": seed}, indent=2), encoding="utf-8")
    return report
