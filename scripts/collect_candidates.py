"""Bounded read-only collection through the existing skill; outputs always require review."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

from src.collection.parsers import parse_feed_detail
from src.collection.provenance import collection_timestamp, provenance_record
from src.collection.skill_client import SkillClient, sanitize_response
from src.data.clean import canonical_for_dedup
from src.data.deduplicate import near_duplicate_candidates
from src.data.quality import suggest_quality_label
from src.data.validate import stable_record_id, write_jsonl


def is_quote_request(title: str, query: str = "") -> bool:
    """Require an explicit title theme; a search term alone is not source evidence."""
    return (
        any(term in title for term in ("嘉豪", "嘉欣", "豪气冲天", "豪情在天", "自在极意豪", "豪味"))
        and (
            "自在极意豪" in title
            or any(
                term in title
                for term in ("文案", "语录", "征集", "自我介绍", "自介", "的一天", "的日常", "言论", "日记", "发言")
            )
            or "one day" in title.lower()
        )
        and not any(
            term in title
            for term in ("这个梗", "是什么梗", "是什么意思", "电子宠物", "上线", "古风", "速成班", "洪嘉豪")
        )
    )


def permission_metadata(path: str | None) -> dict:
    if path is None:
        return {}
    source = Path(path)
    encoded = source.read_bytes()
    evidence = json.loads(encoded.decode("utf-8-sig"))
    if evidence.get("scope_confirmation_status") != "confirmed_by_user":
        raise ValueError("Collection permission scope has not been confirmed")
    required = {"xiaohongshu_public_notes", "xiaohongshu_public_comments"}
    if not required.issubset(evidence.get("platforms_in_scope") or []):
        raise ValueError("Collection permission does not cover these sources")
    if "automated_read_only_collection_via_chrome_bridge" not in (
        evidence.get("automated_collection_methods_in_scope") or []
    ):
        raise ValueError("Collection permission does not cover this method")
    return {
        "collection_permission_status": "user_declared_school_permission_scope_confirmed",
        "permission_evidence_path": str(source.resolve()),
        "permission_evidence_sha256": hashlib.sha256(encoded).hexdigest(),
        "permission_independently_verified": False,
    }


def archive(payload: dict, directory: Path, kind: str) -> str:
    encoded = json.dumps(sanitize_response(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    target = directory / "responses" / f"{kind}-{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(encoded)
    return digest


def comment_arguments(limit: int) -> list[str]:
    if limit not in (10, 50):
        raise ValueError("Only 10 or 50 comments per post are supported")
    return ["--max-comment-items", str(limit)] + (["--load-all-comments"] if limit == 50 else [])


def search_arguments(keyword: str, sort_by: str, note_type: str, no_filters: bool) -> list[str]:
    return ["--keyword", keyword] + ([] if no_filters else ["--sort-by", sort_by, "--note-type", note_type])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-root", default="C:/Users/asus/.codex/skills/xiaohongshu-skills")
    parser.add_argument("--keywords", nargs="+", default=["嘉豪文案", "嘉豪评论区"])
    parser.add_argument("--max-posts", type=int, default=3, choices=range(1, 4))
    parser.add_argument("--max-comments", type=int, default=10, choices=[10, 50])
    parser.add_argument("--feed-ids", nargs="+", help="Explicit source IDs; permits bounded revisits")
    parser.add_argument("--sort-by", default="最多评论", choices=["最多评论", "最新", "综合"])
    parser.add_argument("--note-type", default="图文", choices=["不限", "图文", "视频"])
    parser.add_argument("--no-filters", action="store_true", help="Use the skill's unfiltered search path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permission-evidence", default=None, help="User-confirmed source permission JSON")
    parser.add_argument("--min-interval", type=float, default=8.0, help="Minimum seconds between bridge calls")
    parser.add_argument("--max-interval", type=float, default=None, help="Upper bound for jittered gap between calls")
    args = parser.parse_args()
    if args.min_interval < 0 or (args.max_interval is not None and args.max_interval < args.min_interval):
        parser.error("--max-interval must be >= --min-interval >= 0")
    exclusion_path = Path("data/rejected/collection_skip_ids.json")
    excluded = json.loads(exclusion_path.read_text(encoding="utf-8-sig")) if exclusion_path.exists() else []
    if not isinstance(excluded, list) or any(
        not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{24}", item) for item in excluded
    ):
        raise ValueError("Collection exclusion list must contain valid feed IDs")
    excluded_ids = set(excluded)
    requested_ids = set(args.feed_ids or [])
    if any(not re.fullmatch(r"[0-9a-f]{24}", item) for item in requested_ids):
        parser.error("feed IDs must be 24 lowercase hexadecimal characters")
    if args.max_comments == 50 and (args.max_posts > 2 or not 1 <= len(requested_ids) <= 2):
        parser.error("The 50-comment pilot requires 1-2 explicit feed IDs and max-posts <= 2")
    permission = permission_metadata(args.permission_evidence)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = SkillClient(args.skill_root)
    checkpoint_path = output / "checkpoint.json"
    completed_ids = set(json.loads(checkpoint_path.read_text(encoding="utf-8"))) if checkpoint_path.exists() else set()
    already_read = {path.stem for path in Path("raw_data/posts").glob("*.json")}
    for prior in Path("data/raw").glob("live-*/checkpoint.json"):
        if prior.resolve() != checkpoint_path.resolve():
            already_read.update(json.loads(prior.read_text(encoding="utf-8")))
    existing = set()
    with Path("raw_data/curated/train_text.jsonl").open(encoding="utf-8-sig") as handle:
        existing.update(canonical_for_dedup(json.loads(line)["quote"]) for line in handle if line.strip())
    report = {
        "started_at": collection_timestamp(),
        "transport": "existing_skill_cli",
        "searches": [],
        "posts": [],
        "failures": [],
        "training_approved": 0,
        "permission": permission,
        "note_type": None if args.no_filters else args.note_type,
        "filters_applied": not args.no_filters,
        "max_comments_per_post": args.max_comments,
        "requested_feed_ids": sorted(requested_ids),
        "excluded_feed_ids": sorted(excluded_ids),
        "call_interval_seconds": [args.min_interval, args.max_interval or args.min_interval],
    }
    since_last_call = 0.0
    records = {}
    candidates_path = output / "extracted.jsonl"
    if candidates_path.exists():
        with candidates_path.open(encoding="utf-8") as handle:
            records = {item["id"]: item for line in handle if line.strip() for item in [json.loads(line)]}

    def call(command, *arguments):
        nonlocal since_last_call
        gap = random.uniform(args.min_interval, args.max_interval) if args.max_interval else args.min_interval
        remaining = gap - (time.monotonic() - since_last_call)
        if remaining > 0:
            time.sleep(remaining)
        result = client.call(command, *arguments)
        since_last_call = time.monotonic()
        return result

    try:
        login = call("check-login")
        if login.get("logged_in") is not True:
            raise RuntimeError("Not logged in; stopped without changing credentials")
        report["login_verified"] = True
        for keyword in args.keywords:
            result = call(
                "search-feeds", *search_arguments(keyword, args.sort_by, args.note_type, args.no_filters)
            )
            archive(result, output, "search")
            feeds = result.get("feeds", [])
            report["searches"].append({"keyword": keyword, "count": len(feeds)})
            print(json.dumps({"search": keyword, "count": len(feeds)}, ensure_ascii=False), flush=True)

            # Prioritize explicit quote requests over generic mentions of Jiahao.
            def relevance(feed):
                title = feed.get("displayTitle") or feed.get("note_card", {}).get("display_title", "")
                return 3 * bool(re.search(r"求|征集|分享|评论区", title)) + 2 * bool(re.search(r"文案|语录", title))

            for feed in sorted(feeds, key=relevance, reverse=True):
                feed_id = feed.get("id")
                token = feed.get("xsecToken") or feed.get("xsec_token")
                title = feed.get("displayTitle") or feed.get("note_card", {}).get("display_title", "")
                if (
                    not feed_id
                    or not re.fullmatch(r"[0-9a-f]{24}", str(feed_id))
                    or not token
                    or feed_id in excluded_ids
                    or (requested_ids and feed_id not in requested_ids)
                    or (feed_id in already_read and feed_id not in requested_ids)
                    or feed_id in completed_ids
                    or feed.get("type") == "video"
                    or not is_quote_request(title, keyword)
                ):
                    continue
                if len(report["posts"]) >= args.max_posts:
                    break
                print(json.dumps({"reading": feed_id, "title": title}, ensure_ascii=False), flush=True)
                detail = call(
                    "get-feed-detail",
                    "--feed-id",
                    feed_id,
                    "--xsec-token",
                    token,
                    *comment_arguments(args.max_comments),
                    "--keyword",
                    keyword,
                )
                if isinstance(detail.get("comments"), list):
                    detail["comments"] = detail["comments"][:args.max_comments]
                digest = archive(detail, output, "detail")
                parsed = parse_feed_detail(detail)
                if not parsed:
                    raise RuntimeError("Detail returned no parseable note/comments; inspect saved response")
                for item in parsed:
                    quote = item["text"]
                    record_id = stable_record_id(quote, None)
                    record = provenance_record(
                        record_id=record_id,
                        quote=quote,
                        scene=None,
                        source_platform="xiaohongshu",
                        source_type=item["source_type"],
                        collection_method="existing_skill_cli",
                        source_id=item["source_id"],
                        source_url=f"https://www.xiaohongshu.com/explore/{feed_id}",
                        collected_at=collection_timestamp(),
                        origin="unknown",
                    )
                    record.update(
                        source_group=feed_id,
                        response_sha256=digest,
                        exact_existing=canonical_for_dedup(quote) in existing,
                        rule_suggestion=suggest_quality_label(quote).label,
                        access_status="visible_to_existing_logged_in_session",
                        collection_permission_status="not_independently_verified",
                    )
                    record.update(permission)
                    records.setdefault(record_id, record)
                # Persist data before checkpoint acknowledgement.
                write_jsonl(records.values(), candidates_path)
                completed_ids.add(feed_id)
                checkpoint_path.write_text(json.dumps(sorted(completed_ids)), encoding="utf-8")
                report["posts"].append({"id": feed_id, "title": title, "extracted": len(parsed)})
                print(json.dumps(report["posts"][-1], ensure_ascii=False), flush=True)
            if len(report["posts"]) >= args.max_posts:
                break
    except RuntimeError as exc:
        report["failures"].append(str(exc))
    values = list(records.values())
    write_jsonl(values, candidates_path)
    write_jsonl((pair.__dict__ for pair in near_duplicate_candidates(values)), output / "near_duplicates.jsonl")
    report.update(
        ended_at=collection_timestamp(),
        unique_extracted=len(values),
        exact_existing=sum(item["exact_existing"] for item in values),
        rule_triage=dict(Counter(item["rule_suggestion"] for item in values)),
    )
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
