import json

import pytest

from scripts import collect_candidates
from scripts.collect_candidates import comment_arguments, is_quote_request, permission_metadata, search_arguments
from src.collection.parsers import parse_feed_detail


def test_comment_loading_is_bounded_and_opt_in():
    assert comment_arguments(10) == ["--max-comment-items", "10"]
    assert comment_arguments(50) == ["--max-comment-items", "50", "--load-all-comments"]
    with pytest.raises(ValueError):
        comment_arguments(0)


def test_unfiltered_search_omits_filter_options():
    assert search_arguments("自在极意豪", "最多评论", "图文", True) == ["--keyword", "自在极意豪"]
    assert search_arguments("嘉豪", "最新", "图文", False) == [
        "--keyword", "嘉豪", "--sort-by", "最新", "--note-type", "图文"
    ]


def test_persistent_exclusion_blocks_even_explicit_revisit(tmp_path, monkeypatch):
    feed_id = "a" * 24
    source = tmp_path / "raw_data/curated/train_text.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    excluded = tmp_path / "data/rejected/collection_skip_ids.json"
    excluded.parent.mkdir(parents=True)
    excluded.write_text(json.dumps([feed_id]), encoding="utf-8")

    class Client:
        def call(self, command, *args):
            if command == "check-login":
                return {"logged_in": True}
            if command == "search-feeds":
                return {"feeds": [{"id": feed_id, "xsecToken": "fixture", "displayTitle": "求嘉豪文案"}]}
            pytest.fail("An excluded source reached get-feed-detail")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(collect_candidates, "SkillClient", lambda root: Client())
    monkeypatch.setattr(collect_candidates.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        "sys.argv", ["collect", "--keywords", "嘉豪文案", "--feed-ids", feed_id, "--output-dir", "out"]
    )
    collect_candidates.main()
    report = json.loads((tmp_path / "out/report.json").read_text(encoding="utf-8"))
    assert report["posts"] == [] and report["excluded_feed_ids"] == [feed_id]


@pytest.mark.parametrize("value", [{"id": "a" * 24}, [123], ["bad-id"]])
def test_invalid_exclusions_stop_before_external_calls(tmp_path, monkeypatch, value):
    path = tmp_path / "data/rejected/collection_skip_ids.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["collect", "--output-dir", "out"])
    monkeypatch.setattr(collect_candidates, "SkillClient", lambda root: pytest.fail("Unexpected external setup"))
    with pytest.raises(ValueError, match="exclusion"):
        collect_candidates.main()


def test_feed_selection_requires_explicit_quote_context():
    assert is_quote_request("求嘉豪文案")
    assert is_quote_request("求玩阴阳师然后很嘉豪的文案")
    assert is_quote_request("开学自我介绍 嘉豪")
    assert is_quote_request("求朋友圈嘉豪文案")
    assert is_quote_request("哲学系嘉豪的日常")
    assert is_quote_request("美术嘉豪的 one day")
    assert not is_quote_request("嘉豪是什么梗？")
    assert is_quote_request("豪情在天文案自存")
    assert not is_quote_request("盘点山姆必发的小众文案", "嘉豪山姆文案")
    assert is_quote_request("求豪味文案", "嘉豪画画文案")
    assert is_quote_request("计算机嘉豪的一天")
    assert not is_quote_request("健身房的一天")
    assert not is_quote_request("洪嘉豪演唱会文案", "嘉豪音乐语录")
    assert not is_quote_request("评论区进嘉豪了")
    assert not is_quote_request("点进来看整个评论区最急之人")
    assert not is_quote_request("嘉豪速成班古风版文案")


def test_search_terms_do_not_make_unrelated_titles_eligible():
    assert not is_quote_request("🇭🇰粤语｜超有生命力的旅游歌词文案", "嘉豪旅游文案")
    assert not is_quote_request("逛超市文案🛍️🛒", "嘉豪山姆文案")
    assert is_quote_request("自存嘉豪旅行文案", "嘉豪旅游文案")


def test_jiaxin_is_a_discovery_alias_not_a_positive_label():
    assert is_quote_request("求嘉欣文案", "嘉欣")
    assert is_quote_request("嘉豪/嘉欣语录征集", "嘉豪/嘉欣")
    assert is_quote_request("求自在极意豪文案", "自在极意豪")
    assert not is_quote_request("嘉欣的个人主页", "嘉欣")
    assert not is_quote_request("旅游歌词文案", "嘉欣文案")


def test_explicit_unique_alias_allows_narrative_sources_not_glossaries():
    assert is_quote_request("被女生拒绝之后竟然领悟到了自在极意豪")
    assert is_quote_request("自在极意豪")
    assert not is_quote_request("自在极意豪是什么梗")
    assert not is_quote_request("自在极意豪是什么意思")
    assert not is_quote_request("嘉豪在食堂吃饭")


def test_parser_drops_personal_identifiers() -> None:
    payload = {
        "id": "feed-1",
        "desc": "笔记正文",
        "user": {"nickname": "private-name", "avatar": "url"},
        "comments": [{"id": "comment-1", "content": "评论正文", "user_id": "secret", "nickname": "name"}],
    }
    records = parse_feed_detail(payload)
    assert records == [
        {"text": "笔记正文", "source_type": "public_note", "source_id": "feed-1"},
        {"text": "评论正文", "source_type": "public_comment", "source_id": "feed-1:comment-1"},
    ]


def test_permission_evidence_cannot_claim_independent_or_training_approval(tmp_path):
    assert permission_metadata(None) == {}
    path = tmp_path / "permission.json"
    evidence = {"scope_confirmation_status": "awaiting_user_clarification"}
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="not been confirmed"):
        permission_metadata(str(path))
    evidence.update(
        scope_confirmation_status="confirmed_by_user",
        platforms_in_scope=["xiaohongshu_public_notes", "xiaohongshu_public_comments"],
        automated_collection_methods_in_scope=["automated_read_only_collection_via_chrome_bridge"],
    )
    path.write_text(json.dumps(evidence), encoding="utf-8")
    metadata = permission_metadata(str(path))
    assert metadata["permission_independently_verified"] is False
    assert len(metadata["permission_evidence_sha256"]) == 64
    assert "license_status" not in metadata and "quality_status" not in metadata
    evidence["automated_collection_methods_in_scope"] = []
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="this method"):
        permission_metadata(str(path))
