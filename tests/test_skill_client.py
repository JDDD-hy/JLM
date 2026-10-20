import json
from types import SimpleNamespace

import pytest

from src.collection.skill_client import SkillClient, sanitize_response


def test_response_scrubbing_retains_source_ids_but_not_accounts_or_tokens():
    payload = {
        "id": "note1",
        "xsecToken": "secret",
        "user": {"nickname": "personal"},
        "comments": [{"id": "comment1", "user_id": "account", "content": "保留正文"}],
        "url": "https://example.com/path?xsec_token=secret",
    }
    assert sanitize_response(payload) == {
        "id": "note1",
        "comments": [{"id": "comment1", "content": "保留正文"}],
        "url": "https://example.com/path",
    }


def test_client_rejects_any_write_command_before_execution(tmp_path):
    client = object.__new__(SkillClient)
    with pytest.raises(ValueError, match="read-only"):
        client.call("post-comment", "--content", "blocked")


def test_expanded_comments_have_bounded_longer_timeout(monkeypatch):
    timeouts = []

    def run(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return SimpleNamespace(returncode=0, stdout="{}")

    monkeypatch.setattr("src.collection.skill_client.subprocess.run", run)
    client = object.__new__(SkillClient)
    client.python, client.cli = "python", "cli.py"
    client.call("search-feeds", "--keyword", "test")
    client.call("get-feed-detail", "--load-all-comments", "--max-comment-items", "50")
    assert timeouts == [100, 300]


@pytest.mark.parametrize(
    ("error", "hint"),
    [("Bridge timeout", "timeout"), ("需要验证码", "verification"), ("unrecognized failure", "unknown")],
)
def test_failed_cli_retains_safe_diagnostic_hint_not_raw_secrets(monkeypatch, error, hint):
    result = SimpleNamespace(
        returncode=2, stdout=json.dumps({"error": error + " xsec_token=secret"}), stderr="Cookie: private"
    )
    monkeypatch.setattr("src.collection.skill_client.subprocess.run", lambda *a, **k: result)
    client = object.__new__(SkillClient)
    client.python, client.cli = "python", "cli.py"
    with pytest.raises(RuntimeError) as caught:
        client.call("search-feeds", "--keyword", "test")
    assert f"diagnostic_hint={hint}" in str(caught.value)
    assert "secret" not in str(caught.value) and "private" not in str(caught.value)
