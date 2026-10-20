import json
import sys

from scripts.import_collection import main


def test_saved_import_scrubs_profiles_and_does_not_invent_authorship(tmp_path, monkeypatch):
    source = tmp_path / "posts"
    source.mkdir()
    (source / "one.json").write_text(
        json.dumps(
            {
                "note": {
                    "noteId": "synthetic-note",
                    "desc": "synthetic 测试文本",
                    "user": {"nickname": "private"},
                    "xsecToken": "secret",
                },
                "comments": [],
            }
        ),
        encoding="utf-8",
    )
    target = tmp_path / "import"
    monkeypatch.setattr(sys, "argv", ["import_collection.py", "--input-dir", str(source), "--output-dir", str(target)])
    main()
    main()
    archives = list((target / "responses").glob("*.json"))
    assert len(archives) == 1
    archived = json.loads(archives[0].read_text(encoding="utf-8"))
    assert "user" not in archived["note"] and "xsecToken" not in archived["note"]
    row = json.loads((target / "candidates.jsonl").read_text(encoding="utf-8"))
    assert row["origin"] == "unknown"
    assert row["quality_status"] == "unreviewed"
    assert row["license_status"] == "pending_review"
    assert row["source_group"] == "synthetic-note"
    assert json.loads((target / "report.json").read_text())["new_responses"] == 0
