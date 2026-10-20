import json
import sys

from scripts.import_data import main


def test_import_does_not_equate_no_synthetic_flag_with_human_authorship(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"quote": "synthetic test fixture", "scene": None}), encoding="utf-8")
    target = tmp_path / "imported"
    monkeypatch.setattr(sys, "argv", ["import_data.py", str(source), "--output-dir", str(target)])
    main()
    row = json.loads((target / "validated_records.jsonl").read_text(encoding="utf-8"))
    assert row["origin"] == "unknown"
    assert row["license_status"] == "pending_review"
