from __future__ import annotations

from pathlib import Path

from src.evaluation.io import load_jsonl, write_jsonl_record


def test_write_and_load_jsonl(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    write_jsonl_record(path, {"question": "hello", "passed": True})
    records = load_jsonl(path)
    assert len(records) == 1
    assert records[0]["passed"] is True