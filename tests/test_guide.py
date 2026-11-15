"""Offline tests of the JSON schema and citation guard; Ollama is not needed."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rag import RAGError, answer_question, load_guide

ROOT = Path(__file__).resolve().parents[1]


def test_guide_schema():
    guide = load_guide((ROOT / "knowledge_base.json").read_text(encoding="utf-8"))
    assert 12 <= len(guide) <= 20
    assert len({entry["id"] for entry in guide}) == len(guide)
    assert all(entry.get("category") and entry.get("tags") for entry in guide)


def test_reject_duplicate_id():
    entries = json.loads((ROOT / "knowledge_base.json").read_text(encoding="utf-8"))
    entries[1]["id"] = entries[0]["id"]
    try:
        load_guide(json.dumps(entries))
    except RAGError as exc:
        assert "Duplicate ID" in str(exc)
    else:
        raise AssertionError("Duplicate ID should fail")


def test_unknown_citation_is_blocked():
    guide = load_guide((ROOT / "knowledge_base.json").read_text(encoding="utf-8"))
    vectors = np.ones((len(guide), 3))
    with patch("rag.retrieve", return_value=[{**guide[0], "score": 0.8}]), patch(
        "rag.generate", return_value="Invented price is 5 VND [G99]."
    ):
        result = answer_question("fake question", (guide, vectors), "http://localhost:11434")
    assert result["answer"].startswith("I don't know")
