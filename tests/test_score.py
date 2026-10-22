"""Tests for judge.score with small made-up answers. No GPU, no network."""

import json

import pytest

from judge.prompt import SYSTEM_MESSAGE, build_messages
from judge.score import load_jsonl, parse_answer, score


def pair(pair_id, label, kind=None):
    return {
        "id": pair_id,
        "person": "Test Person",
        "a": {"text": "Test Person likes tea.", "written_at": "2025-01-01"},
        "b": {"text": "Test Person likes coffee.", "written_at": "2025-01-02"},
        "label": label,
        "type": kind,
        "reason": "Made up for a test.",
    }


def said(pair_id, label, kind=None):
    return {"id": pair_id, "raw": json.dumps({"label": label, "type": kind, "reason": "x"})}


PAIRS = [
    pair("p1", "contradiction"),
    pair("p2", "contradiction"),
    pair("p3", "contradiction"),
    pair("p4", "non_contradiction", "update"),
    pair("p5", "non_contradiction", "compatible_detail"),
    pair("p6", "non_contradiction", "merely_related"),
]


def test_perfect_answers_score_one():
    answers = [said(p["id"], p["label"], p["type"]) for p in PAIRS]
    metrics, _ = score(PAIRS, answers)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["type_accuracy"] == 1.0
    assert metrics["invalid"] == 0


def test_precision_recall_accuracy_and_type_accuracy():
    answers = [
        said("p1", "contradiction"),  # caught
        said("p2", "contradiction"),  # caught
        said("p3", "non_contradiction", "update"),  # missed
        said("p4", "contradiction"),  # false alarm
        said("p5", "non_contradiction", "compatible_detail"),  # label and type right
        said("p6", "non_contradiction", "update"),  # label right, type wrong
    ]
    metrics, _ = score(PAIRS, answers)
    assert metrics["precision"] == pytest.approx(2 / 3)  # 2 right of 3 "contradiction" answers
    assert metrics["recall"] == pytest.approx(2 / 3)  # 2 caught of 3 real ones
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["type_accuracy"] == pytest.approx(1 / 2)  # only p5 and p6 count
    assert metrics["type_accuracy_of"] == 2


def test_invalid_answers_count_as_wrong():
    answers = [
        {"id": "p1", "raw": "contradiction"},  # not JSON
        {"id": "p2", "raw": json.dumps({"label": "conflict", "type": None})},  # unknown label
        {"id": "p3", "raw": json.dumps(["contradiction"])},  # JSON but not an object
        said("p4", "non_contradiction", "update"),
        said("p5", "non_contradiction", "compatible_detail"),
        # p6 has no answer at all
    ]
    metrics, rows = score(PAIRS, answers)
    assert metrics["invalid"] == 4
    assert metrics["invalid_not_json"] == 3
    assert metrics["invalid_unknown_label"] == 1
    assert metrics["recall"] == 0.0
    assert metrics["precision"] is None  # never said "contradiction"
    assert metrics["accuracy"] == pytest.approx(2 / 6)
    assert [r["problem"] for r in rows][:3] == ["not valid JSON", "unknown label", "not valid JSON"]


def test_fenced_json_is_accepted_and_counted():
    raw = '```json\n{"label": "contradiction", "type": null, "reason": "x"}\n```'
    answer, problem, fenced = parse_answer(raw)
    assert problem is None
    assert fenced is True
    assert answer["label"] == "contradiction"


def test_text_around_json_is_invalid():
    answer, problem, _ = parse_answer('Sure! {"label": "contradiction", "type": null}')
    assert answer is None
    assert problem == "not valid JSON"


def test_load_jsonl_reads_answers_file(tmp_path):
    path = tmp_path / "answers.jsonl"
    lines = [said("p1", "contradiction"), said("p2", "contradiction")]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    assert load_jsonl(path) == lines


def test_build_messages_has_system_and_pair_without_label():
    messages = build_messages(PAIRS[0])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_MESSAGE
    user = messages[1]["content"]
    assert "Test Person" in user
    assert "2025-01-01" in user and "2025-01-02" in user
    assert "Made up for a test" not in user
