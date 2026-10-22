"""Tests for judge.score with small made-up answers. No GPU, no network."""

import json

import pytest

from judge.prompt import SYSTEM_MESSAGE, build_messages
from judge.score import format_problem, load_jsonl, parse_answer, score


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


def test_invalid_answers_lower_recall_and_accuracy_not_precision():
    # D-024: an invalid answer is neither "yes" nor "no".
    pairs = [pair(f"c{i}", "contradiction") for i in range(1, 5)] + [
        pair("n1", "non_contradiction", "update"),
        pair("n2", "non_contradiction", "update"),
        pair("n3", "non_contradiction", "merely_related"),
        pair("n4", "non_contradiction", "compatible_detail"),
    ]
    answers = [
        said("c1", "contradiction"),  # caught
        said("c2", "contradiction"),  # caught
        {"id": "c3", "raw": "not json"},  # invalid, right label is contradiction
        said("c4", "non_contradiction", "update"),  # missed
        said("n1", "contradiction"),  # false alarm
        {"id": "n2", "raw": json.dumps({"label": "update", "type": None})},  # invalid, right label is non-contradiction
        said("n3", "non_contradiction", "merely_related"),  # right
        said("n4", "non_contradiction", "compatible_detail"),  # right
    ]
    metrics, _ = score(pairs, answers)
    assert metrics["precision"] == pytest.approx(2 / 3)  # 2 caught of 3 "contradiction" answers; invalid ones not counted
    assert metrics["recall"] == pytest.approx(2 / 4)  # c3 (invalid) and c4 (missed) both lower recall
    assert metrics["accuracy"] == pytest.approx(4 / 8)  # c1, c2, n3, n4; both invalid answers count as wrong
    assert metrics["invalid"] == 2

    # Leaving the two invalid answers out gives the same scores: an invalid
    # answer is treated like no answer, never like "contradiction".
    valid_only = [a for a in answers if a["id"] not in ("c3", "n2")]
    metrics_valid, _ = score(pairs, valid_only)
    for name in ("precision", "recall", "accuracy"):
        assert metrics_valid[name] == metrics[name]


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


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_json_constants_are_invalid(constant):
    answer, problem, _ = parse_answer('{"label": "contradiction", "type": null, "reason": %s}' % constant)
    assert answer is None
    assert problem == "not valid JSON"


def test_repeated_field_names_are_invalid():
    raw = '{"label": "contradiction", "label": "non_contradiction", "type": null, "reason": "x"}'
    answer, problem, _ = parse_answer(raw)
    assert answer is None
    assert problem == "repeated field"
    metrics, _ = score([pair("p1", "non_contradiction", "update")], [{"id": "p1", "raw": raw}])
    assert metrics["invalid"] == 1
    assert metrics["invalid_repeated_field"] == 1
    assert metrics["accuracy"] == 0.0  # not scored as the second label


@pytest.mark.parametrize(
    "answer, problem",
    [
        ({"label": "contradiction", "type": None, "reason": "x"}, None),
        ({"label": "non_contradiction", "type": "update", "reason": "x"}, None),
        ({"label": "contradiction", "type": "null", "reason": "x"}, "type not null for contradiction"),
        ({"label": "contradiction", "type": "update", "reason": "x"}, "type not null for contradiction"),
        ({"label": "contradiction", "reason": "x"}, "no type"),
        ({"label": "non_contradiction", "type": "other", "reason": "x"}, "unknown type"),
        ({"label": "non_contradiction", "type": None, "reason": "x"}, "unknown type"),
        ({"label": "contradiction", "type": None}, "no reason text"),
        ({"label": "contradiction", "type": None, "reason": "  "}, "no reason text"),
        ({"label": "contradiction", "type": None, "reason": 3}, "no reason text"),
    ],
)
def test_format_problem(answer, problem):
    assert format_problem(answer) == problem


def test_format_errors_are_counted_but_do_not_change_label_scores():
    answers = [
        {"id": "p1", "raw": json.dumps({"label": "contradiction", "type": "null", "reason": "x"})},
        {"id": "p2", "raw": json.dumps({"label": "contradiction"})},
        said("p3", "contradiction"),
        {"id": "p4", "raw": json.dumps({"label": "non_contradiction", "type": "other", "reason": "x"})},
        said("p5", "non_contradiction", "compatible_detail"),
        said("p6", "non_contradiction", "merely_related"),
    ]
    metrics, rows = score(PAIRS, answers)
    assert metrics["format_errors"] == 3
    assert metrics["invalid"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["type_accuracy"] == pytest.approx(2 / 3)  # p4's made-up type is a wrong type
    assert [r["format_problem"] for r in rows][:2] == ["type not null for contradiction", "no type"]


def test_repeated_answer_ids_stop_scoring():
    answers = [said("p1", "contradiction"), said("p1", "non_contradiction", "update")]
    with pytest.raises(ValueError, match=r"more than one answer for pair\(s\) \['p1'\]"):
        score(PAIRS, answers)


def test_unknown_answer_ids_stop_scoring():
    answers = [said("p1", "contradiction"), said("zz", "contradiction")]
    with pytest.raises(ValueError, match=r"not in the test set: \['zz'\]"):
        score(PAIRS, answers)


def test_missing_answers_still_count_as_wrong():
    metrics, rows = score(PAIRS, [said("p1", "contradiction")])
    assert metrics["accuracy"] == pytest.approx(1 / 6)
    assert metrics["invalid"] == 5
    assert rows[1]["problem"] == "no answer"
