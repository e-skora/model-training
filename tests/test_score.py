"""Tests for judge.score with small made-up answers. No GPU, no network."""

import json

import pytest

from judge.prompt import SYSTEM_MESSAGE, build_messages, prompt_sha256
from judge.score import (
    VERSIONS,
    derive,
    format_problem,
    load_jsonl,
    parse_answer,
    score,
    score_by_trap,
    score_versions,
    side_by_side,
    trap_table,
)


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


def test_each_kind_of_invalid_answer_is_counted():
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


def test_unclear_is_valid_but_never_a_yes():
    # D-025: "unclear" never enters precision, lowers recall on a real contradiction, and is counted.
    answers = [
        said("p1", "contradiction"),  # caught
        said("p2", "unclear"),  # real contradiction, not caught
        said("p3", "contradiction"),  # caught
        said("p4", "unclear"),  # real update, wrong label but no false alarm
        said("p5", "non_contradiction", "compatible_detail"),
        said("p6", "non_contradiction", "merely_related"),
    ]
    metrics, rows = score(PAIRS, answers)
    assert metrics["unclear"] == 2
    assert metrics["invalid"] == 0
    assert metrics["format_errors"] == 0
    assert metrics["precision"] == 1.0  # the two "unclear" answers are not counted as "contradiction"
    assert metrics["recall"] == pytest.approx(2 / 3)  # p2 is missed
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert rows[1]["label"] == "unclear" and rows[1]["problem"] is None


def test_unclear_right_answer_counts_for_accuracy_and_its_false_alarm_for_precision():
    pairs = [pair("u1", "unclear"), pair("u2", "unclear"), pair("c1", "contradiction")]
    answers = [said("u1", "unclear"), said("u2", "contradiction"), said("c1", "contradiction")]
    metrics, _ = score(pairs, answers)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(1 / 2)  # u2 is a false alarm
    assert metrics["recall"] == 1.0


@pytest.mark.parametrize(
    "answer, problem",
    [
        ({"label": "unclear", "type": None, "reason": "x"}, None),
        ({"label": "unclear", "type": "update", "reason": "x"}, "type not null for unclear"),
    ],
)
def test_format_problem_unclear(answer, problem):
    assert format_problem(answer) == problem


def test_derived_versions_swap_or_drop_dates_without_touching_the_original():
    original = [dict(PAIRS[0], a=dict(PAIRS[0]["a"]), b=dict(PAIRS[0]["b"]))]
    before = json.dumps(original, sort_keys=True)

    assert derive(original, "as_is") == original
    swapped = derive(original, "swapped")[0]
    assert swapped["a"] == original[0]["b"] and swapped["b"] == original[0]["a"]
    assert swapped["label"] == original[0]["label"]
    no_dates = derive(original, "no_dates")[0]
    assert no_dates["a"]["written_at"] is None and no_dates["b"]["written_at"] is None
    assert no_dates["a"]["text"] == original[0]["a"]["text"]

    assert json.dumps(original, sort_keys=True) == before  # the loaded test set is never changed
    with pytest.raises(ValueError, match="unknown version"):
        derive(original, "reversed")


def test_derived_versions_build_the_right_prompts():
    swapped = build_messages(derive([PAIRS[0]], "swapped")[0])[1]["content"]
    assert swapped.index("likes coffee") < swapped.index("likes tea")
    assert "Memory A (written 2025-01-02)" in swapped
    no_dates = build_messages(derive([PAIRS[0]], "no_dates")[0])[1]["content"]
    assert "written" not in no_dates and "2025" not in no_dates
    assert "Memory A: Test Person likes tea." in no_dates


def test_score_versions_scores_each_version_against_the_same_labels():
    right = [said(p["id"], p["label"], p["type"]) for p in PAIRS]
    flipped = [said(p["id"], "non_contradiction" if p["label"] == "contradiction" else "contradiction", "update")
               for p in PAIRS]
    metrics = score_versions(PAIRS, {"as_is": right, "swapped": flipped, "no_dates": right})
    assert list(metrics) == ["as_is", "swapped", "no_dates"]
    assert metrics["as_is"]["accuracy"] == 1.0
    assert metrics["swapped"]["accuracy"] == 0.0
    table = side_by_side(metrics)
    assert table.splitlines()[0].split() == ["metric", "as_is", "swapped", "no_dates"]
    assert any(line.split() == ["accuracy", "1.000", "0.000", "1.000"] for line in table.splitlines())
    assert any(line.split()[0] == "unclear" for line in table.splitlines())


def test_main_scores_three_versions_side_by_side(tmp_path, monkeypatch, capsys):
    import judge.score as score_module

    files = []
    for version in VERSIONS:
        path = tmp_path / f"{version}.jsonl"
        path.write_text("\n".join(json.dumps(said(p["id"], p["label"], p["type"])) for p in PAIRS) + "\n")
        files.append(str(path))
    monkeypatch.setattr(score_module, "load_frozen", lambda path: PAIRS)
    score_module.main(["score", "test.jsonl", *files])
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["metric", "as_is", "swapped", "no_dates"]


def test_build_messages_without_dates_and_prompt_mentions_aeverum_rules():
    user = build_messages(PAIRS[0], dates=False)[1]["content"]
    assert "2025" not in user
    assert user.splitlines() == [
        "Person: Test Person",
        "Memory A: Test Person likes tea.",
        "Memory B: Test Person likes coffee.",
    ]
    assert '"unclear"' in SYSTEM_MESSAGE
    assert "order of the two memories means nothing" in SYSTEM_MESSAGE
    assert "dates can be missing" in SYSTEM_MESSAGE


def test_prompt_sha256_is_fixed_and_changes_with_the_prompt(monkeypatch):
    import judge.prompt as prompt_module

    first = prompt_sha256()
    assert len(first) == 64 and first == prompt_sha256()
    monkeypatch.setattr(prompt_module, "SYSTEM_MESSAGE", SYSTEM_MESSAGE + " ")
    assert prompt_module.prompt_sha256() != first


# M2 hard test set: per-trap scoring. Tiny made-up pairs, no files.
HARD_PAIRS = [
    pair("h1", "contradiction") | {"hard": "explain_away"},
    pair("h2", "contradiction") | {"hard": "explain_away"},
    pair("h3", "contradiction") | {"hard": "paraphrase"},
    pair("h4", "non_contradiction", "merely_related") | {"hard": "rule5_bait"},
    pair("h5", "non_contradiction", "merely_related") | {"hard": "near_identical"},
    pair("h6", "non_contradiction", "compatible_detail") | {"hard": "near_identical"},
]
HARD_ANSWERS = [
    said("h1", "contradiction"),
    said("h2", "unclear"),
    {"id": "h3", "raw": "not json"},
    said("h4", "contradiction"),
    said("h5", "non_contradiction", "merely_related"),
    said("h6", "non_contradiction", "compatible_detail"),
]


def test_score_by_trap_reports_precision_recall_and_unclear_per_trap():
    traps = score_by_trap(HARD_PAIRS, HARD_ANSWERS)
    assert list(traps) == ["explain_away (contradiction)", "paraphrase (contradiction)", "rule5_bait (merely_related)",
                           "near_identical (merely_related)", "near_identical (compatible_detail)"]
    explain = traps["explain_away (contradiction)"]
    assert (explain["pairs"], explain["precision"], explain["recall"], explain["unclear"]) == (2, 1.0, 0.5, 1)
    paraphrase = traps["paraphrase (contradiction)"]
    assert (paraphrase["precision"], paraphrase["recall"], paraphrase["invalid"]) == (None, 0.0, 1)
    bait = traps["rule5_bait (merely_related)"]  # a false alarm: precision 0, no recall to speak of
    assert (bait["precision"], bait["recall"], bait["accuracy"]) == (0.0, None, 0.0)
    assert traps["near_identical (compatible_detail)"]["accuracy"] == 1.0
    table = trap_table(traps)
    assert "explain_away (contradiction)" in table and "unclear" in table.splitlines()[0]


def test_score_by_trap_needs_a_tag_on_every_pair_and_known_ids():
    untagged = HARD_PAIRS[:2] + [pair("h9", "contradiction")]
    with pytest.raises(ValueError, match="without a 'hard' tag"):
        score_by_trap(untagged, [])
    with pytest.raises(ValueError, match="not in the test set"):
        score_by_trap(HARD_PAIRS, [said("zz", "contradiction")])


def test_rows_carry_the_trap_only_for_tagged_pairs():
    _, rows = score(HARD_PAIRS, HARD_ANSWERS)
    assert rows[0]["hard"] == "explain_away"
    _, m1_rows = score(PAIRS, [])
    assert all("hard" not in row for row in m1_rows)


def test_trap_tag_never_reaches_the_model():
    tagged = HARD_PAIRS[0]
    plain = {k: v for k, v in tagged.items() if k != "hard"}
    for dates in (True, False):
        assert build_messages(tagged, dates) == build_messages(plain, dates)
        assert "explain_away" not in json.dumps(build_messages(tagged, dates))


# The M1 saved answers, rescored with today's code, give exactly the numbers
# recorded in STATE.md and the M1 summaries. Reads repo files; writes nothing.
M1_RESCORE = {
    # file: (precision, recall, accuracy, unclear, invalid)
    "gguf-v1/m1-judge-v1.as_is.jsonl": (0.9795918367346939, 0.96, 0.965, 2, 0),
    "gguf-v1/m1-judge-v1.swapped.jsonl": (0.9411764705882353, 0.96, 0.95, 2, 0),
    "gguf-v1/m1-judge-v1.no_dates.jsonl": (0.9791666666666666, 0.94, 0.96, 1, 0),
    "trained-v1/qwen3-4b-trained-v1.as_is.jsonl": (0.9895833333333334, 0.95, 0.965, 3, 0),
    "trained-v1/qwen3-4b-trained-v1.swapped.jsonl": (0.9504950495049505, 0.96, 0.94, 6, 0),
    "trained-v1/qwen3-4b-trained-v1.no_dates.jsonl": (0.9782608695652174, 0.9, 0.94, 3, 0),
    "baseline/qwen3-8b.jsonl": (0.9591836734693877, 0.94, 0.95, 0, 0),
}


@pytest.mark.parametrize("name", sorted(M1_RESCORE))
def test_m1_saved_answers_rescore_to_the_same_numbers(name):
    from pathlib import Path

    from judge.frozen import load_frozen

    repo = Path(__file__).resolve().parents[1]
    metrics, _ = score(load_frozen(repo / "data/m1/test.jsonl"), load_jsonl(repo / "results/m1" / name))
    assert (metrics["precision"], metrics["recall"], metrics["accuracy"], metrics["unclear"], metrics["invalid"]) == M1_RESCORE[name]
