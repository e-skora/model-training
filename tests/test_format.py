"""Tests for judge.format with made-up pairs. No GPU, no network."""

import json

from judge.format import SEED, answer_text, format_pairs, to_example
from judge.prompt import SYSTEM_MESSAGE


def pair(i, label="non_contradiction", kind="update"):
    return {
        "id": f"x-{i:04d}",
        "person": "Test Person",
        "a": {"text": f"Test Person lives in town {i}.", "written_at": "2025-01-01"},
        "b": {"text": f"Test Person moved to city {i}.", "written_at": "2025-03-01"},
        "label": label,
        "type": kind,
        "reason": "The newer memory says the move happened.",
    }


PAIRS = [pair(i) for i in range(300)]


def test_example_is_prompt_plus_right_answer():
    example = to_example(pair(1))
    roles = [m["role"] for m in example["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert example["messages"][0]["content"] == SYSTEM_MESSAGE
    assert json.loads(example["messages"][2]["content"]) == {
        "label": "non_contradiction",
        "type": "update",
        "reason": "The newer memory says the move happened.",
    }


def test_unclear_answer_has_null_type():
    text = answer_text(pair(1, "unclear", None))
    assert text == '{"label": "unclear", "type": null, "reason": "The newer memory says the move happened."}'


def test_swap_shows_later_memory_first_and_keeps_the_answer():
    plain = to_example(pair(1))["messages"]
    swapped = to_example(pair(1), swap=True)["messages"]
    user = swapped[1]["content"].splitlines()
    assert user[1] == "Memory A (written 2025-03-01): Test Person moved to city 1."
    assert user[2] == "Memory B (written 2025-01-01): Test Person lives in town 1."
    assert swapped[2] == plain[2]


def test_no_dates_leaves_both_dates_out():
    user = to_example(pair(1), dates=False)["messages"][1]["content"]
    assert "written" not in user and "2025" not in user


def test_shares_are_about_half_swapped_and_a_third_without_dates():
    examples = format_pairs(PAIRS)
    assert sum(e["swapped"] for e in examples) == 150
    assert sum(not e["dates"] for e in examples) == 100
    # The two choices are made independently, so all four mixes occur.
    mixes = {(e["swapped"], e["dates"]) for e in examples}
    assert mixes == {(True, True), (True, False), (False, True), (False, False)}
    assert [e["id"] for e in examples] == [p["id"] for p in PAIRS]


def test_same_seed_same_examples_and_input_unchanged():
    before = json.dumps(PAIRS)
    assert format_pairs(PAIRS) == format_pairs(PAIRS, seed=SEED)
    assert format_pairs(PAIRS) != format_pairs(PAIRS, seed=SEED + 1)
    assert json.dumps(PAIRS) == before


def test_main_writes_examples(tmp_path, capsys):
    from judge.format import main

    source = tmp_path / "pairs.jsonl"
    source.write_text("\n".join(json.dumps(p) for p in PAIRS[:6]) + "\n", encoding="utf-8")
    out = tmp_path / "examples.jsonl"
    main(["format", str(source), "--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert "6 examples, 3 swapped, 2 without dates" in capsys.readouterr().out
