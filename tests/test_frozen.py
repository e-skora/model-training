"""Tests for judge.frozen with a temporary STATE.md and test file. No real repo files."""

import hashlib
import json

import pytest

import judge.frozen
from judge.frozen import FrozenFileError, frozen_entries, load_frozen
from judge.score import main

PAIRS = [
    {"id": "p1", "person": "Test Person", "a": {"text": "x", "written_at": "2025-01-01"},
     "b": {"text": "y", "written_at": "2025-01-02"}, "label": "contradiction", "type": None, "reason": "z"},
]


def make_repo(tmp_path):
    """A made-up repo folder: data/m1/test.jsonl and a STATE.md that freezes it."""
    test = tmp_path / "data" / "m1" / "test.jsonl"
    test.parent.mkdir(parents=True)
    test.write_text("".join(json.dumps(p) + "\n" for p in PAIRS), encoding="utf-8")
    digest = hashlib.sha256(test.read_bytes()).hexdigest()
    state = tmp_path / "STATE.md"
    state.write_text(
        "# STATE.md\n\n## Done\n- `data/m1/other.jsonl`, SHA-256 `" + "0" * 64 + "` (not frozen)\n\n"
        f"## Frozen files\n- `data/m1/test.jsonl`, 1 pair, SHA-256 `{digest}`, frozen 2025-01-03.\n\n"
        "## Open items\n- none\n",
        encoding="utf-8",
    )
    return test, state


def test_frozen_entries_reads_only_the_frozen_section(tmp_path):
    test, state = make_repo(tmp_path)
    assert list(frozen_entries(state)) == ["data/m1/test.jsonl"]


def test_load_frozen_returns_pairs_when_fingerprint_matches(tmp_path):
    test, state = make_repo(tmp_path)
    assert load_frozen(test, state) == PAIRS


def test_one_changed_byte_stops_loading(tmp_path):
    test, state = make_repo(tmp_path)
    data = bytearray(test.read_bytes())
    data[data.index(b"x")] = ord("w")
    test.write_bytes(bytes(data))
    with pytest.raises(FrozenFileError, match="not the frozen"):
        load_frozen(test, state)


def test_file_not_listed_as_frozen_stops_loading(tmp_path):
    test, state = make_repo(tmp_path)
    other = tmp_path / "data" / "m1" / "other.jsonl"
    other.write_bytes(test.read_bytes())
    with pytest.raises(FrozenFileError, match="not listed"):
        load_frozen(other, state)


def test_scorer_command_stops_before_scoring_on_a_changed_byte(tmp_path, monkeypatch, capsys):
    test, state = make_repo(tmp_path)
    monkeypatch.setattr(judge.frozen, "REPO", tmp_path)
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"id": "p1", "raw": '{"label": "contradiction", "type": null, "reason": "r"}'}) + "\n")

    main(["score", str(test), str(answers)])  # unchanged file: scores print
    assert '"precision": 1.0' in capsys.readouterr().out

    test.write_bytes(test.read_bytes().replace(b'"y"', b'"Y"'))
    with pytest.raises(SystemExit, match="STOP: data/m1/test.jsonl SHA-256"):
        main(["score", str(test), str(answers)])
    assert capsys.readouterr().out == ""  # no scores printed
