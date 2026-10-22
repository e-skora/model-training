"""Tests for scripts/check_dataset.py on tiny made-up files in a temporary folder."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_dataset.py"


def pair(pair_id, person, a, b, label="contradiction", kind=None, a_date="2025-01-01", b_date="2025-01-01"):
    return {"id": pair_id, "person": person, "a": {"text": a, "written_at": a_date},
            "b": {"text": b, "written_at": b_date}, "label": label, "type": kind, "reason": "Made up."}


TEST = [pair("t-1", "Zed Quorra", "Zed keeps a red kayak at the lake.", "Zed keeps a green kayak at the lake.")]
GOOD = [
    pair("x-1", "Pim Oduya", "Pim bakes rye bread on Sundays.", "Pim has never baked bread."),
    pair("x-2", "Lio Varga", "Lio rents a flat near the harbour.", "Lio moved into a flat near the market.",
         "non_contradiction", "update", b_date="2025-04-01"),
    pair("x-3", "Ude Saro", "Ude's cat might be allergic to fish.", "Ude feeds the cat salmon sometimes.", "unclear"),
]


def write(tmp_path, name, pairs):
    path = tmp_path / name
    path.write_text("".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
    return str(path)


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_good_file_passes_all_walls(tmp_path):
    result = run(write(tmp_path, "train.jsonl", GOOD), "--against", write(tmp_path, "test.jsonl", TEST),
                 "--distinct-from", write(tmp_path, "valid.jsonl", [GOOD[0] | {"id": "v-1", "a": {"text": "Pim swims.", "written_at": "2025-01-01"}}]))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "unclear: 1" in result.stdout
    assert "PASS" in result.stdout


def test_unclear_with_a_type_fails(tmp_path):
    bad = [GOOD[2] | {"type": "update"}]
    result = run(write(tmp_path, "train.jsonl", bad))
    assert result.returncode != 0 and "must be null" in result.stderr


def test_test_persons_name_fails(tmp_path):
    bad = [pair("x-9", "Pim Quorra", "Pim walks to work.", "Pim cycles to work.")]
    result = run(write(tmp_path, "train.jsonl", bad), "--against", write(tmp_path, "test.jsonl", TEST))
    assert result.returncode != 0 and "'Quorra' belongs to a person" in result.stderr


def test_near_copy_of_a_test_pair_fails(tmp_path):
    bad = [pair("x-9", "Pim Oduya", "Pim keeps a red kayak at the lake.", "Pim keeps a blue kayak at the lake.")]
    result = run(write(tmp_path, "train.jsonl", bad), "--against", write(tmp_path, "test.jsonl", TEST))
    assert result.returncode != 0 and "near copy" in result.stderr and "t-1" in result.stderr


def test_pair_repeated_in_other_file_fails_even_when_swapped(tmp_path):
    swapped = GOOD[0] | {"id": "v-1", "a": GOOD[0]["b"], "b": GOOD[0]["a"]}
    result = run(write(tmp_path, "train.jsonl", GOOD), "--distinct-from", write(tmp_path, "valid.jsonl", [swapped]))
    assert result.returncode != 0 and "repeats v-1" in result.stderr


def test_banned_name_fails(tmp_path):
    bad = [pair("x-9", "Mara Oduya", "Mara walks to work.", "Mara cycles to work.")]
    result = run(write(tmp_path, "train.jsonl", bad))
    assert result.returncode != 0 and "banned name" in result.stderr


def test_sentence_template_used_four_times_fails(tmp_path):
    people = ["Ama Kito", "Bel Rusu", "Cai Domo", "Dov Pell"]
    bad = [pair(f"x-{i}", p, f"{p.split()[0]} grew up in Tarno.", f"{p.split()[0]} has a garden.", "non_contradiction",
                "merely_related") for i, p in enumerate(people)]
    result = run(write(tmp_path, "train.jsonl", bad))
    assert result.returncode != 0 and "sentence pattern used more than 3 times" in result.stderr
