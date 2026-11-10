"""Tests for scripts/compare_v2.py, the D-060 rule. Made-up pairs and answers only; no frozen test file is opened."""

import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pytest

from judge.frozen import FrozenFileError
from judge.score import VERSIONS

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_v2_script", REPO / "scripts" / "compare_v2.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)


def c(caught, said, real, invalid=0, breaks=None, unclear=0):
    return {"pairs": 0, "caught": caught, "said": said, "real": real, "unclear": unclear, "invalid": invalid,
            "format_errors": 0, "style_a_breaks": invalid if breaks is None else breaks}


def same(k):
    return {v: dict(k) for v in VERSIONS}


# v1 as D-060 records it on the hard set: worst-case recall 134/150, worst-case precision 134/142.
V1_HARD = {"as_is": c(136, 142, 150), "swapped": c(134, 142, 150), "no_dates": c(135, 140, 150)}
V1_M1 = {"as_is": c(90, 92, 100), "swapped": c(88, 90, 100), "no_dates": c(85, 88, 100)}


def verdict(v2_hard, v2_m1=None):
    return cmp.judge({"hard": V1_HARD, "m1": V1_M1}, {"hard": v2_hard, "m1": v2_m1 or V1_M1})


def test_item_1_needs_139_of_150_worst_case_recall():
    # 139/150 is the least recall that is at least 134/150 + 0.03; precision kept at 1.
    assert verdict(same(c(139, 139, 150)))["items"]["1"]["passed"]
    assert not verdict(same(c(138, 138, 150)))["items"]["1"]["passed"]


def test_item_1_worst_case_is_the_lowest_version():
    v2 = same(c(145, 145, 150))
    v2["no_dates"] = c(138, 138, 150)
    assert not verdict(v2)["items"]["1"]["passed"]


def test_item_1_precision_may_drop_by_0_01_and_no_more():
    floor = Fraction(134, 142) - Fraction(1, 100)  # about 0.9337
    # 140 caught: 140/149 = 0.9396 passes, 140/150 = 0.9333 misses.
    assert Fraction(140, 149) >= floor > Fraction(140, 150)
    assert verdict(same(c(140, 149, 150)))["items"]["1"]["passed"]
    result = verdict(same(c(140, 150, 150)))["items"]["1"]
    assert not result["passed"]
    assert [check["passed"] for check in result["checks"]] == [True, False]


def test_item_1_uses_exact_fractions_at_the_boundary():
    # v1 worst recall 1/2; v2 at exactly 1/2 + 3/100 = 53/100 passes (a float sum would round).
    v1 = same(c(50, 50, 100))
    assert cmp.item_1(v1, same(c(53, 53, 100)))["passed"]
    assert not cmp.item_1(v1, same(c(52, 52, 100)))["passed"]


def test_item_2_is_judged_in_each_version():
    v2 = {v: dict(k) for v, k in V1_M1.items()}
    assert verdict(same(c(145, 145, 150)), v2)["items"]["2"]["passed"]
    # no_dates recall 85/100 to 82/100 is 0.03 lower: a miss, though other versions are higher.
    v2 = {"as_is": c(99, 99, 100), "swapped": c(99, 99, 100), "no_dates": c(82, 85, 100)}
    result = verdict(same(c(145, 145, 150)), v2)["items"]["2"]
    assert not result["passed"]
    assert [ch["what"] for ch in result["checks"] if not ch["passed"]] == ["M1 no_dates recall"]


def test_item_2_allows_exactly_0_02_lower():
    v2 = {v: c(k["caught"] - 2, k["said"] - 2, 100) for v, k in V1_M1.items()}  # recall exactly 0.02 lower
    assert all(cmp.recall(v2[v]) == cmp.recall(V1_M1[v]) - Fraction(2, 100) for v in VERSIONS)
    assert verdict(same(c(145, 145, 150)), v2)["items"]["2"]["passed"]


def test_item_3_one_invalid_answer_anywhere_misses():
    good = same(c(145, 145, 150))
    assert verdict(good)["items"]["3"]["passed"]
    m1 = {v: dict(k) for v, k in V1_M1.items()}
    m1["swapped"]["invalid"] = 1
    result = verdict(good, m1)
    assert not result["items"]["3"]["passed"] and not result["passed"]


def test_format_breaks_are_reported_but_not_judged():
    hard = same(c(145, 145, 150, breaks=2))
    result = verdict(hard)
    assert result["items"]["3"]["passed"] and result["passed"]
    assert sum(ch["style_a_breaks"] for ch in result["items"]["3"]["checks"]) == 6


def test_no_contradiction_answers_count_as_zero_precision():
    assert cmp.precision(c(0, 0, 10)) == 0


def test_overall_passes_only_when_every_item_passes():
    assert verdict(same(c(145, 145, 150)))["passed"]
    assert not verdict(same(c(138, 138, 150)))["passed"]


# End to end: made-up frozen files, made-up answer files.

def pair(i, hard):
    contra = i % 2 == 0
    p = {"id": f"t-{i:03d}", "person": "Test Person",
         "a": {"text": f"Test Person owns {i} cats.", "written_at": "2025-01-01"},
         "b": {"text": f"Test Person owns {i + 1} cats.", "written_at": "2025-01-02"},
         "label": "contradiction" if contra else "non_contradiction",
         "type": None if contra else "merely_related", "reason": "Made up."}
    if hard:
        p["hard"] = "near_identical"
    return p


def answer(p, label=None, reason="Made up."):
    label = label or p["label"]
    return {"id": p["id"], "raw": json.dumps({"label": label, "type": None if label == "contradiction"
                                              else "merely_related", "reason": reason})}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_repo(tmp_path, v2_answer=answer):
    repo = tmp_path / "repo"
    sets = {"hard": [pair(i, True) for i in range(10)], "m1": [pair(i, False) for i in range(10)]}
    paths = {"hard": repo / "data/hard.jsonl", "m1": repo / "data/m1.jsonl"}
    digests = {name: write_jsonl(paths[name], sets[name]) for name in sets}
    (repo / "STATE.md").write_text("## Frozen files\n" + "".join(
        f"- `data/{n}.jsonl`, SHA-256 `{digests[n]}`\n" for n in ("hard", "m1")), encoding="utf-8")
    files = {}
    for model, make in (("v1", answer), ("v2", v2_answer)):
        files[model] = {}
        for name, pairs in sets.items():
            folder = repo / model / name
            for version in VERSIONS:
                write_jsonl(folder / f"{model}.{version}.jsonl", [make(p) for p in pairs])
            files[model][name] = (folder, model)
    return repo, {n: str(p) for n, p in paths.items()}, files


def run(repo, sets, files):
    return cmp.compare(sets, files["v1"], files["v2"], state_path=repo / "STATE.md")


def test_equal_answers_miss_item_1_and_pass_the_rest(tmp_path):
    result = run(*make_repo(tmp_path))
    items = result["verdict"]["items"]
    assert (items["1"]["passed"], items["2"]["passed"], items["3"]["passed"]) == (False, True, True)
    assert result["counts"]["v2"]["hard"]["as_is"] == {"pairs": 10, "caught": 5, "said": 5, "real": 5,
                                                       "unclear": 0, "invalid": 0, "format_errors": 0,
                                                       "style_a_breaks": 0}
    assert "Overall: MISS" in cmp.report(result)
    every = {f"{m} {v}": 5 for m in ("v1", "v2") for v in VERSIONS}
    assert result["traps"] == {"near_identical (contradiction)": {"pairs": 5, **every},
                               "near_identical (merely_related)": {"pairs": 5, **every}}


def test_an_unreadable_answer_is_invalid_and_a_missing_reason_is_a_break(tmp_path):
    def v2(p):
        if p["id"] == "t-001":
            return {"id": p["id"], "raw": "not json"}
        if p["id"] == "t-003":
            return answer(p, reason="")
        return answer(p)

    result = run(*make_repo(tmp_path, v2))
    k = result["counts"]["v2"]["m1"]["as_is"]
    assert (k["invalid"], k["format_errors"], k["style_a_breaks"]) == (1, 1, 2)
    assert not result["verdict"]["items"]["3"]["passed"]


def test_an_edited_test_file_stops_before_any_score(tmp_path):
    repo, sets, files = make_repo(tmp_path)
    path = Path(sets["hard"])
    path.write_text(path.read_text(encoding="utf-8").replace("owns 2 cats", "owns 9 cats"), encoding="utf-8")
    with pytest.raises(FrozenFileError, match="not the frozen"):
        run(repo, sets, files)


def test_the_default_files_point_at_the_recorded_answers():
    assert cmp.SETS == {"hard": "data/m2/test-hard.jsonl", "m1": "data/m1/test.jsonl"}
    for folder, name in cmp.V1.values():
        assert all((REPO / folder / f"{name}.{v}.jsonl").is_file() for v in VERSIONS)
    assert cmp.V2 == {"hard": ("results/m2/v2/hard", "qwen3-4b-grpo-v2"),
                      "m1": ("results/m2/v2/m1", "qwen3-4b-grpo-v2")}
