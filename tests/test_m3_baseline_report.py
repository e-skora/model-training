"""Tests for scripts/m3_baseline_report.py. Made-up pairs and answers only; no frozen file is opened."""

import hashlib
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from judge.frozen import FrozenFileError
from judge.score import VERSIONS

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m3_baseline_report", REPO / "scripts" / "m3_baseline_report.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

C = json.dumps({"label": "contradiction", "type": None, "reason": "They clash."})
N = json.dumps({"label": "non_contradiction", "type": "compatible_detail", "reason": "Both fit."})
U = json.dumps({"label": "unclear", "type": None, "reason": "Cannot tell."})
BREAK = json.dumps({"label": "contradiction", "type": "update", "reason": "They clash."})  # type must be null

# 10 contradictions (paraphrase), then 10 non-contradictions (looks_like_clash).
PAIRS = ([{"id": f"p{i:02}", "a": "x", "b": "y", "label": "contradiction", "type": None, "hard": "paraphrase"}
          for i in range(10)] +
         [{"id": f"q{i:02}", "a": "x", "b": "y", "label": "non_contradiction", "type": "compatible_detail",
           "hard": "looks_like_clash"} for i in range(10)])


def answers(contra, non):
    """Answers in file order: `contra` for the 10 contradictions, `non` for the 10 others."""
    return [{"id": p["id"], "raw": raw} for p, raw in zip(PAIRS, contra + non)]


# v1: catches 8 of 10 (one unclear, one missed), one false alarm, so 8 caught of 9 said.
V1 = answers([C] * 8 + [U, N], [C] + [N] * 9)
# v2: catches 9 of 10, no false alarm in q00, but q09 says contradiction with a type (breaks v1's format).
V2 = answers([C] * 9 + [N], [N] * 9 + [BREAK])


def write_setup(tmp_path, v1=V1, v2=V2):
    data = tmp_path / "data"
    data.mkdir()
    sets, lines = {}, []
    for name in ("test", "valid"):
        path = data / f"{name}.jsonl"
        path.write_text("".join(json.dumps(p) + "\n" for p in PAIRS), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"- `data/{name}.jsonl`, 20 pairs, SHA-256 `{digest}`, frozen.")
        sets[name] = str(path)
    state = tmp_path / "STATE.md"
    state.write_text("# STATE\n\n## Frozen files\n" + "\n".join(lines) + "\n", encoding="utf-8")
    root = tmp_path / "results"
    for model, ans in (("v1", v1), ("v2", v2)):
        for name in sets:
            folder = root / model / name
            folder.mkdir(parents=True)
            for version in VERSIONS:
                (folder / f"{rep.NAMES[model]}.{version}.jsonl").write_text(
                    "".join(json.dumps(a) + "\n" for a in ans), encoding="utf-8")
    m2 = tmp_path / "compare.json"
    row = {"pairs": 30, **{f"{m} {v}": 25 for m in ("v1", "v2") for v in VERSIONS}}
    m2.write_text(json.dumps({"counts": {"hard": {}}, "traps": {"paraphrase (contradiction)": row}}),
                  encoding="utf-8")
    return dict(sets=sets, root=str(root), m2_compare=str(m2), state_path=str(state))


def test_counts_are_exact(tmp_path):
    result = rep.build(**write_setup(tmp_path))
    v1, v2 = result["counts"]["v1"]["test"]["as_is"], result["counts"]["v2"]["test"]["as_is"]
    assert (v1["caught"], v1["said"], v1["real"], v1["unclear"], v1["invalid"], v1["format_breaks"]) == (8, 9, 10, 1, 0, 0)
    # The format break is still a "contradiction" answer to the scorer, so it is a false alarm, not invalid.
    assert (v2["caught"], v2["said"], v2["real"], v2["unclear"], v2["invalid"], v2["format_breaks"]) == (9, 10, 10, 0, 0, 1)


def test_report_prints_counts_not_reduced_fractions(tmp_path):
    text = rep.report(rep.build(**write_setup(tmp_path)))
    # 8 of 10 must print as 8/10, never 4/5 (lesson A6).
    assert "8/9 (0.889)" in text and "8/10 (0.800)" in text
    assert "4/5" not in text
    assert "need at least 9/10 (1 more caught)" in text


def test_bar_is_worst_case_plus_three_hundredths_in_pairs():
    c = lambda caught, said: {"caught": caught, "said": said, "real": 300}
    v1 = {"as_is": c(280, 290), "swapped": c(276, 286), "no_dates": c(278, 292)}
    b = rep.bar(v1)
    assert b["recall"] == {"v1_version": "swapped", "v1_caught": 276, "real": 300, "need_caught": 285,
                           "more_than_v1": 9}
    p = b["precision"]
    need = Fraction(278, 292) - Fraction(1, 100)  # no_dates is v1's worst precision
    assert (p["v1_version"], p["v1_caught"], p["v1_said"]) == ("no_dates", 278, 292)
    assert p["need"] == f"{need.numerator}/{need.denominator}"
    assert Fraction(285, p["at_need_caught_max_said"]) >= need > Fraction(285, p["at_need_caught_max_said"] + 1)
    assert p["at_need_caught_max_false_alarms"] == p["at_need_caught_max_said"] - 285


def test_bar_rounds_a_fractional_recall_need_up():
    v1 = {v: {"caught": 8, "said": 9, "real": 10} for v in VERSIONS}
    assert rep.bar(v1)["recall"]["need_caught"] == math.ceil(Fraction(83, 10)) == 9


def test_paired_count(tmp_path):
    paired = rep.build(**write_setup(tmp_path))["paired"]["as_is"]
    # v2 gets p08 (v1 unclear) and q00 (v1 false alarm); v1 gets q09 (v2 said contradiction there).
    assert paired["v1_wrong_v2_right"] == ["p08", "q00"]
    assert paired["v1_right_v2_wrong"] == ["q09"]
    assert paired["contradictions_v1_misses_v2_catches"] == ["p08"]
    assert paired["contradictions_v2_misses_v1_catches"] == []


def test_traps_and_shares(tmp_path):
    result = rep.build(**write_setup(tmp_path))
    row = result["traps"]["paraphrase (contradiction)"]
    assert row["pairs"] == 10 and row["v1 as_is"] == 8 and row["v2 as_is"] == 9 and row["v1 as_is unclear"] == 1
    assert result["shares"]["m3"]["paraphrase (contradiction)"] == {"v1": (24, 30), "v2": (27, 30)}
    assert result["shares"]["m2"]["paraphrase (contradiction)"] == {"v1": (75, 90), "v2": (75, 90)}
    text = rep.report(result)
    assert "75/90 (0.833) -> 24/30 (0.800)" in text
    assert "looks_like_clash (compatible_detail): - -> " in text  # not in the made-up M2 table


def test_stops_when_a_set_is_not_the_frozen_file(tmp_path):
    setup = write_setup(tmp_path)
    Path(setup["sets"]["valid"]).write_text("", encoding="utf-8")
    with pytest.raises(FrozenFileError):
        rep.build(**setup)


def test_save_never_replaces_a_file(tmp_path, monkeypatch):
    setup = write_setup(tmp_path)
    build = rep.build
    monkeypatch.setattr(rep, "build", lambda: build(**setup))
    out = tmp_path / "report"
    rep.main(["m3_baseline_report.py", "--save", str(out)])
    assert out.with_suffix(".txt").read_text(encoding="utf-8").startswith("test set 2:")
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["bar"]["recall"]["need_caught"] == 9
    with pytest.raises(FileExistsError):
        rep.main(["m3_baseline_report.py", "--save", str(out)])


def test_names_and_sets_match_what_eval_trained_writes():
    spec_eval = importlib.util.spec_from_file_location("eval_trained_for_names", REPO / "scripts" / "eval_trained.py")
    ev = importlib.util.module_from_spec(spec_eval)
    spec_eval.loader.exec_module(ev)
    assert rep.NAMES == {"v1": ev.NAME, "v2": ev.V2["name"]}
    assert rep.SETS == {"test": "data/m3/test-hard-2.jsonl", "valid": "data/m3/valid-hard-2.jsonl"}
