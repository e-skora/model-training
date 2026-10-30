"""Tests for judge.reward with made-up answers. No GPU, no network. Fingerprint tests use temporary copies of the code."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import judge.reward as reward_module
from judge.reward import REWARD, GRPOReward, reward, reward_sha256


def ans(label, type_=None, reason="A short reason."):
    return json.dumps({"label": label, "type": type_, "reason": reason})


# Every row of the decided-answer table (D-040): right answer, model answer, points.
DECIDED = [
    ("contradiction", None, ans("contradiction"), 1.0),
    ("non_contradiction", "update", ans("non_contradiction", "update"), 1.25),
    ("non_contradiction", "compatible_detail", ans("non_contradiction", "compatible_detail"), 1.25),
    ("non_contradiction", "merely_related", ans("non_contradiction", "merely_related"), 1.25),
    ("non_contradiction", "update", ans("non_contradiction", "merely_related"), 1.0),  # right label, wrong type
    ("non_contradiction", "update", ans("contradiction"), -1.5),  # false alarm
    ("contradiction", None, ans("non_contradiction", "update"), -1.0),  # miss
    ("contradiction", None, ans("unclear"), -0.5),
    ("non_contradiction", "merely_related", ans("unclear"), -0.5),
]


@pytest.mark.parametrize("gold_label,gold_type,raw,points", DECIDED)
def test_decided_table(gold_label, gold_type, raw, points):
    assert reward(raw, gold_label, gold_type) == points


# Every row of the unclear-answer table (D-046).
@pytest.mark.parametrize("raw,points", [
    (ans("unclear"), 1.0),
    (ans("non_contradiction", "merely_related"), 0.0),
    (ans("non_contradiction", "update"), 0.0),  # no type bonus on an unclear pair
    (ans("contradiction"), -1.5),
])
def test_unclear_table(raw, points):
    assert reward(raw, "unclear", None) == points


INVALID = [
    "I think they contradict.",  # not JSON
    '{"label": "contradiction", "label": "unclear", "type": null, "reason": "x"}',  # repeated field
    ans("maybe"),  # unknown label
    '{"label": "contradiction", "type": NaN, "reason": "x"}',  # NaN is not JSON
    "Answer: " + ans("contradiction"),  # extra text before
    ans("contradiction") + " Hope that helps.",  # extra text after
    ans("contradiction", "update"),  # a type on a contradiction
    ans("unclear", "merely_related"),  # a type on unclear
    ans("non_contradiction", None),  # no type on a non-contradiction
    ans("non_contradiction", "similar"),  # unknown type
    '{"label": "contradiction", "type": null}',  # no reason
    ans("contradiction", reason="   "),  # empty reason
    '["contradiction"]',  # JSON, but not an object
    "",
    None,
]


@pytest.mark.parametrize("raw", INVALID)
@pytest.mark.parametrize("gold_label,gold_type", [("contradiction", None), ("non_contradiction", "update"), ("unclear", None)])
def test_invalid_answers(raw, gold_label, gold_type):
    assert reward(raw, gold_label, gold_type) == -2.0


def test_fenced_answer_counts_as_the_scorer_counts_it():
    # The scorer accepts one code fence around the whole answer; so does the reward.
    assert reward("```json\n" + ans("contradiction") + "\n```", "contradiction", None) == 1.0


def test_reason_earns_nothing():
    short, long = ans("contradiction", reason="x"), ans("contradiction", reason="A careful, detailed reason. " * 10)
    assert reward(short, "contradiction", None) == reward(long, "contradiction", None) == 1.0


def test_type_bonus_only_with_the_right_label():
    # The right type on a wrong label earns no bonus: unclear and contradiction carry type null anyway,
    # and on an unclear pair a non_contradiction with any type scores 0.
    assert reward(ans("non_contradiction", "update"), "non_contradiction", "update") == 1.25
    assert reward(ans("non_contradiction", "update"), "contradiction", None) == -1.0
    assert reward(ans("non_contradiction", "update"), "unclear", None) == 0.0


def test_unknown_gold_label_stops():
    with pytest.raises(ValueError):
        reward(ans("contradiction"), "contradicts", None)


def test_wrapper_on_a_small_batch():
    fn = GRPOReward()
    # Conversational completions, as TRL passes them: a list of messages per completion.
    completions = [[{"role": "assistant", "content": text}] for text in (
        ans("contradiction"), ans("unclear"), "not json", ans("non_contradiction", "update"), ans("contradiction"),
    )]
    gold_label = ["contradiction", "contradiction", "contradiction", "non_contradiction", "unclear"]
    gold_type = [None, None, None, "update", None]
    logged = {}
    rewards = fn(completions=completions, gold_label=gold_label, gold_type=gold_type,
                 prompts=["p"] * 5, completion_ids=[[1]] * 5, trainer_state=None,
                 log_metric=lambda name, value: logged.__setitem__(name, value), hard=["x"] * 5)
    assert rewards == [1.0, -0.5, -2.0, 1.25, -1.5]
    assert all(isinstance(r, float) for r in rewards)
    assert fn.last_counts == {"contradiction": 2, "non_contradiction": 1, "unclear": 1, "invalid": 1}
    assert logged == {"answers/contradiction": 0.4, "answers/non_contradiction": 0.2,
                      "answers/unclear": 0.2, "answers/invalid": 0.2}
    assert fn.__name__ == "judge_reward"

    # Plain-text completions work too, and totals add up across calls.
    assert fn([ans("unclear")], gold_label=["unclear"], gold_type=[None]) == [1.0]
    assert fn.last_counts == {"contradiction": 0, "non_contradiction": 0, "unclear": 1, "invalid": 0}
    assert fn.total_counts == {"contradiction": 2, "non_contradiction": 1, "unclear": 2, "invalid": 1}


def test_wrapper_stops_on_mismatched_lengths():
    with pytest.raises(ValueError):
        GRPOReward()([ans("unclear")], gold_label=["unclear", "unclear"], gold_type=[None])


# The fingerprint reads source files (D-048), so these tests change temporary copies of the judge code.
# Each copy loads in a fresh Python: ambient settings ignored (-I), no .pyc files written (-B), no environment.
JUDGE_SOURCE = Path(reward_module.__file__).parent
CONTRADICTION = (ans("contradiction"), "contradiction", None)
REPEATED = ('{"label": "contradiction", "label": "contradiction", "type": null, "reason": "r"}', "contradiction", None)
PROBE = """
import json, os, sys
sys.path.insert(0, sys.argv[1])
import judge
assert os.path.realpath(judge.__file__).startswith(os.path.realpath(sys.argv[1]) + os.sep)
from judge.reward import GRPOReward, reward, reward_sha256
raw, gold_label, gold_type = json.loads(sys.argv[2])
print(json.dumps({"sha": reward_sha256(), "points": reward(raw, gold_label, gold_type),
                  "wrapped": GRPOReward()([raw], [gold_label], [gold_type])[0]}))
"""


def judge_copy(folder):
    shutil.copytree(JUDGE_SOURCE, folder / "judge", ignore=shutil.ignore_patterns("__pycache__"))
    return folder


def edit(folder, file, old, new):
    path = folder / "judge" / file
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new), encoding="utf-8")


def add_file(folder, file, text):
    path = folder / "judge" / file
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_probe(folder, answer=CONTRADICTION, cwd=None):
    command = [sys.executable, "-I", "-B", "-c", PROBE, str(folder), json.dumps(answer)]
    return subprocess.run(command, cwd=cwd or folder, env={}, capture_output=True, text=True)


def probe(folder, answer=CONTRADICTION, cwd=None):
    done = run_probe(folder, answer, cwd)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def use_reader(folder, helper, hook="read_fields"):
    """Make the copy's answer reader call a new helper, written just above `parse_answer`."""
    edit(folder, "score.py", "def parse_answer(raw):", f"{helper}\n\n\ndef parse_answer(raw):")
    edit(folder, "score.py", "object_pairs_hook=no_repeated_fields", f"object_pairs_hook={hook}")


def assert_points_and_fingerprint_move(strict, loose):
    """Repeated fields are refused by one copy and allowed by the other: points and fingerprint both differ."""
    before, after = probe(strict, REPEATED), probe(loose, REPEATED)
    assert (before["points"], after["points"]) == (-2.0, 1.0)
    assert before["sha"] != after["sha"]


@pytest.mark.parametrize("key", REWARD)
def test_fingerprint_changes_with_a_value(tmp_path, key):
    before = probe(judge_copy(tmp_path / "before"))["sha"]
    after = judge_copy(tmp_path / "after")
    edit(after, "reward.py", f'"{key}": {REWARD[key]}', f'"{key}": {REWARD[key] + 0.1}')
    assert probe(after)["sha"] != before


# Codex review M2-04, finding 1, as edits to the source. (file, text, new text, answer)
READER_CHANGES = [
    ("prompt.py", 'TYPES = ("update", ', "TYPES = (", (ans("non_contradiction", "update"), "non_contradiction", "update")),
    ("score.py", "object_pairs_hook=no_repeated_fields", "object_pairs_hook=dict", REPEATED),
    ("score.py", "parse_constant=reject_constant", "parse_constant=lambda name: None",
     ('{"label": "contradiction", "type": NaN, "reason": "r"}', "contradiction", None)),
    ("score.py", r'r"\A```[a-zA-Z]*\s*\n(.*?)\n?```\Z"', r'r"\A~~~[a-zA-Z]*\s*\n(.*?)\n?~~~\Z"',
     ("~~~json\n" + ans("contradiction") + "\n~~~", "contradiction", None)),
]


@pytest.mark.parametrize("file, old, new, answer", READER_CHANGES)
def test_fingerprint_changes_with_the_answer_reader(tmp_path, file, old, new, answer):
    before = probe(judge_copy(tmp_path / "before"), answer)
    changed = judge_copy(tmp_path / "after")
    edit(changed, file, old, new)
    after = probe(changed, answer)
    assert after["points"] != before["points"]  # the change matters to the reward
    assert after["sha"] != before["sha"]


def test_fingerprint_changes_with_how_a_completion_is_read(tmp_path):
    before = probe(judge_copy(tmp_path / "before"))["sha"]
    after = judge_copy(tmp_path / "after")
    edit(after, "reward.py", 'return completion[0]["content"]', 'return completion[-1]["content"]')
    assert probe(after)["sha"] != before


# Codex review M2-04b, finding 1: three ordinary ways of writing a helper the name walk could not see.
STRICT_HELPER = '''STRICT = {strict}


def read_fields(items, strict=STRICT):
    if strict and len(items) != len(dict(items)):
        raise RepeatedField("repeated field")
    return dict(items)'''


def test_fingerprint_changes_with_a_default_value(tmp_path):
    strict, loose = judge_copy(tmp_path / "strict"), judge_copy(tmp_path / "loose")
    use_reader(strict, STRICT_HELPER.format(strict=True))
    use_reader(loose, STRICT_HELPER.format(strict=False))
    assert_points_and_fingerprint_move(strict, loose)


CLOSURE_HELPER = '''def make_reader(allow):
    def read_fields(items):
        if not allow and len(items) != len(dict(items)):
            raise RepeatedField("repeated field")
        return dict(items)
    return read_fields


read_fields = make_reader({allow})'''


def test_fingerprint_changes_with_a_captured_value(tmp_path):
    strict, loose = judge_copy(tmp_path / "strict"), judge_copy(tmp_path / "loose")
    use_reader(strict, CLOSURE_HELPER.format(allow=False))
    use_reader(loose, CLOSURE_HELPER.format(allow=True))
    assert_points_and_fingerprint_move(strict, loose)


STRICT_MODULE = '''def read_fields(items):
    if len(items) != len(dict(items)):
        raise ValueError("repeated field")
    return dict(items)
'''
LOOSE_MODULE = "def read_fields(items):\n    return dict(items)\n"


# (the import written in score.py, the call, the file the helper lives in)
MODULE_FORMS = [
    ("import judge.policy", "judge.policy.read_fields", "policy.py"),
    ("import judge.policy as policy", "policy.read_fields", "policy.py"),
    ("from judge import policy", "policy.read_fields", "policy.py"),
    ("from . import policy", "policy.read_fields", "policy.py"),
    ("from .sub import policy", "policy.read_fields", "sub/policy.py"),
    ("from judge.sub import policy", "policy.read_fields", "sub/policy.py"),
]


@pytest.mark.parametrize("line, call, file", MODULE_FORMS)
def test_fingerprint_changes_with_a_helper_reached_through_a_module(tmp_path, line, call, file):
    strict, loose = judge_copy(tmp_path / "strict"), judge_copy(tmp_path / "loose")
    for folder, source in ((strict, STRICT_MODULE), (loose, LOOSE_MODULE)):
        add_file(folder, "sub/__init__.py", "")
        add_file(folder, file, source)
        use_reader(folder, line, f"lambda items: {call}(items)")
    assert_points_and_fingerprint_move(strict, loose)


def test_fingerprint_follows_a_module_reached_without_importing_it(tmp_path):
    # `judge.policy` works here once any other file imported it; the fingerprint must not need that file.
    strict, loose = judge_copy(tmp_path / "strict"), judge_copy(tmp_path / "loose")
    for folder, source in ((strict, STRICT_MODULE), (loose, LOOSE_MODULE)):
        add_file(folder, "policy.py", source)
        edit(folder, "score.py", "def parse_answer(raw):",
             "import judge.prompt\n\n\ndef read_fields(items):\n    return judge.policy.read_fields(items)\n\n\ndef parse_answer(raw):")
    assert probe(strict)["sha"] != probe(loose)["sha"]


def test_fingerprint_follows_a_relative_import_two_levels_up(tmp_path):
    strict, loose = judge_copy(tmp_path / "strict"), judge_copy(tmp_path / "loose")
    reader = "from .. import policy\n\n\ndef read_fields(items):\n    return policy.read_fields(items)\n"
    for folder, source in ((strict, STRICT_MODULE), (loose, LOOSE_MODULE)):
        add_file(folder, "policy.py", source)
        add_file(folder, "sub/__init__.py", "")
        add_file(folder, "sub/reader.py", reader)
        use_reader(folder, "from judge.sub.reader import read_fields")
    assert_points_and_fingerprint_move(strict, loose)


# Codex review M2-04b, finding 2.
def test_fingerprint_changes_with_the_training_wrapper(tmp_path):
    before = probe(judge_copy(tmp_path / "before"))
    changed = judge_copy(tmp_path / "after")
    edit(changed, "reward.py", "rewards.append(reward(text, label, type_))", "rewards.append(2 * reward(text, label, type_))")
    after = probe(changed)
    assert (before["wrapped"], after["wrapped"]) == (1.0, 2.0)
    assert before["points"] == after["points"] == 1.0  # `reward` itself is the same
    assert after["sha"] != before["sha"]


# A module added later, imported only inside a function, and a second one it imports in turn.
LATER_POLICY = """from judge.rules import ALLOW_REPEATS


def read_fields(items):
    if not ALLOW_REPEATS and len(items) != len(dict(items)):
        raise ValueError("repeated field")
    return dict(items)
"""


def add_later_modules(folder, allow, policy=LATER_POLICY):
    add_file(folder, "rules.py", f"ALLOW_REPEATS = {allow}\n")
    add_file(folder, "policy.py", policy)
    edit(folder, "score.py", "    text = raw.strip()\n", "    from judge.policy import read_fields\n    text = raw.strip()\n")
    edit(folder, "score.py", "object_pairs_hook=no_repeated_fields", "object_pairs_hook=read_fields")


def test_a_module_added_later_is_covered_without_a_list(tmp_path):
    strict, loose = judge_copy(tmp_path / "strict"), judge_copy(tmp_path / "loose")
    add_later_modules(strict, False)
    add_later_modules(loose, False, LATER_POLICY.replace("not ALLOW_REPEATS and len(items) != len(dict(items))", "False"))
    assert_points_and_fingerprint_move(strict, loose)


def test_a_module_imported_by_a_later_module_is_covered(tmp_path):
    strict, loose = judge_copy(tmp_path / "strict"), judge_copy(tmp_path / "loose")
    add_later_modules(strict, False)
    add_later_modules(loose, True)
    assert_points_and_fingerprint_move(strict, loose)


def test_any_edit_to_a_covered_file_moves_the_fingerprint(tmp_path):
    before = probe(judge_copy(tmp_path / "before"))["sha"]
    after = judge_copy(tmp_path / "after")
    edit(after, "frozen.py", "REPO = Path(__file__)", "# A comment, far from the reward code.\nREPO = Path(__file__)")
    assert probe(after)["sha"] != before


# Code that loads other code in a way the file reading cannot follow, inside a function so it never runs.
UNFOLLOWABLE = [
    "import importlib",
    "import importlib.util",
    "from importlib import import_module",
    "__import__('json')",
    "exec('x = 1')",
    "eval('1')",
    "loader = eval",
    "import builtins",
    "from builtins import eval",
    "import judge",
    "import judge as whole_package",
    "import judge.ghost",
    "from judge.ghost import helper",
    "from .. import outside",
]


@pytest.mark.parametrize("line", UNFOLLOWABLE)
@pytest.mark.parametrize("file", ["reward.py", "frozen.py"])
def test_code_that_cannot_be_followed_stops_with_an_error(tmp_path, file, line):
    folder = judge_copy(tmp_path / "copy")
    path = folder / "judge" / file
    path.write_text(path.read_text(encoding="utf-8") + f"\n\ndef later():\n    {line}\n", encoding="utf-8")
    done = run_probe(folder)
    assert done.returncode != 0
    assert "reward fingerprint" in done.stderr and f"judge/{file}" in done.stderr


def test_a_star_import_from_the_package_stops_with_an_error(tmp_path):
    folder = judge_copy(tmp_path / "copy")
    path = folder / "judge" / "frozen.py"
    path.write_text(path.read_text(encoding="utf-8") + "\nfrom judge import *\n", encoding="utf-8")
    done = run_probe(folder)
    assert done.returncode != 0
    assert "reward fingerprint" in done.stderr and "judge/frozen.py" in done.stderr


def test_project_code_beside_the_package_stops_with_an_error(tmp_path):
    folder = judge_copy(tmp_path / "copy")
    (folder / "policy.py").write_text(LOOSE_MODULE, encoding="utf-8")
    path = folder / "judge" / "frozen.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n\ndef later():\n    import policy\n", encoding="utf-8")
    done = run_probe(folder)
    assert done.returncode != 0
    assert "reward fingerprint" in done.stderr and "judge/frozen.py" in done.stderr


def test_fingerprint_is_stable_and_does_not_depend_on_the_folder(tmp_path):
    here, there = judge_copy(tmp_path / "here"), judge_copy(tmp_path / "some" / "other" / "place")
    first, second, elsewhere = probe(here)["sha"], probe(here)["sha"], probe(there, cwd=tmp_path)["sha"]
    assert len(first) == 64
    assert first == second == elsewhere == reward_sha256()


def test_fingerprint_reads_files_not_memory(monkeypatch):
    # The limit of D-048: a change made in memory while a program runs does not show.
    before = reward_sha256()
    monkeypatch.setitem(reward_module.REWARD, "invalid", 0.0)
    assert reward_sha256() == before


def test_fingerprint_covers_these_files_today():
    assert sorted(reward_module.fingerprinted_files()) == [
        "judge/__init__.py", "judge/frozen.py", "judge/prompt.py", "judge/reward.py", "judge/score.py",
    ]
