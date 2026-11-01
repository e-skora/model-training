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


# The fingerprint reads the whole src/ folder (D-049), so these tests change temporary copies of it.
# Each copy loads in a fresh Python: ambient settings ignored (-I), no .pyc files written (-B), no environment.
SOURCE = Path(reward_module.__file__).parents[1]
CONTRADICTION = (ans("contradiction"), "contradiction", None)
REPEATED = ('{"label": "contradiction", "label": "contradiction", "type": null, "reason": "r"}', "contradiction", None)
PROBE = """
import json, os, sys
sys.path.insert(0, sys.argv[1])
import judge
assert os.path.realpath(judge.__file__).startswith(os.path.realpath(sys.argv[1]) + os.sep)
for name in sys.argv[3:]:  # modules another program would have loaded first
    __import__(name)
from judge.reward import GRPOReward, reward, reward_sha256
raw, gold_label, gold_type = json.loads(sys.argv[2])
print(json.dumps({"sha": reward_sha256(), "points": reward(raw, gold_label, gold_type),
                  "wrapped": GRPOReward()([raw], [gold_label], [gold_type])[0]}))
"""


def src_copy(folder):
    shutil.copytree(SOURCE, folder / "src", ignore=shutil.ignore_patterns("__pycache__"))
    return folder


def edit(folder, file, old, new):
    path = folder / "src" / "judge" / file
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new), encoding="utf-8")


def add_anywhere(folder, file, text):
    """Write a file at a path inside the copied repo, such as `scripts/tool.py` or `src/policy/reader.py`."""
    path = folder / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def add_file(folder, file, text):
    add_anywhere(folder, f"src/judge/{file}", text)


def run_probe(folder, answer=CONTRADICTION, cwd=None, first=(), source="src"):
    command = [sys.executable, "-I", "-B", "-c", PROBE, str(folder / source), json.dumps(answer), *first]
    return subprocess.run(command, cwd=cwd or folder, env={}, capture_output=True, text=True)


def probe(folder, answer=CONTRADICTION, cwd=None, first=()):
    done = run_probe(folder, answer, cwd, first)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def add_later(folder, file, line):
    """Add code to a copied file, inside a function so it never runs. Returns where a stop must point."""
    path = folder / "src" / "judge" / file
    text = path.read_text(encoding="utf-8")
    path.write_text(f"{text}\n\ndef later():\n    {line}\n", encoding="utf-8")
    return f"judge/{file}, line {text.count(chr(10)) + 4}"


def assert_stops(folder, where, **run):
    done = run_probe(folder, **run)
    assert done.returncode != 0
    assert "reward fingerprint" in done.stderr and where in done.stderr, done.stderr


def use_reader(folder, helper, hook="read_fields"):
    """Make the copy's answer reader call a new helper, written just above `parse_answer`."""
    edit(folder, "score.py", "def parse_answer(raw):", f"{helper}\n\n\ndef parse_answer(raw):")
    edit(folder, "score.py", "object_pairs_hook=no_repeated_fields", f"object_pairs_hook={hook}")


def assert_points_and_fingerprint_move(strict, loose, first=()):
    """Repeated fields are refused by one copy and allowed by the other: points and fingerprint both differ."""
    before, after = probe(strict, REPEATED, first=first), probe(loose, REPEATED, first=first)
    assert (before["points"], after["points"]) == (-2.0, 1.0)
    assert before["sha"] != after["sha"]


@pytest.mark.parametrize("key", REWARD)
def test_fingerprint_changes_with_a_value(tmp_path, key):
    before = probe(src_copy(tmp_path / "before"))["sha"]
    after = src_copy(tmp_path / "after")
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
    before = probe(src_copy(tmp_path / "before"), answer)
    changed = src_copy(tmp_path / "after")
    edit(changed, file, old, new)
    after = probe(changed, answer)
    assert after["points"] != before["points"]  # the change matters to the reward
    assert after["sha"] != before["sha"]


def test_fingerprint_changes_with_how_a_completion_is_read(tmp_path):
    before = probe(src_copy(tmp_path / "before"))["sha"]
    after = src_copy(tmp_path / "after")
    edit(after, "reward.py", 'return completion[0]["content"]', 'return completion[-1]["content"]')
    assert probe(after)["sha"] != before


# Codex review M2-04b, finding 1: three ordinary ways of writing a helper the name walk could not see.
STRICT_HELPER = '''STRICT = {strict}


def read_fields(items, strict=STRICT):
    if strict and len(items) != len(dict(items)):
        raise RepeatedField("repeated field")
    return dict(items)'''


def test_fingerprint_changes_with_a_default_value(tmp_path):
    strict, loose = src_copy(tmp_path / "strict"), src_copy(tmp_path / "loose")
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
    strict, loose = src_copy(tmp_path / "strict"), src_copy(tmp_path / "loose")
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
    strict, loose = src_copy(tmp_path / "strict"), src_copy(tmp_path / "loose")
    for folder, source in ((strict, STRICT_MODULE), (loose, LOOSE_MODULE)):
        add_file(folder, "sub/__init__.py", "")
        add_file(folder, file, source)
        use_reader(folder, line, f"lambda items: {call}(items)")
    assert_points_and_fingerprint_move(strict, loose)


def test_fingerprint_follows_a_module_reached_without_importing_it(tmp_path):
    # `judge.policy` works here once any other file imported it; the fingerprint must not need that file.
    strict, loose = src_copy(tmp_path / "strict"), src_copy(tmp_path / "loose")
    for folder, source in ((strict, STRICT_MODULE), (loose, LOOSE_MODULE)):
        add_file(folder, "policy.py", source)
        edit(folder, "score.py", "def parse_answer(raw):",
             "import judge.prompt\n\n\ndef read_fields(items):\n    return judge.policy.read_fields(items)\n\n\ndef parse_answer(raw):")
    assert probe(strict)["sha"] != probe(loose)["sha"]


def test_fingerprint_follows_a_relative_import_two_levels_up(tmp_path):
    strict, loose = src_copy(tmp_path / "strict"), src_copy(tmp_path / "loose")
    reader = "from .. import policy\n\n\ndef read_fields(items):\n    return policy.read_fields(items)\n"
    for folder, source in ((strict, STRICT_MODULE), (loose, LOOSE_MODULE)):
        add_file(folder, "policy.py", source)
        add_file(folder, "sub/__init__.py", "")
        add_file(folder, "sub/reader.py", reader)
        use_reader(folder, "from judge.sub.reader import read_fields")
    assert_points_and_fingerprint_move(strict, loose)


# Codex review M2-04b, finding 2.
def test_fingerprint_changes_with_the_training_wrapper(tmp_path):
    before = probe(src_copy(tmp_path / "before"))
    changed = src_copy(tmp_path / "after")
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
    strict, loose = src_copy(tmp_path / "strict"), src_copy(tmp_path / "loose")
    add_later_modules(strict, False)
    add_later_modules(loose, False, LATER_POLICY.replace("not ALLOW_REPEATS and len(items) != len(dict(items))", "False"))
    assert_points_and_fingerprint_move(strict, loose)


def test_a_module_imported_by_a_later_module_is_covered(tmp_path):
    strict, loose = src_copy(tmp_path / "strict"), src_copy(tmp_path / "loose")
    add_later_modules(strict, False)
    add_later_modules(loose, True)
    assert_points_and_fingerprint_move(strict, loose)


def test_any_edit_to_a_covered_file_moves_the_fingerprint(tmp_path):
    before = probe(src_copy(tmp_path / "before"))["sha"]
    after = src_copy(tmp_path / "after")
    edit(after, "frozen.py", "REPO = Path(__file__)", "# A comment, far from the reward code.\nREPO = Path(__file__)")
    assert probe(after)["sha"] != before


@pytest.mark.parametrize("file", ["load.py", "modal_image.py", "train_data.py"])
def test_an_edit_to_a_file_the_reward_never_imports_moves_the_fingerprint(tmp_path, file):
    before = probe(src_copy(tmp_path / "before"))["sha"]
    after = src_copy(tmp_path / "after")
    path = after / "src" / "judge" / file
    path.write_text(path.read_text(encoding="utf-8") + "# A comment.\n", encoding="utf-8")
    assert probe(after)["sha"] != before


# Code that loads other code by name (D-049, stop 1), inside a function so it never runs.
LOADS_BY_NAME = [
    "import importlib",
    "import importlib.util",
    "from importlib import import_module",
    "__import__('json')",
    "exec('x = 1')",
    "eval('1')",
    "loader = eval",
    "import builtins",
    "from builtins import eval",
    "loader = __builtins__",
    "import runpy",
    "from runpy import run_path",
    "compile('1', 'x', 'eval')",
    "loader = compile",
    "from re import compile",
    "from other import exec",
    "loader = other.exec",
    "loader = other.compile",
    # Only `re.compile`, on Python's own `re`, is allowed (D-050).
    "import re as r; r.compile('x')",
]


@pytest.mark.parametrize("line", LOADS_BY_NAME)
@pytest.mark.parametrize("file", ["reward.py", "frozen.py"])
def test_code_that_loads_code_by_name_stops_with_an_error(tmp_path, file, line):
    folder = src_copy(tmp_path / "copy")
    assert_stops(folder, add_later(folder, file, line))


@pytest.mark.parametrize("text, line", [
    ("import re\n\n\ndef later(re):\n    return re.compile('x')\n", 5),
    ("import json as re\n\nre.compile('x')\n", 3),
    ("import re\n\nre = None\nre.compile('x')\n", 4),
    ("from other import re\n\nre.compile('x')\n", 3),
    ("re.compile('x')\n", 1),  # no `import re` at all
])
def test_compile_on_anything_but_python_s_re_stops_with_an_error(tmp_path, text, line):
    folder = src_copy(tmp_path / "copy")
    add_file(folder, "notes.py", text)
    assert_stops(folder, f"judge/notes.py, line {line}")


def test_a_loader_in_a_file_the_reward_never_imports_stops_with_an_error(tmp_path):
    folder = src_copy(tmp_path / "copy")
    add_anywhere(folder, "src/other/thing.py", "import runpy\n")
    assert_stops(folder, "other/thing.py, line 1")


# Codex review M2-04d, finding 1: runpy reaches a helper that no import names.
RUNPY_FORMS = [
    "import runpy\n\n\nread_fields = runpy.run_module('judge.policy')['read_fields']",
    "import runpy\n\n\nread_fields = runpy.run_path(__file__.replace('score.py', 'policy.py'))['read_fields']",
    "from runpy import run_module\n\n\nread_fields = run_module('judge.policy')['read_fields']",
]


@pytest.mark.parametrize("helper", RUNPY_FORMS)
def test_runpy_stops_with_an_error(tmp_path, helper):
    folder = src_copy(tmp_path / "copy")
    add_file(folder, "policy.py", LOOSE_MODULE)
    use_reader(folder, helper)
    assert_stops(folder, "judge/score.py, line")


# Codex review M2-04d, finding 2: a folder beside `judge` with no __init__.py.
@pytest.mark.parametrize("line, hook", [
    ("from policy.reader import read_fields", "read_fields"),
    ("import policy.reader", "lambda items: policy.reader.read_fields(items)"),
])
def test_fingerprint_covers_a_folder_with_no_init_file(tmp_path, line, hook):
    strict, loose = src_copy(tmp_path / "strict"), src_copy(tmp_path / "loose")
    for folder, source in ((strict, STRICT_MODULE), (loose, LOOSE_MODULE)):
        add_anywhere(folder, "src/policy/reader.py", source)
        use_reader(folder, line, hook)
    assert_points_and_fingerprint_move(strict, loose)


# Codex review M2-04d, finding 3: a nested module loaded first by another import, then called by its full name.
def test_fingerprint_covers_a_nested_module_loaded_by_another_import(tmp_path):
    strict, loose = src_copy(tmp_path / "strict"), src_copy(tmp_path / "loose")
    for folder, source in ((strict, STRICT_MODULE), (loose, LOOSE_MODULE)):
        add_file(folder, "sub/__init__.py", "")
        add_file(folder, "sub/policy.py", source)
        use_reader(folder, "import judge.prompt", "lambda items: judge.sub.policy.read_fields(items)")
    assert_points_and_fingerprint_move(strict, loose, first=["judge.sub.policy"])


def test_code_beside_the_package_is_covered(tmp_path):
    strict, loose = src_copy(tmp_path / "strict"), src_copy(tmp_path / "loose")
    for folder, source in ((strict, STRICT_MODULE), (loose, LOOSE_MODULE)):
        add_anywhere(folder, "src/policy.py", source)
        use_reader(folder, "from policy import read_fields")
    assert_points_and_fingerprint_move(strict, loose)


@pytest.mark.parametrize("file", ["judge/notes.txt", "judge/sub/settings.json", "other/reader.py", "loose.txt", "judge/.hidden"])
def test_any_new_file_under_src_moves_the_fingerprint(tmp_path, file):
    before = probe(src_copy(tmp_path / "before"))["sha"]
    after = src_copy(tmp_path / "after")
    add_anywhere(after, f"src/{file}", "x\n")
    assert probe(after)["sha"] != before


def test_a_file_moved_to_another_path_moves_the_fingerprint(tmp_path):
    here, there = src_copy(tmp_path / "here"), src_copy(tmp_path / "there")
    add_anywhere(here, "src/one/notes.txt", "x\n")
    add_anywhere(there, "src/two/notes.txt", "x\n")
    assert probe(here)["sha"] != probe(there)["sha"]


def test_where_one_file_ends_and_the_next_starts_counts(tmp_path):
    one, two = src_copy(tmp_path / "one"), src_copy(tmp_path / "two")
    add_anywhere(one, "src/a", "bc")
    add_anywhere(one, "src/b", "d")
    add_anywhere(two, "src/a", "b")
    add_anywhere(two, "src/b", "cd")
    assert probe(one)["sha"] != probe(two)["sha"]


@pytest.mark.parametrize("file", [
    "judge/__pycache__/score.cpython-312.pyc", "judge/__pycache__/notes.txt", "other/__pycache__/x.pyc", "judge/loose.pyc",
])
def test_cache_files_do_not_move_the_fingerprint(tmp_path, file):
    # Passes on the old code too; `reviews/M2-04-fixes.md` says how it was shown to catch a problem.
    before = probe(src_copy(tmp_path / "before"))["sha"]
    after = src_copy(tmp_path / "after")
    add_anywhere(after, f"src/{file}", "x\n")
    assert probe(after)["sha"] == before


@pytest.mark.parametrize("line", [
    "import sys; sys.path.insert(0, 'x')",
    "import sys; sys.path.append('x')",
    "import sys; sys.path = []",
    "import sys as system; system.path.insert(0, 'x')",
    "from sys import path",
    "from sys import path as places; places.insert(0, 'x')",
])
@pytest.mark.parametrize("file", ["score.py", "load.py"])
def test_changing_sys_path_stops_with_an_error(tmp_path, file, line):
    folder = src_copy(tmp_path / "copy")
    assert_stops(folder, add_later(folder, file, line))


def test_a_star_import_from_sys_stops_with_an_error(tmp_path):
    folder = src_copy(tmp_path / "copy")
    add_file(folder, "notes.py", "from sys import *\n")
    assert_stops(folder, "judge/notes.py, line 1")


# Project code outside src/ (D-049, stop 2).
@pytest.mark.parametrize("place", [
    "tool.py", "tool/x.py", "scripts/tool.py", "scripts/tool/x.py", "tests/tool.py", "tests/tool/x.py",
])
@pytest.mark.parametrize("line", ["import tool", "from tool.x import y", "import tool.x as renamed"])
def test_importing_project_code_outside_src_stops_with_an_error(tmp_path, place, line):
    folder = src_copy(tmp_path / "copy")
    add_anywhere(folder, place, "")
    assert_stops(folder, add_later(folder, "frozen.py", line))


@pytest.mark.parametrize("entry", ["policy", "policy.py"])
def test_a_link_under_src_stops_with_an_error(tmp_path, entry):
    # A link could point anywhere, so the fingerprint cannot say it read the file behind it.
    folder = src_copy(tmp_path / "copy")
    add_anywhere(tmp_path, "outside/policy.py", LOOSE_MODULE)
    target = tmp_path / "outside" / ("policy.py" if entry.endswith(".py") else "")
    (folder / "src" / entry).symlink_to(target)
    assert_stops(folder, entry)


def test_the_fingerprint_stops_when_the_package_is_not_in_a_src_folder(tmp_path):
    # Modal copies `judge` into /root, and an install into site-packages: "the folder around judge" would be the wrong one.
    folder = src_copy(tmp_path / "copy")
    (folder / "src").rename(folder / "elsewhere")
    assert_stops(folder, "src folder", source="elsewhere")


def test_words_in_strings_and_comments_do_not_stop(tmp_path):
    # Passes on the old code too; `reviews/M2-04-fixes.md` says how it was shown to catch a problem.
    folder = src_copy(tmp_path / "copy")
    add_file(folder, "notes.py", '"""importlib, runpy, compile."""\n# exec, eval, sys.path.insert\nWORDS = ["__import__", "builtins"]\n')
    assert len(probe(folder)["sha"]) == 64


def test_fingerprint_is_stable_and_does_not_depend_on_the_folder(tmp_path):
    here, there = src_copy(tmp_path / "here"), src_copy(tmp_path / "some" / "other" / "place")
    first, second, elsewhere = probe(here)["sha"], probe(here)["sha"], probe(there, cwd=tmp_path)["sha"]
    assert len(first) == 64
    assert first == second == elsewhere == reward_sha256()


def test_fingerprint_reads_files_not_memory(monkeypatch):
    # The limit of D-048, kept by D-049: a change made in memory while a program runs does not show.
    before = reward_sha256()
    monkeypatch.setitem(reward_module.REWARD, "invalid", 0.0)
    assert reward_sha256() == before


def test_fingerprint_covers_these_files_today():
    assert sorted(reward_module.fingerprinted_files()) == [
        "judge/__init__.py", "judge/format.py", "judge/frozen.py", "judge/generate.py", "judge/load.py",
        "judge/modal_image.py", "judge/pool.py", "judge/prompt.py", "judge/reward.py", "judge/score.py",
        "judge/train_data.py",
    ]
