"""The reward function for GRPO: points for one model answer, given the right label and type.

Every number lives in REWARD below, so a change touches one place and shows
in `reward_sha256()` (D-040, D-046). Answers are read with the scorer's own
strict reader (`judge.score.parse_answer`) and its field check
(`judge.score.format_problem`), so "valid JSON in the exact format" means
exactly what scoring means. The `reason` text earns nothing.

This file imports no GPU libraries, so tests run it on the Mac.
"""

import ast
import hashlib
import json
from pathlib import Path

from judge.prompt import LABELS
from judge.score import format_problem, parse_answer

REWARD = {
    # Any answer the scorer counts as invalid, or whose type or reason break the format.
    "invalid": -2.0,
    # When the right answer is decided (contradiction or non_contradiction), D-040:
    "right_label": 1.0,
    "right_type_bonus": 0.25,  # added to right_label, non-contradictions only
    "false_alarm": -1.5,  # said contradiction when it is not
    "miss": -1.0,  # said non_contradiction on a real contradiction
    "unclear_on_decided": -0.5,
    # When the right answer is unclear, D-046:
    "unclear_on_unclear": 1.0,
    "non_contradiction_on_unclear": 0.0,
    "contradiction_on_unclear": -1.5,
}

ANSWER_KINDS = ("contradiction", "non_contradiction", "unclear", "invalid")

# The folder that holds `judge`, and names that load code in a way reading imports cannot follow.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
UNFOLLOWABLE = {"importlib", "builtins", "__builtins__", "__import__", "exec", "eval"}


def read_answer(raw):
    """The answer's label, or None when it is invalid by the scorer's rules. Returns (label, answer)."""
    answer, problem, _ = parse_answer(raw)
    if problem or format_problem(answer):
        return None, None
    return answer["label"], answer


def reward(raw_answer, gold_label, gold_type):
    """Points for one raw model answer. `gold_type` is the right type, or None for contradiction and unclear."""
    if gold_label not in LABELS:
        raise ValueError(f"unknown right label {gold_label!r}")
    label, answer = read_answer(raw_answer)
    if label is None:
        return REWARD["invalid"]
    if gold_label == "unclear":
        return {
            "unclear": REWARD["unclear_on_unclear"],
            "non_contradiction": REWARD["non_contradiction_on_unclear"],
            "contradiction": REWARD["contradiction_on_unclear"],
        }[label]
    if label == "unclear":
        return REWARD["unclear_on_decided"]
    if label != gold_label:
        return REWARD["false_alarm"] if label == "contradiction" else REWARD["miss"]
    if label == "non_contradiction" and answer["type"] == gold_type:
        return REWARD["right_label"] + REWARD["right_type_bonus"]
    return REWARD["right_label"]


class FingerprintError(Exception):
    """The reward fingerprint cannot be sure which code it covers."""


def module_file(name):
    """The source file of a dotted `judge` module name, or None when there is no such file."""
    path = SOURCE_ROOT.joinpath(*name.split("."))
    for candidate in (path / "__init__.py", path.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    return None


def imported_modules(path, text):
    """The `judge` modules one file imports, read from its code. Stops on code it cannot follow."""
    where = path.relative_to(SOURCE_ROOT)
    package = where.parent.parts

    def stop(why):
        return FingerprintError(f"the reward fingerprint cannot follow {where.as_posix()}: {why}")

    def check_import(name):
        top = name.split(".")[0]
        if top in UNFOLLOWABLE or (top != "judge" and module_file(top)):  # project code beside `judge`
            raise stop(f"it imports {name}")

    found = set()
    for node in ast.walk(ast.parse(text, filename=str(path))):
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if name in UNFOLLOWABLE:
            raise stop(f"it uses {name}")
        # `judge.x` works without importing x, once any other file has imported it.
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "judge":
            if module_file(f"judge.{node.attr}"):
                found.add(f"judge.{node.attr}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                check_import(alias.name)
                if alias.name == "judge":
                    raise stop("`import judge` can reach any module, so import the modules by name")
                if alias.name.startswith("judge."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > len(package):
                raise stop("a relative import leaves the judge package")
            start = package[:len(package) - node.level + 1] if node.level else ()
            base = ".".join(start + tuple(filter(None, [node.module])))
            check_import(base)
            if base.split(".")[0] != "judge":
                continue
            found.add(base)
            for alias in node.names:
                if alias.name == "*":
                    raise stop(f"`from {base} import *` hides what it imports")
                if module_file(f"{base}.{alias.name}"):
                    found.add(f"{base}.{alias.name}")
    for name in list(found):
        while "." in name:  # importing a.b.c runs a.b too
            name = name.rpartition(".")[0]
            found.add(name)
    missing = sorted(name for name in found if module_file(name) is None)
    if missing:
        raise stop(f"no source file for {missing}")
    return found


def fingerprinted_files():
    """The source files behind the reward: {path inside the source folder: full text}.

    Starts from this file and `judge/__init__.py` (Python runs it on any `judge`
    import) and follows every `judge` import, so a new module needs no list.
    """
    files, todo = {}, ["judge", "judge.reward"]
    while todo:
        path = module_file(todo.pop())
        key = path.relative_to(SOURCE_ROOT).as_posix()
        if key not in files:
            files[key] = path.read_bytes().decode("utf-8")
            todo += imported_modules(path, files[key])
    return files


def reward_sha256():
    """SHA-256 of the full text of the source files behind the reward, with their paths, in sorted order (D-048).

    It covers `fingerprinted_files()`: the table, the reader, how a completion
    is read and the wrapper TRL calls. So any edit to those files gives a new
    fingerprint, even a comment (then recheck the saved pool offline). Paths
    start at the package, so the value is the same in any folder.

    It does not cover Python or outside packages (`json`, `re`; their versions
    are in `uv.lock` and the pinned image), data files or settings the code
    might read, or changes made in memory while a program runs: it reads the
    files on disk when called. It stops on the plain ways of loading code it
    cannot follow (UNFOLLOWABLE, project code beside `judge`), but it is a check
    against mistakes, not against someone hiding code on purpose.
    """
    return hashlib.sha256(json.dumps(fingerprinted_files(), sort_keys=True).encode("utf-8")).hexdigest()


def completion_text(completion):
    """The answer text from one TRL completion: a string, or a list of chat messages (conversational data)."""
    if isinstance(completion, str):
        return completion
    return completion[0]["content"]


class GRPOReward:
    """The reward in the shape TRL's GRPOTrainer calls it (TRL 1.15.0, checked 2026-10-27).

    TRL calls `reward_funcs` with `completions` and every dataset column (but
    `prompt`) as keyword arguments, one value per completion, and wants one
    float back per completion. The dataset needs the columns `gold_label` and
    `gold_type`. Each call also counts the answers by kind (the three labels
    and invalid), so Step 5 can see reward hacking such as one label every
    time: `last_counts` holds the latest call, `total_counts` all calls, and,
    when TRL passes `log_metric`, each kind's share is logged as a metric.
    """

    def __init__(self):
        self.__name__ = "judge_reward"  # TRL logs rewards/<name>/mean
        self.last_counts = dict.fromkeys(ANSWER_KINDS, 0)
        self.total_counts = dict.fromkeys(ANSWER_KINDS, 0)

    def __call__(self, completions, gold_label, gold_type, log_metric=None, **kwargs):
        if not len(completions) == len(gold_label) == len(gold_type):
            raise ValueError("completions, gold_label and gold_type differ in length")
        counts = dict.fromkeys(ANSWER_KINDS, 0)
        rewards = []
        for completion, label, type_ in zip(completions, gold_label, gold_type):
            text = completion_text(completion)
            counts[read_answer(text)[0] or "invalid"] += 1
            rewards.append(reward(text, label, type_))
        self.last_counts = counts
        for kind, n in counts.items():
            self.total_counts[kind] += n
        if log_metric is not None:
            for kind, n in counts.items():
                log_metric(f"answers/{kind}", n / len(completions))
        return rewards
