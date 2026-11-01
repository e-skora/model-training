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

# The folder that holds `judge`, the repo around it, and the names that load code by name (D-049).
SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SOURCE_ROOT.parent
LOADERS = {"importlib", "runpy", "__import__", "exec", "eval", "compile", "builtins", "__builtins__"}


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


def outside_code_names():
    """Top-level names of project code outside `src/`: `.py` files and folders in the repo root, `scripts/` and `tests/`."""
    names = set()
    for folder in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "tests"):
        if folder.is_dir():
            names |= {e.name.removesuffix(".py") for e in folder.iterdir() if e.is_dir() or e.suffix == ".py"}
    return names


def gives_re_another_meaning(node):
    """True when this code makes the name `re` mean something other than a plain `import re`."""
    if isinstance(node, ast.Import):
        return any(alias.asname == "re" for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return any((alias.asname or alias.name) == "re" for alias in node.names)
    if isinstance(node, ast.Name):
        return node.id == "re" and not isinstance(node.ctx, ast.Load)
    if isinstance(node, ast.arg):
        return node.arg == "re"
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler)) and node.name == "re"


def stop_problems(tree, outside):
    """What in one parsed file breaks stop 1 or stop 2: a list of (line, why). Strings and comments do not count."""
    nodes = list(ast.walk(tree))
    imports = [alias for node in nodes if isinstance(node, ast.Import) for alias in node.names]
    # `re.compile` builds a text pattern and loads no code (D-050). No other `compile` is allowed.
    re_is_the_module = any(alias.name == "re" for alias in imports) and not any(map(gives_re_another_meaning, nodes))
    sys_names = {"sys"} | {alias.asname for alias in imports if alias.name == "sys" and alias.asname}
    problems = []
    for node in nodes:
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.attr == "compile" and node.value.id == "re" and re_is_the_module:
                continue
            if node.attr == "path" and node.value.id in sys_names:
                problems.append((node.lineno, "it touches sys.path"))
        if name in LOADERS:
            problems.append((node.lineno, f"it uses {name}"))
        if isinstance(node, ast.Import):
            tops, names = [alias.name.split(".")[0] for alias in node.names], []
        elif isinstance(node, ast.ImportFrom):
            tops = [node.module.split(".")[0]] if not node.level else []
            names = [alias.name for alias in node.names]
        else:
            continue
        for found in tops + names:
            if found in LOADERS:
                problems.append((node.lineno, f"it imports {found}"))
        if "sys" in tops and ("path" in names or "*" in names):
            problems.append((node.lineno, "it imports sys.path"))
        for top in tops:
            if top in outside:
                problems.append((node.lineno, f"it imports {top}, which is project code outside src/"))
    return problems


def fingerprinted_files():
    """Every file the fingerprint covers: {path inside `src/`: bytes}, sorted by path. Stops if a file breaks a stop."""
    if SOURCE_ROOT.name != "src":
        raise FingerprintError(f"the reward fingerprint needs the repo's src folder, and judge is in {SOURCE_ROOT}")
    outside = outside_code_names()
    files = {}
    for path in SOURCE_ROOT.rglob("*"):
        key = path.relative_to(SOURCE_ROOT)
        if "__pycache__" in key.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise FingerprintError(f"the reward fingerprint stops at {key}: a link could point outside src/")
        if not path.is_file():
            continue
        files[key.as_posix()] = path.read_bytes()
        if path.suffix == ".py":
            problems = stop_problems(ast.parse(files[key.as_posix()], filename=str(key)), outside)
            if problems:
                line, why = min(problems)
                raise FingerprintError(f"the reward fingerprint stops at {key.as_posix()}, line {line}: {why}")
    return dict(sorted(files.items()))


def reward_sha256():
    """SHA-256 of every file under `src/`: each path inside `src/`, then its bytes, in sorted order (D-049).

    Covered: all of `src/` (`fingerprinted_files()`), whatever imports it, so
    the table, the reader, how a completion is read and the wrapper TRL calls
    all count, and so does a file the reward never uses. Any edit there, even
    a comment or a new file, gives a new fingerprint (then recheck the saved
    pool offline). Only `__pycache__` folders and `.pyc` files are skipped.
    Paths start at `src/`, so the value is the same in any folder.

    It stops with an error rather than give a wrong answer, so code outside
    `src/` cannot slip in unnoticed:
    - Stop 1: a file loads code by name (LOADERS, with `re.compile` the only
      use of `compile` allowed, D-050) or touches `sys.path`.
    - Stop 2: a file imports a name that is a `.py` file or a folder in the
      repo root, `scripts/` or `tests/`.
    It also stops on a link under `src/`, and when `judge` is not in a folder
    called `src` (on Modal or from an install, take the fingerprint on the Mac).

    Not covered: Python and outside packages (`json`, `re`; their versions are
    in `uv.lock` and the pinned image); data and settings files outside `src/`;
    changes made in memory while a program runs, because it reads the files on
    disk when called; wrappers outside `src/`, such as Step 5's script; and
    `.pyc` files. The stops read the code for the plain ways of loading code
    named above. They are a check against mistakes, not against someone hiding
    code on purpose, and rarer loaders such as `pkgutil` are not on the list.
    """
    digest = hashlib.sha256()
    for key, data in fingerprinted_files().items():
        digest.update(f"{key}\0{len(data)}\0".encode("utf-8") + data)
    return digest.hexdigest()


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
