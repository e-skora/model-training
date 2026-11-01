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

# This file as Python found it, links not followed, so the fingerprint can check that path (D-052).
THIS_FILE = Path(__file__).absolute()
# The folder that holds `judge`, the repo around it, and the names that load code by name (D-049).
SOURCE_ROOT = THIS_FILE.parents[1]
REPO_ROOT = SOURCE_ROOT.parent
LOADERS = {"importlib", "runpy", "__import__", "exec", "eval", "compile", "builtins", "__builtins__"}
# All a file under `src/` may do with `sys`, after one plain `import sys` (D-052).
SYS_ALLOWED = {"argv", "exit", "stdout", "stderr"}
# Code whose text fields name something without giving it a value.
BINDS_NOTHING = (ast.Attribute, ast.keyword, ast.MatchClass, ast.ImportFrom)


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


def written_names(node):
    """Every name written in this piece of code: each text field, with `a.b` cut to `a`. A string's text is not a name."""
    if isinstance(node, ast.Constant):
        return []
    fields = (value if isinstance(value, list) else [value] for _, value in ast.iter_fields(node))
    return [item.split(".")[0] for field in fields for item in field if isinstance(item, str)]


def bound_names(node):
    """The names this piece of code gives a value. A written name counts unless it is known not to, so new syntax stops."""
    if isinstance(node, ast.alias):
        return [node.asname or node.name.split(".")[0]]
    if isinstance(node, BINDS_NOTHING) or (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
        return []
    return written_names(node)


def plain_import(tree, name):
    """The first `import name` at the top of a file: the one binding of `sys` or `re` allowed there (D-052), or None."""
    imports = [statement for statement in tree.body if isinstance(statement, ast.Import)]
    return next((alias for statement in imports for alias in statement.names if alias.name == name and not alias.asname), None)


def stop_problems(tree, outside):
    """What in one parsed file breaks a stop: a list of (line, why). Strings and comments do not count."""
    nodes = list(ast.walk(tree))
    plain_re, plain_sys = plain_import(tree, "re"), plain_import(tree, "sys")
    star_import = any(isinstance(node, ast.alias) and node.name == "*" for node in nodes)
    other_re = [node for node in nodes if "re" in bound_names(node) and node is not plain_re]
    # `re.compile` builds a text pattern and loads no code (D-050). No other `compile` is allowed.
    re_is_the_module = plain_re is not None and not star_import and not other_re
    sys_reads = set()  # the `sys` in each allowed read such as `sys.argv`
    if plain_sys is not None and not star_import:
        sys_reads = {node.value for node in nodes
                     if isinstance(node, ast.Attribute) and node.attr in SYS_ALLOWED and isinstance(node.ctx, ast.Load)}
    problems = [(node.lineno, "it binds re in a way other than one plain top-level import") for node in other_re]
    for node in nodes:
        if "__path__" in written_names(node):
            problems.append((node.lineno, "it uses __path__"))
        if "sys" in written_names(node) and node is not plain_sys and node not in sys_reads:
            problems.append((node.lineno, "it uses sys, and only one plain top-level import and "
                                          "reading sys.argv, sys.exit, sys.stdout or sys.stderr may pass"))
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if isinstance(node, ast.Attribute) and node.attr == "compile" and isinstance(node.value, ast.Name) and node.value.id == "re":
            if not re_is_the_module:
                problems.append((node.lineno, "re.compile needs one plain top-level import re and no star import"))
            continue
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
        for top in tops:
            if top in outside:
                problems.append((node.lineno, f"it imports {top}, which is project code outside src/"))
    return problems


def fingerprinted_files():
    """Every file the fingerprint covers: {path inside `src/`: bytes}, sorted by path. Stops if a file breaks a stop."""
    for path in (SOURCE_ROOT, THIS_FILE.parent, THIS_FILE):
        if path.is_symlink():
            raise FingerprintError(f"the reward fingerprint stops at {path}: a link would hide which folder is read")
    if SOURCE_ROOT.name != "src":
        raise FingerprintError(f"the reward fingerprint needs the repo's src folder, and judge is in {SOURCE_ROOT}")
    outside = outside_code_names()
    files = {}
    for path in SOURCE_ROOT.rglob("*"):
        key = path.relative_to(SOURCE_ROOT)
        if path.is_symlink():
            raise FingerprintError(f"the reward fingerprint stops at {key}: a link could point outside src/")
        if "__pycache__" in key.parent.parts or path.suffix == ".pyc" or not path.is_file():
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
    pool offline). Only `.pyc` files, folders called `__pycache__` and what is
    inside those folders are skipped; a file called `__pycache__` is covered.
    Paths start at `src/`, so the value is the same in any folder.

    It stops with an error rather than give a wrong answer, so code outside
    `src/` cannot slip in unnoticed. Most of these are allow-lists (D-052):
    they say what may pass, and everything else stops. A full `ast` check
    reads them, because it gives the line. Every text field of the code counts,
    so a name in a new kind of syntax stops rather than slips by. `symtable`
    is not used, because it gives no line for a name.
    - `sys`: one plain `import sys` at the top of the file, then reading
      `sys.argv`, `sys.exit`, `sys.stdout` or `sys.stderr`. Any other use of
      the name stops, and so does any of those reads in a file with a star
      import, because the star could bind `sys`.
    - `re`: one plain `import re` at the top of the file is its only binding.
      Any other binding stops, even in a file that never calls `re.compile`.
    - `re.compile`, the only `compile` allowed (D-050): it also needs that
      plain `import re` and no star import in the file.
    - `__path__`: any use stops.
    - Links: `src`, `src/judge` and `src/judge/reward.py` are checked as Python
      found them, before anything else, and so is everything under `src/`,
      skipped places included. Any link stops.
    It also stops, as a block-list:
    - a file loads code by name (LOADERS);
    - a file imports a name that is a `.py` file or a folder in the repo root,
      `scripts/` or `tests/`;
    - `judge` is not in a folder called `src` (on Modal or from an install,
      take the fingerprint on the Mac).

    Not covered: Python and outside packages (`json`, `re`; their versions are
    in `uv.lock` and the pinned image); data and settings files outside `src/`;
    changes made in memory while a program runs, because it reads the files on
    disk when called; wrappers outside `src/`, such as Step 5's script; and
    `.pyc` files. It catches honest changes, not code written to fool it
    (D-053), and rarer loaders such as `pkgutil` are not on the list.
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
