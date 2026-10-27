"""The reward function for GRPO: points for one model answer, given the right label and type.

Every number lives in REWARD below, so a change touches one place and shows
in `reward_sha256()` (D-040, D-046). Answers are read with the scorer's own
strict reader (`judge.score.parse_answer`) and its field check
(`judge.score.format_problem`), so "valid JSON in the exact format" means
exactly what scoring means. The `reason` text earns nothing.

This file imports no GPU libraries, so tests run it on the Mac.
"""

import hashlib
import inspect
import json
import re

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


def describe(value):
    """A value the reward rules use, as JSON: source text for code, pattern and flags for a regex."""
    if isinstance(value, re.Pattern):
        return {"pattern": value.pattern, "flags": value.flags}
    if inspect.isfunction(value) or inspect.isclass(value):
        try:
            return inspect.getsource(value)
        except (OSError, TypeError):  # a built-in, such as dict
            return repr(value)
    if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
        return value
    raise TypeError(f"the reward fingerprint cannot describe {value!r}")


def rule_parts(*roots):
    """Every function, class and constant the roots use, found by following the names in their code.

    Each function's module-level names are followed (modules such as `json`
    are skipped), so a helper or constant added later is fingerprinted
    without anyone listing it.
    """
    parts, seen, todo = {}, set(), list(roots)
    while todo:
        fn = todo.pop()
        if fn in seen:
            continue
        seen.add(fn)
        parts[f"{fn.__module__}.{fn.__qualname__}"] = describe(fn)
        codes = [fn.__code__]
        while codes:
            code = codes.pop()
            codes += [c for c in code.co_consts if inspect.iscode(c)]
            for name in code.co_names:
                if name not in fn.__globals__:
                    continue  # a built-in or an attribute name
                value = fn.__globals__[name]
                if inspect.isfunction(value):
                    todo.append(value)
                elif not inspect.ismodule(value):
                    parts[f"{fn.__globals__['__name__']}.{name}"] = describe(value)
    return parts


def reward_sha256():
    """SHA-256 of the reward table and the code that applies it, the way `prompt_sha256()` fingerprints the prompt.

    It covers everything `reward` and `completion_text` use, followed name by
    name: the table, the scorer's reader (`parse_answer`, `format_problem`)
    with its helpers, labels, types and code-fence pattern. So a change to a
    value, a rule, what counts as valid, or how a completion is read gives a
    new fingerprint. Comments inside that code change it too (harmless).
    """
    rules = rule_parts(reward, completion_text)
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode("utf-8")).hexdigest()


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
