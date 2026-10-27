"""Tests for judge.reward with made-up answers. No GPU, no network, no files."""

import json
import re

import pytest

import judge.reward as reward_module
import judge.score as score_module
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


def test_fingerprint_is_stable_and_changes_with_a_value(monkeypatch):
    before = reward_sha256()
    assert before == reward_sha256() and len(before) == 64
    for key in REWARD:
        monkeypatch.setitem(reward_module.REWARD, key, REWARD[key] + 0.1)
        assert reward_sha256() != before, key
        monkeypatch.undo()
    assert reward_sha256() == before


# Codex review M2-04, finding 1: each change below alters what earns points, and before the fix
# the fingerprint stayed the same for all of them. (name, replacement, answer, right label, right type)
READER_CHANGES = [
    ("TYPES", ("compatible_detail", "merely_related"), ans("non_contradiction", "update"), "non_contradiction", "update"),
    ("no_repeated_fields", dict, '{"label": "contradiction", "label": "contradiction", "type": null, "reason": "r"}',
     "contradiction", None),
    ("reject_constant", lambda name: None, '{"label": "contradiction", "type": NaN, "reason": "r"}', "contradiction", None),
    ("FENCE", re.compile(r"\A~~~[a-zA-Z]*\s*\n(.*?)\n?~~~\Z", re.DOTALL), "~~~json\n" + ans("contradiction") + "\n~~~",
     "contradiction", None),
]


@pytest.mark.parametrize("name, replacement, raw, gold_label, gold_type", READER_CHANGES)
def test_fingerprint_changes_with_the_answer_reader(monkeypatch, name, replacement, raw, gold_label, gold_type):
    before_sha, before_points = reward_sha256(), reward(raw, gold_label, gold_type)
    monkeypatch.setattr(score_module, name, replacement)
    assert reward(raw, gold_label, gold_type) != before_points  # the change matters to the reward
    assert reward_sha256() != before_sha


def test_fingerprint_changes_with_how_a_completion_is_read(monkeypatch):
    before = reward_sha256()
    monkeypatch.setattr(reward_module, "completion_text", lambda completion: completion[-1]["content"])
    assert reward_sha256() != before


def test_fingerprint_covers_the_whole_reader():
    parts = reward_module.rule_parts(reward, reward_module.completion_text)
    for name in ("REWARD", "LABELS", "reward", "read_answer", "completion_text"):
        assert f"judge.reward.{name}" in parts
    for name in ("parse_answer", "format_problem", "reject_constant", "no_repeated_fields", "RepeatedField",
                 "FENCE", "LABELS", "TYPES"):
        assert f"judge.score.{name}" in parts
