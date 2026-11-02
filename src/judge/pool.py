"""The RL pool (M2 Step 3 part B): which pairs GRPO trains on, by the D-046 rule.

The rule is fixed here, in code, before any of v1's answers is seen:
- v1 answers each candidate 8 times at temperature 1.0;
- mixed (the 8 rewards are not all the same): in the pool, because only then can GRPO learn;
- always right: a random ALWAYS_RIGHT_SHARE of the mixed count is added (KEEP_SEED), so v1 does not forget them;
- always wrong the same way: listed for Claude in Cowork to check the label, and kept out.

This file imports no GPU libraries, so tests run it on the Mac.
"""

import random

from judge.format import format_pairs
from judge.reward import ANSWER_KINDS, read_answer, reward

SAMPLES = 8  # answers per candidate
SHOW_SEED = 2026102704  # which candidates are shown swapped or without dates (judge.format's rules)
KEEP_SEED = 2026102705  # which always-right pairs join the pool
ALWAYS_RIGHT_SHARE = 0.2
M1_UNCLEAR_FIELDS = {"source": "synthetic", "label_by": "claude-code", "hard": "m1_unclear"}  # D-045, pool only


def candidates(hard_pairs, m1_pairs):
    """The M2 hard training pairs, then M1's `unclear` training pairs with the D-045 fields added (copies)."""
    m1_unclear = [{**pair, **M1_UNCLEAR_FIELDS} for pair in m1_pairs if pair["label"] == "unclear"]
    pairs = list(hard_pairs) + m1_unclear
    ids = [pair["id"] for pair in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("two candidates share an id")
    return pairs


def version_name(swapped, dates):
    if swapped and not dates:
        return "swapped_no_dates"
    return "swapped" if swapped else "as_is" if dates else "no_dates"


def show(pairs, seed=SHOW_SEED):
    """Each pair as training will show it: {"id", "swapped", "dates", "version", "messages"}.

    Swapping and dates follow `judge.format.format_pairs` (half swapped, a
    third without dates, chosen independently), so some are both.
    """
    shown = []
    for example in format_pairs(pairs, seed=seed):
        shown.append({
            "id": example["id"],
            "swapped": example["swapped"],
            "dates": example["dates"],
            "version": version_name(example["swapped"], example["dates"]),
            "messages": example["messages"][:-1],  # the prompt only, without the right answer
        })
    return shown


def answer_kind(raw):
    return read_answer(raw)[0] or "invalid"


def judge_samples(pair, raws):
    """Rewards, answer kinds and the outcome for one pair's sampled answers."""
    rewards = [reward(raw, pair["label"], pair["type"]) for raw in raws]
    kinds = [answer_kind(raw) for raw in raws]
    counts = {kind: kinds.count(kind) for kind in ANSWER_KINDS}
    if len(set(rewards)) > 1:
        outcome = "mixed"
    elif kinds[0] == pair["label"]:
        outcome = "always_right"
    else:
        outcome = "always_wrong"  # equal rewards mean the same kind of wrong answer every time
    return {"rewards": rewards, "kinds": kinds, "answer_counts": counts, "outcome": outcome}


def choose(judged, seed=KEEP_SEED):
    """Split judged pairs ({"id", "outcome", ...}) into (pool ids, always-wrong ids), in the order given.

    The pool is every mixed pair plus round(ALWAYS_RIGHT_SHARE * mixed) always-right
    pairs drawn with `seed` (all of them if there are fewer).
    """
    mixed = [row["id"] for row in judged if row["outcome"] == "mixed"]
    right = sorted(row["id"] for row in judged if row["outcome"] == "always_right")
    keep = set(random.Random(seed).sample(right, min(round(ALWAYS_RIGHT_SHARE * len(mixed)), len(right))))
    pool = [row["id"] for row in judged if row["id"] in keep or row["outcome"] == "mixed"]
    wrong = [row["id"] for row in judged if row["outcome"] == "always_wrong"]
    return pool, wrong
