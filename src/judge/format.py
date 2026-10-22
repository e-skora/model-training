"""Turn memory pairs into chat training examples: the prompt's messages plus the right JSON answer.

Order means nothing and dates may be missing (D-026), so the model must see
both. With the fixed seed below, about half the examples show the later
memory first, and about a third leave the dates out. The data files keep the
earlier memory as `a`; the shuffling happens only here.

Run from the repo folder:
    uv run python -m judge.format data/m1/train.jsonl --out /some/scratch/train-chat.jsonl
"""

import json
import random
import sys

from judge.prompt import build_messages

SEED = 20261022
SWAP_SHARE = 1 / 2
NO_DATES_SHARE = 1 / 3


def answer_text(pair):
    """The right answer, written exactly as the prompt asks."""
    return json.dumps({"label": pair["label"], "type": pair["type"], "reason": pair["reason"]}, ensure_ascii=False)


def to_example(pair, swap=False, dates=True):
    """One chat example. `swap` shows memory b as Memory A; `dates=False` leaves both dates out."""
    shown = {**pair, "a": pair["b"], "b": pair["a"]} if swap else pair
    messages = build_messages(shown, dates=dates)
    messages.append({"role": "assistant", "content": answer_text(pair)})
    return {"id": pair["id"], "swapped": swap, "dates": dates, "messages": messages}


def format_pairs(pairs, seed=SEED):
    """Chat examples for all pairs, in the same order.

    Exactly round(n / 2) examples are swapped and round(n / 3) have no dates,
    each picked at random with `seed`, independently of each other.
    """
    rng = random.Random(seed)
    indexes = range(len(pairs))
    swapped = set(rng.sample(indexes, round(len(pairs) * SWAP_SHARE)))
    no_dates = set(rng.sample(indexes, round(len(pairs) * NO_DATES_SHARE)))
    return [to_example(pair, swap=i in swapped, dates=i not in no_dates) for i, pair in enumerate(pairs)]


def main(argv):
    if len(argv) != 4 or argv[2] != "--out":
        sys.exit("usage: python -m judge.format <pairs.jsonl> --out <examples.jsonl>")
    with open(argv[1], encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    examples = format_pairs(pairs)
    with open(argv[3], "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"{len(examples)} examples, {sum(e['swapped'] for e in examples)} swapped, "
          f"{sum(not e['dates'] for e in examples)} without dates, seed {SEED}")


if __name__ == "__main__":
    main(sys.argv)
