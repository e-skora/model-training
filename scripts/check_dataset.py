"""Check a memory-pair dataset file (JSONL) before it is used.

Prints how many pairs there are per label and type, how many different
people appear, how many pairs per label and type contain a change word
(a word like "now" or "moved" that often marks an update) and how many have
both memories on the same day. Stops with a clear message at the first
problem it finds.

Quality checks on the file itself (the test-set lessons in LESSONS.md):
- each label and type with at least 20 pairs has change words in at least
  10% of its pairs, and at least 10% of updates have none (said indirectly);
- each label and type with at least 20 pairs, except updates, has between
  35% and 65% of its pairs written on the same day;
- no sentence pattern is used more than 3 times in any block of 100 pairs:
  a memory may not look like (TEMPLATE_SIMILARITY or more shared words)
  memories in 3 or more other pairs of its block;
- `unclear` reasons do not follow one template: no more than 3 in any 100
  of them start with the same REASON_OPENING words (the person's names
  count as one word, "NAME").

Trap tags (M2): when any pair has a `hard` field, every pair must have one,
it must be one of HARD_TAGS, and it must fit the pair's label and type.
The checker then also prints, per trap, how many pairs, how many with a
change word and how many on the same day.

Walls between files:
- `--against <test.jsonl>`: no name of a test person appears anywhere in
  the file, and no pair is a near copy of a test pair;
- `--distinct-from <other.jsonl>`: no pair repeats or nearly copies a pair
  of the other file (for train and valid). For a file with trap tags (M2),
  no name of a person in the other file may appear either. M1's own files
  keep the old rule: M1 train and valid share 7 name words, and M1 is
  frozen and published.
A near copy shares more than NEAR_COPY of its words with the other pair:
the words of both memories, without the person's name, counted as
shared / all different words of the two pairs together.

Run from the repo folder:
    uv run python scripts/check_dataset.py data/m1/test.jsonl
    uv run python scripts/check_dataset.py data/m1/train.jsonl --against data/m1/test.jsonl --distinct-from data/m1/valid.jsonl
    uv run python scripts/check_dataset.py data/m2/test-hard.jsonl --against data/m1/test.jsonl --distinct-from data/m1/train.jsonl --distinct-from data/m1/valid.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date

FIELDS = ("id", "person", "a", "b", "label", "type", "reason")
MEMORY_FIELDS = ("text", "written_at")
LABELS = ("contradiction", "non_contradiction", "unclear")
TYPES = ("update", "compatible_detail", "merely_related")
# Names that must never appear (they belong to Aeverum's own data).
BANNED = (re.compile(r"\bmara\b", re.IGNORECASE), re.compile(r"lindqvist", re.IGNORECASE))
# Words that often mark a change. If only updates had them, a model could
# score well by spotting the word instead of reading the memories.
CHANGE_WORD_LIST = ("now", "moved", "switched", "since", "stopped", "later", "replaced")
CHANGE_WORDS = re.compile(r"\b(" + "|".join(CHANGE_WORD_LIST) + r")\b", re.IGNORECASE)
WORD = re.compile(r"[^\W\d_]+|\d+")
NEAR_COPY = 0.6
TEMPLATE_SIMILARITY = 0.6
BLOCK = 100
MIN_GROUP = 20  # quality shares are only checked for groups this big
REASON_OPENING = 3  # words that make up the start of a reason
REASON_MAX = 3  # times one start may be used in any REASON_WINDOW unclear reasons
REASON_WINDOW = 100
# M2 trap tags and the (label, type) each one may have (prompt M2-01).
CONTRADICTION = {("contradiction", None)}
HARD_TAGS = {
    "explain_away": CONTRADICTION,
    "paraphrase": CONTRADICTION,
    "small_difference": CONTRADICTION,
    "change_word_contradiction": CONTRADICTION,
    "far_apart_no_change": CONTRADICTION,
    "indirect_update": {("non_contradiction", "update")},
    "update_with_overlap": {("non_contradiction", "update")},
    "looks_like_clash": {("non_contradiction", "compatible_detail")},
    "near_identical": {("non_contradiction", "compatible_detail"), ("non_contradiction", "merely_related")},
    "rule5_bait": {("non_contradiction", "merely_related")},
}


def fail(line_no, message):
    where = f"line {line_no}: " if line_no else ""
    sys.exit(f"FAIL: {where}{message}")


def parse_date(value, line_no, which):
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        fail(line_no, f"memory {which} has written_at {value!r}, which is not a YYYY-MM-DD date")


def check_pair(pair, line_no):
    for field in FIELDS:
        if field not in pair:
            fail(line_no, f"missing field {field!r}")
    for which in ("a", "b"):
        memory = pair[which]
        if not isinstance(memory, dict):
            fail(line_no, f"memory {which} is not an object")
        for field in MEMORY_FIELDS:
            if field not in memory:
                fail(line_no, f"memory {which} is missing field {field!r}")
        if not isinstance(memory["text"], str) or not memory["text"].strip():
            fail(line_no, f"memory {which} has no text")

    label, kind = pair["label"], pair["type"]
    if label not in LABELS:
        fail(line_no, f"label {label!r} is not one of {LABELS}")
    if label in ("contradiction", "unclear") and kind is not None:
        fail(line_no, f"type is {kind!r} on {label}; it must be null")
    if label == "non_contradiction" and kind not in TYPES:
        fail(line_no, f"type is {kind!r} on a non_contradiction; it must be one of {TYPES}")

    a_date = parse_date(pair["a"]["written_at"], line_no, "a")
    b_date = parse_date(pair["b"]["written_at"], line_no, "b")
    if b_date < a_date:
        fail(line_no, "memory b is dated earlier than memory a; a must never be the later one")
    if kind == "update" and b_date <= a_date:
        fail(line_no, "an update needs memory b written later than memory a (reading rule 2)")

    # Whole-word match, so "Ana" inside "Anaïs" or "Banana" does not count.
    first_name = pair["person"].split()[0]
    name_pattern = re.compile(rf"(?<!\w){re.escape(first_name)}(?!\w)")
    for which in ("a", "b"):
        if not name_pattern.search(pair[which]["text"]):
            fail(line_no, f"memory {which} does not name {first_name}")

    for text in (pair["person"], pair["a"]["text"], pair["b"]["text"], pair["reason"]):
        for pattern in BANNED:
            if pattern.search(text):
                fail(line_no, f"banned name found ({pattern.pattern}) in: {text}")


def words(text, person):
    """The set of lowercase words in a text, leaving out the person's names and a possessive "s"."""
    names = set(person.split())
    return frozenset(w.lower() for w in WORD.findall(text) if w not in names and w != "s")


def pair_words(pair):
    return words(pair["a"]["text"] + " " + pair["b"]["text"], pair["person"])


def similarity(x, y):
    return len(x & y) / len(x | y) if x | y else 0.0


def group(pair):
    return pair["type"] or pair["label"]


def read_pairs(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [(line_no, json.loads(line)) for line_no, line in enumerate(f, start=1) if line.strip()]
    except OSError as e:
        fail(None, f"cannot open {path} ({e.strerror})")
    except json.JSONDecodeError as e:
        fail(None, f"{path} is not valid JSONL ({e})")


def check_templates(pairs):
    """No memory may look like memories in 3 or more other pairs of its block of 100."""
    for start in range(0, len(pairs), BLOCK):
        block = pairs[start : start + BLOCK]
        memories = [(i, line_no, pair[w]["text"], words(pair[w]["text"], pair["person"]))
                    for i, (line_no, pair) in enumerate(block) for w in ("a", "b")]
        for i, line_no, text, these in memories:
            alike = {j for j, _, _, other in memories if j != i and similarity(these, other) >= TEMPLATE_SIMILARITY}
            if len(alike) >= 3:
                examples = sorted({block[j][1]["a"]["text"] for j in alike})[:3]
                fail(line_no, f"sentence pattern used more than 3 times in lines {start + 1}-{start + len(block)}: "
                              f"{text!r} looks like memories in {len(alike)} other pairs, such as {examples}")


def reason_opening(pair):
    """The first REASON_OPENING words of a reason, lowercase, with the person's names as NAME and no possessive "s"."""
    names = set(WORD.findall(pair["person"]))
    tokens = ["NAME" if w in names else w.lower() for w in WORD.findall(pair["reason"]) if w != "s"]
    return " ".join(tokens[:REASON_OPENING])


def check_reasons(pairs):
    """No start of an `unclear` reason is used more than REASON_MAX times in any REASON_WINDOW of them, in file order."""
    unclear = [(line_no, reason_opening(pair)) for line_no, pair in pairs if pair["label"] == "unclear"]
    for start in range(max(1, len(unclear) - REASON_WINDOW + 1)):
        window = unclear[start : start + REASON_WINDOW]
        if not window:
            return
        opening, count = Counter(o for _, o in window).most_common(1)[0]
        if count > REASON_MAX:
            line_no = [n for n, o in window if o == opening][REASON_MAX]
            fail(line_no, f"{count} of {len(window)} unclear reasons in a row start {opening!r}; at most {REASON_MAX} may")


def check_quality(labels, types, changed, same_day):
    groups = {"contradiction": labels["contradiction"], "unclear": labels["unclear"], **types}
    for name, count in groups.items():
        if count < MIN_GROUP:
            continue
        if name == "update":
            if count - changed[name] < 0.10 * count:
                fail(None, f"only {count - changed[name]} of {count} updates are said without a change word; need at least 10%")
            continue
        if changed[name] < 0.10 * count:
            fail(None, f"only {changed[name]} of {count} {name} pairs contain a change word; need at least 10%")
        if not 0.35 <= same_day[name] / count <= 0.65:
            fail(None, f"{same_day[name]} of {count} {name} pairs are same-day; need about half (35% to 65%)")


def check_hard(pairs):
    """If any pair has a trap tag, every pair needs one that is known and fits its label and type."""
    if not any("hard" in pair for _, pair in pairs):
        return False
    for line_no, pair in pairs:
        if "hard" not in pair:
            fail(line_no, "missing field 'hard'; when one pair has a trap tag, every pair needs one")
        tag = pair["hard"]
        if tag not in HARD_TAGS:
            fail(line_no, f"hard {tag!r} is not one of {tuple(HARD_TAGS)}")
        if (pair["label"], pair["type"]) not in HARD_TAGS[tag]:
            fail(line_no, f"hard {tag!r} does not fit label {pair['label']!r} with type {pair['type']!r}")
    return True


def check_names(pairs, other_pairs, other_path):
    """No name word of a person in the other file appears anywhere in this file. Returns how many name words."""
    names = {token for pair in other_pairs for token in pair["person"].split() if token[:1].isupper()}
    name_pattern = re.compile(r"(?<!\w)(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")(?!\w)")
    for line_no, pair in pairs:
        for text in (pair["person"], pair["a"]["text"], pair["b"]["text"], pair["reason"]):
            match = name_pattern.search(text)
            if match:
                fail(line_no, f"name {match.group(1)!r} belongs to a person in {other_path}: {text}")
    return len(names)


def check_against(pairs, test_path):
    """The walls against the frozen test set: no test person's name, no near copy of a test pair."""
    test_pairs = [pair for _, pair in read_pairs(test_path)]
    names = check_names(pairs, test_pairs, test_path)
    check_near_copies(pairs, test_pairs, test_path)
    return names


def check_near_copies(pairs, other_pairs, other_path):
    other = [(pair, pair_words(pair), (pair["a"]["text"], pair["b"]["text"])) for pair in other_pairs]
    for line_no, pair in pairs:
        these = pair_words(pair)
        texts = {(pair["a"]["text"], pair["b"]["text"]), (pair["b"]["text"], pair["a"]["text"])}
        for other_pair, those, other_texts in other:
            if other_texts in texts:
                fail(line_no, f"pair repeats {other_pair['id']} of {other_path}")
            shared = similarity(these, those)
            if shared > NEAR_COPY:
                fail(line_no, f"pair is a near copy ({shared:.0%} shared words) of {other_pair['id']} of {other_path}")


def main():
    parser = argparse.ArgumentParser(description="Check a memory-pair dataset file (JSONL).")
    parser.add_argument("path")
    parser.add_argument("--against", help="the frozen test set: no shared names, no near copies")
    parser.add_argument("--distinct-from", action="append", default=[], help="another set: no repeated or near-copied pairs")
    args = parser.parse_args()
    path = args.path
    try:
        open(path, encoding="utf-8").close()
    except OSError as e:
        fail(None, f"cannot open {path} ({e.strerror})")

    seen_ids, seen_texts = {}, {}
    labels, types, people = Counter(), Counter(), set()
    changed, same_day = Counter(), Counter()
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                fail(line_no, "empty line")
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as e:
                fail(line_no, f"not valid JSON ({e})")
            if not isinstance(pair, dict):
                fail(line_no, "not a JSON object")
            check_pair(pair, line_no)

            if pair["id"] in seen_ids:
                fail(line_no, f"id {pair['id']!r} repeats (first seen on line {seen_ids[pair['id']]})")
            seen_ids[pair["id"]] = line_no
            texts = (pair["a"]["text"], pair["b"]["text"])
            if texts in seen_texts:
                fail(line_no, f"the (a, b) text pair repeats (first seen on line {seen_texts[texts]})")
            seen_texts[texts] = line_no

            labels[pair["label"]] += 1
            if pair["type"]:
                types[pair["type"]] += 1
            people.add(pair["person"])
            if any(CHANGE_WORDS.search(pair[which]["text"]) for which in ("a", "b")):
                changed[group(pair)] += 1
            if pair["a"]["written_at"] == pair["b"]["written_at"]:
                same_day[group(pair)] += 1
            pairs.append((line_no, pair))

    if not seen_ids:
        fail(None, f"{path} has no pairs")
    check_quality(labels, types, changed, same_day)
    check_templates(pairs)
    check_reasons(pairs)
    hard = check_hard(pairs)
    walls = []
    if args.against:
        names = check_against(pairs, args.against)
        walls.append(f"no names of the {names} name words in {args.against}, no near copies of its pairs")
    for other_path in args.distinct_from:
        other_pairs = [pair for _, pair in read_pairs(other_path)]
        check_near_copies(pairs, other_pairs, other_path)
        if hard:
            names = check_names(pairs, other_pairs, other_path)
            walls.append(f"no names of the {names} name words in {other_path}, no repeated or near-copied pairs")
        else:
            walls.append(f"no repeated or near-copied pairs from {other_path}")

    total = sum(labels.values())
    print(f"File: {path}")
    print(f"Pairs: {total}")
    for label in LABELS:
        print(f"  {label}: {labels[label]} ({labels[label] / total:.0%})")
        if label == "non_contradiction":
            for kind in TYPES:
                print(f"    {kind}: {types[kind]}")
    print(f"Different people: {len(people)}")
    groups = [("contradiction", labels["contradiction"])] + [(kind, types[kind]) for kind in TYPES] + [("unclear", labels["unclear"])]
    print("Pairs with a change word (" + ", ".join(CHANGE_WORD_LIST) + ") / same day:")
    for name, count in groups:
        print(f"  {name}: {changed[name]} of {count} / {same_day[name]} of {count}")
    if hard:
        traps = Counter((pair["hard"], group(pair)) for _, pair in pairs)
        trap_changed = Counter((pair["hard"], group(pair)) for _, pair in pairs
                               if any(CHANGE_WORDS.search(pair[which]["text"]) for which in ("a", "b")))
        trap_same_day = Counter((pair["hard"], group(pair)) for _, pair in pairs
                                if pair["a"]["written_at"] == pair["b"]["written_at"])
        print("Traps: pairs / with a change word / same day:")
        for tag in HARD_TAGS:
            for name in ("contradiction", *TYPES):
                key = (tag, name)
                if traps[key]:
                    print(f"  {tag} ({name}): {traps[key]} / {trap_changed[key]} / {trap_same_day[key]}")
    print(f"Sentence patterns: none used more than 3 times in any block of {BLOCK} pairs")
    print(f"Unclear reasons: no start of {REASON_OPENING} words used more than {REASON_MAX} times in any {REASON_WINDOW}")
    for wall in walls:
        print(f"Wall: {wall}")
    print("PASS")


if __name__ == "__main__":
    main()
