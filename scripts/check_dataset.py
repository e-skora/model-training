"""Check a memory-pair dataset file (JSONL) before it is used.

Prints how many pairs there are per label and type, and how many different
people appear. Stops with a clear message at the first problem it finds.

Run from the repo folder:
    uv run python scripts/check_dataset.py data/m1/test.jsonl
"""

import json
import re
import sys
from collections import Counter
from datetime import date

FIELDS = ("id", "person", "a", "b", "label", "type", "reason")
MEMORY_FIELDS = ("text", "written_at")
LABELS = ("contradiction", "non_contradiction")
TYPES = ("update", "compatible_detail", "merely_related")
# Names that must never appear (they belong to Aeverum's own data).
BANNED = (re.compile(r"\bmara\b", re.IGNORECASE), re.compile(r"lindqvist", re.IGNORECASE))


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
    if label == "contradiction" and kind is not None:
        fail(line_no, f"type is {kind!r} on a contradiction; it must be null")
    if label == "non_contradiction" and kind not in TYPES:
        fail(line_no, f"type is {kind!r} on a non_contradiction; it must be one of {TYPES}")

    a_date = parse_date(pair["a"]["written_at"], line_no, "a")
    b_date = parse_date(pair["b"]["written_at"], line_no, "b")
    if kind == "update" and b_date <= a_date:
        fail(line_no, "an update needs memory b written later than memory a (reading rule 2)")

    first_name = pair["person"].split()[0]
    for which in ("a", "b"):
        if first_name not in pair[which]["text"]:
            fail(line_no, f"memory {which} does not name {first_name}")

    for text in (pair["person"], pair["a"]["text"], pair["b"]["text"], pair["reason"]):
        for pattern in BANNED:
            if pattern.search(text):
                fail(line_no, f"banned name found ({pattern.pattern}) in: {text}")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/check_dataset.py <file.jsonl>")
    path = sys.argv[1]
    try:
        open(path, encoding="utf-8").close()
    except OSError as e:
        fail(None, f"cannot open {path} ({e.strerror})")

    seen_ids, seen_texts = {}, {}
    labels, types, people = Counter(), Counter(), set()
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

    if not seen_ids:
        fail(None, f"{path} has no pairs")
    print(f"File: {path}")
    print(f"Pairs: {sum(labels.values())}")
    for label in LABELS:
        print(f"  {label}: {labels[label]}")
    for kind in TYPES:
        print(f"    {kind}: {types[kind]}")
    print(f"Different people: {len(people)}")
    print("PASS")


if __name__ == "__main__":
    main()
