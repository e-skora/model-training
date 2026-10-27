# M2 hard test set

`test-hard.jsonl` is the test set for M2. It is 300 pairs of memories about made-up people. Each pair is built around a trap: a way a judge could get it wrong by reading the surface instead of the meaning.

It is used only to score models. **It is never used for training**, and no model has been run on it yet.

**Status:** written and checked by Claude Code on 2026-10-26. Not frozen yet. A separate agent reviews it blind first (D-042); then Claude in Cowork freezes it and records its fingerprint in `STATE.md`.

SHA-256 as written: `d9e8e5f3a53369981d11c835a0a4bf68b9b105ae0c8dd78b2d02df967645ff82`

## Made up, all of it

Every person and every memory in this file was made up by Claude Code. There is no real personal data. No name in it appears in any file in `data/m1/`, and it never uses Aeverum's data or the name Mara Lindqvist.

## What one line looks like

The M1 format plus one field, `hard`, which names the pair's trap:

```json
{"id": "h-001", "person": "Mulugeta Girma", "a": {"text": "Mulugeta takes coffee with three spoons of sugar.", "written_at": "2025-09-06"}, "b": {"text": "Mulugeta drinks coffee without any sugar at all.", "written_at": "2025-09-06"}, "label": "contradiction", "type": null, "hard": "explain_away", "reason": "One says lots of sugar and the other says none, for the same coffee habit."}
```

The model never sees `hard`, `label`, `type` or `reason`. The prompt only uses the person and the two memories.

## Counts

| Label | Type | Pairs |
|---|---|---|
| contradiction | (none) | 150 |
| non_contradiction | update | 50 |
| non_contradiction | compatible_detail | 50 |
| non_contradiction | merely_related | 50 |
| unclear | (none) | 0 |

There are no `unclear` right answers, the same as the M1 test set.

## The traps

| Trap | Label and type | Pairs | Same day | With a change word |
|---|---|---|---|---|
| `explain_away`: you could invent a way both are true ("maybe two dogs"), but reading rule 4 says no | contradiction | 40 | 20 | 0 |
| `paraphrase`: the clash is said in different words | contradiction | 30 | 15 | 1 |
| `small_difference`: one number, day, colour or place differs | contradiction | 30 | 15 | 0 |
| `change_word_contradiction`: a change word appears, but both describe the same moment or the same past change told two ways | contradiction | 25 | 15 | 25 |
| `far_apart_no_change`: written months apart, no change described (reading rule 1) | contradiction | 25 | 0 | 0 |
| `indirect_update`: a change told without the usual change words | update | 25 | 0 | 0 |
| `update_with_overlap`: wording close to a contradiction, plus a clear change | update | 25 | 0 | 23 |
| `looks_like_clash`: units, counts, part and whole, nicknames, general and specific | compatible_detail | 30 | 15 | 2 |
| `near_identical`: almost the same sentence, and both are true | compatible_detail | 20 | 10 | 4 |
| `rule5_bait`: different days, times or places with no exclusive word (reading rule 5) | merely_related | 25 | 13 | 3 |
| `near_identical`: almost the same sentence, but a different fact | merely_related | 25 | 13 | 3 |

"Same day" means both memories were written on the same date. "Change word" means one of: now, moved, switched, since, stopped, later, replaced. Updates are never same-day, because memory b of an update is always written later.

## How it was written

- **People:** 300 different people, one per pair, named in both memories. Names come from many cultures (West, East and Southern Africa, the Arab world, Iran, Turkey, South and East Asia, the Pacific, Latin America, Europe, Central Asia, the Caucasus, Native North America). Each name was checked by a program against every name word in `data/m1/`.
- **Topics:** everyday life only, as in M1: homes, jobs, food, pets, family, hobbies, travel.
- **Batches:** written by Claude Code in batches of 25 to 40, one trap at a time, each batch reread against reading rules 1 to 5 before the next. No helper agents.
- **Text alone decides** (D-038). Each pair was checked twice: once with its dates, once as if the dates were gone. Both readings give the same label. An update always shows its change in its own words, and a contradiction never depends on the dates.
- **Dates match the words.** Where an update says when the change happened ("got married in September"), memory a is dated before that time and memory b after it. Without this, the dated version could read as a contradiction.
- **Hard by design** (D-039). Pairs were written to fit their trap. No model was run on them, and none were picked or dropped by how a model answers.

## The order and the seed

The pairs are in a shuffled order, and no label comes more than 3 times in a row. The order was built one pair at a time with the rule built in (LESSONS.md, 2026-10-21), using Python's `random` with **seed 20261026**. The same seed also drew the dates. Ids `h-001` to `h-300` follow the final order, so an id says nothing about the label or trap.

## Checks

`scripts/check_dataset.py` passes on it:

```
uv run python scripts/check_dataset.py data/m2/test-hard.jsonl --against data/m1/test.jsonl --distinct-from data/m1/train.jsonl --distinct-from data/m1/valid.jsonl
```

It checks every rule from M1 (fields, labels, dates, names, sentence patterns, change-word and same-day shares), and for this file also:

- every pair has a `hard` tag from the list above that fits its label and type;
- no name of any person in the M1 test, training or validation file appears anywhere in it;
- no pair repeats or nearly copies a pair from any of those files.

## Scoring per trap

`judge.score.score_by_trap` scores each trap on its own: precision, recall, `unclear`, accuracy and invalid answers. The command-line scorer prints a per-trap table for this file once it is frozen.
