# M2 hard test set

`test-hard.jsonl` is the test set for M2. It is 300 pairs of memories about made-up people. Each pair is built around a trap: a way a judge could get it wrong by reading the surface instead of the meaning.

It is used only to score models. **It is never used for training**, and no model has been run on it yet.

**Status:** frozen 2026-10-26 after a blind review (`reviews/M2-01-blind-review.md`); fingerprint in `STATE.md`.

**Training and validation files, final (2026-10-27, D-045):** `train-hard.jsonl` 971 pairs (1 `unclear`), `valid-hard.jsonl` 97 pairs (0 `unclear`). The 32 `unclear` pairs the blind review disputed were moved to `dropped-unclear.jsonl`. The counts in the sections below are as first written. Details: `reviews/M2-03-blind-review.md`.

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

# M2 hard training and validation pairs (Step 3, part A)

`train-hard.jsonl` (1,000 pairs) and `valid-hard.jsonl` (100 pairs) are new practice pairs for reinforcement learning, built from the same trap list as the test set. **Neither file is ever used to score the final model.** The validation file only helps choose a checkpoint during training; the final score comes from the frozen `test-hard.jsonl` and the M1 test set.

**Status:** written and checked by Claude Code on 2026-10-27. Waiting for a blind review by Claude in Cowork.

SHA-256 as written:
- `train-hard.jsonl`: `e3069a5f1718d16e34499181e086dedf7093fa83b16b138340798b9c025814aa`
- `valid-hard.jsonl`: `a93ab3c6169b50146540cd25824ee86829c9bc4bd94c1e7c098c9fa8b152261a`

## Format

The test-set format plus two fields that say where each row came from (D-043): `"source": "synthetic"` (made up) and `"label_by": "claude-code"` (who set the label). Ids are `tr2-0001` to `tr2-1000` and `va2-001` to `va2-100`. Everything is made up, about fictional people. Every person appears in one pair only, and no name word is shared between the two files or with any M1 file or the hard test set.

## Counts

| Label | Type | Trap | Train | Valid |
|---|---|---|---|---|
| contradiction | | `explain_away` | 125 | 13 |
| contradiction | | `paraphrase` | 95 | 9 |
| contradiction | | `small_difference` | 95 | 9 |
| contradiction | | `change_word_contradiction` | 77 | 8 |
| contradiction | | `far_apart_no_change` | 78 | 8 |
| non_contradiction | update | `indirect_update` | 80 | 8 |
| non_contradiction | update | `update_with_overlap` | 80 | 8 |
| non_contradiction | compatible_detail | `looks_like_clash` | 100 | 10 |
| non_contradiction | compatible_detail | `near_identical` | 70 | 7 |
| non_contradiction | merely_related | `rule5_bait` | 85 | 9 |
| non_contradiction | merely_related | `near_identical` | 85 | 8 |
| unclear | | `true_unclear` | 30 | 3 |
| **All** | | | **1,000** | **100** |

Per label in train: contradiction 470, update 160, compatible_detail 170, merely_related 170, unclear 30. In valid: 47, 16, 17, 17, 3. The trap counts were fixed in prompt M2-03 before any pair was written. No model answered these pairs, and the baseline's per-trap results did not steer the mix.

Same day and change words (now, moved, switched, since, stopped, later, replaced), in train: contradiction 226 and 99 of 470; update 0 and 51 of 160; compatible_detail 90 and 41 of 170; merely_related 89 and 43 of 170; unclear 15 and 4 of 30. The checker prints the full table for both files.

## How it was written

- **One brief:** `train-writing-brief.md` (format, reading rules 1 to 5, D-044, the trap list with a made-up example each, the date, name and `unclear` rules, and what each writer hands back). The writers never opened the hard test set, the M1 data, or the results.
- **11 helper agents** (Claude, w01 to w11), each with its own cultures for names and its own topic focus, wrote 103 pairs each: 97 decidable pairs (their share of each trap) plus 6 `unclear` candidates. Each wrote in 4 batches of about 26 and ran the checker after every batch.
- **Claude Code reread every batch** against reading rules 1 to 5 before accepting it (log below), then compared all 1,133 pairs with each other to find near copies across writers.
- **`unclear` pairs** went through a blind second judge (below). One more helper (u2) wrote 60 new `unclear` candidates after the first round kept too few.
- **Validation draw:** the 100 validation pairs were drawn per trap from the accepted pairs with Python's `random`, **seed 20261027**. The rest became the training set.
- **Order:** each file was then ordered one pair at a time with the rule built in that no label or type comes more than 3 times in a row (same seed), and renumbered. So an id says nothing about the label or trap.

## Reread log

59 pairs were fixed or replaced while rereading; 57 of them are in the final files (the other 2 were `unclear` candidates later dropped by the blind judge). No fix changed a decidable label: every decidable label the writers set was kept.

| Why | Pairs |
|---|---|
| A flaw in the pair: not truly `unclear` (1), a forced change word (1), a confusing side clash (1), a pronoun slip (1), a slot not clearly single (1), too close to an idea that `LESSONS.md` quotes from the hard test set (1) | 6 |
| The same idea as pairs by other writers (for example the 8th "vegetarian orders meat" pair, the 7th "never owned a car, drives one"), rewritten for variety | 29 |
| Near copy of another writer's pair, found by comparing all pairs (one was word for word the same) | 22 |
| Almost the same sentences as another pair but a different label (`unclear` against `contradiction`), rewritten so the data does not teach noise | 1 |
| A surname shared by two people, renamed | 1 |

Main finding: the writers' labels were right, but separate writers kept reaching for the same everyday ideas. This table summarises the working log (one line per fix), which stayed in the session's scratch folder and is not in git.

## The `unclear` double-check (D-030)

Every `unclear` pair was judged blind by a separate helper agent, mixed with ordinary pairs so the judge could not guess which were which. It read only the reading rules and the pairs, never the labels. Only pairs it also called `unclear` were kept.

| Round | `unclear` candidates | Judge also said `unclear` | Ordinary pairs mixed in | Judge agreed on the label | Judge agreed on label and type |
|---|---|---|---|---|---|
| 1 (writers w01 to w11) | 66 | 5 | 66 | 66 | 65 |
| 2 (writer u2, a new judge) | 60 | 51 | 40 | 40 | 38 |

- **Round 1:** the judge found a more natural reading for 61 of 66 candidates: 32 as `compatible_detail`, 27 as `contradiction`, 2 as `update`. Most were "whose is it?" or "a visit or a move?" pairs, where the wording hints at an answer. All 61 were dropped, not relabelled.
- **Round 2:** u2 wrote pairs where one memory reports a single act whose deciding detail is left out, and both values are ordinary (was it paid? which language? was it live or on a screen? homemade or bought?). The judge kept 51 of 60.
- **Choosing 33 of the 56 agreed pairs:** with seed 20261027, at most 3 per kind of missing fact, across 13 kinds, so no single kind becomes a shortcut. The 3 validation `unclear` pairs were drawn from these 33.
- **Type disagreements on ordinary pairs (3, kept as written):** w09-024 (work clothes and gym clothes laid out the night before), w01-037 (grandmother taught pickles, and dumplings) and w03-043 (canteen lunch, and bench lunch on fine days). The judge said `compatible_detail`; they are two separate facts, so `merely_related` by the brief's rule, as for the 4 `near_identical` disagreements in the test-set review.

## Checks

`scripts/check_dataset.py` passes on both files:

```
uv run python scripts/check_dataset.py data/m2/train-hard.jsonl --against data/m2/test-hard.jsonl --against data/m1/test.jsonl --distinct-from data/m1/train.jsonl --distinct-from data/m1/valid.jsonl --distinct-from data/m2/valid-hard.jsonl
uv run python scripts/check_dataset.py data/m2/valid-hard.jsonl --against data/m2/test-hard.jsonl --against data/m1/test.jsonl --distinct-from data/m1/train.jsonl --distinct-from data/m1/valid.jsonl --distinct-from data/m2/train-hard.jsonl
```

New in the checker for these files: `true_unclear` is a valid trap only on `unclear`; `source` and `label_by` must be on every line, with the allowed values; and `--against` takes several test sets, so the name wall and the near-copy check also cover the hard test set.
