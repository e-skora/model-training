# M3 Step 1: blind review of hard test set 2 and hard validation set 2 (Claude in Cowork, 2026-11-13)

**Files:** `data/m3/test-hard-2.jsonl` (600 pairs, SHA-256 as written `a57843b6...720d3b63`) and `data/m3/valid-hard-2.jsonl` (200 pairs, `f819f987...cc18f6`), branch `m3-01-test-set-2` at `90cdf8d`. Lessons applied: C3 (blind review with control pairs), C4 (dates fit the time words).

## How it was checked
- **Blind items:** all 600 test pairs, 60 validation pairs drawn at random, and 40 **control pairs** from `data/m2/train-hard.jsonl` whose labels earlier blind reviews had agreed with (seed 20261113). 700 in all, shuffled together and given new ids, so no reviewer could tell which set a pair came from.
- **Reviewers:** 7 separate Claude agents (Sonnet), 100 items each, with `reviews/blind-brief.md`. They saw only the person and the two memories, never a label, type, trap or reason, and never the writing brief.
- **Compared** with the files' labels by Claude in Cowork.

## Result
| Items | Pairs | Same label | Same type (where the label matched) |
|---|---|---|---|
| Test set 2 | 600 | 593 (98.8%) | all |
| Validation sample | 60 | 60 | all |
| Controls | 40 | 40 | all |

The controls all matched, so the reviewers were reading carefully and not just agreeing.

## The 7 test disagreements
**4 kept as written** (both memories can be true, reading rule 3, the same pattern M2's review kept):

| Pair | Memories | File | Reviewer |
|---|---|---|---|
| h2-099 | quilt with stars on her bed / on her sofa | merely_related | contradiction (not sure) |
| h2-161 | dried mulberries / dried apricots in his coat pocket | merely_related | contradiction |
| h2-190 | referees in the women's rugby league / plays fly-half in it | merely_related | contradiction (not sure) |
| h2-555 | a thermos of tea / a thermos of porridge to the farm | merely_related | contradiction |

**3 fixed: the time word let the event fall after the first memory** (lesson C4). Each is a contradiction ("never had X") against an event dated only by a time word that, with the dates given, could fall after the first memory, which reads as an update. The reviewer called all three updates. The event now clearly comes before the first memory, so the pair is a contradiction with or without dates; the label, type, trap and dates did not change.

| Pair | Memory a (date) | Memory b before | Memory b now |
|---|---|---|---|
| h2-522 | has no godchildren (2023-04-04) | "At the naming party, ..." | "At a naming party two years ago, ..." |
| h2-136 | brother has never had a passport (2025-03-10) | "Last month ... flew abroad" (written 2025-04-14) | "In January ... flew abroad" |
| h2-559 | daughter never spent a night away (2024-11-19) | "Last weekend ... slept over" (written 2024-11-25) | "In September ... slept over" |

## With the dates hidden
Apart from the fixed pairs and one control, the reviewers changed 14 labels when imagining the dates hidden and the order swapped (12 test, 2 validation). All 14 are updates whose text shows the change in its own words ("replaced", "quit", "no longer", "now ... instead"), so the text alone decides them. They stay as written.

## Lesson
The M2 date check (C4) reads time words such as "in March" or "since January". It does not catch a relative time word in memory b ("last month", "last weekend", or none at all) that, counted back from b's date, can land after memory a's date. Claude Code's reread missed all three. **Fix for attempt 2's training pairs (Step 3):** for every contradiction built on "never X" against an event, the event must be dated before memory a, either by an absolute time word or by b's date; the checker flags "last week", "last weekend", "last month" and "recently" in memory b of a contradiction when b minus that span is after a.

## Decision
Both files frozen after the 3 fixes and a clean checker run. SHA-256 in `STATE.md` (Frozen files).
