# M3 hard test set 2 and hard validation set 2

`test-hard-2.jsonl` (600 pairs) is the test set for M3, the second attempt at beating v1 with reinforcement learning. `valid-hard-2.jsonl` (200 pairs) is its validation set: it only helps choose a checkpoint during training. Each pair is two memories about one made-up person, built around a trap: a way a judge could get it wrong by reading the surface instead of the meaning. **Neither file is ever used for training**, and no model has been run on either.

**Status:** written and checked by Claude Code on 2026-11-11 (prompt M3-01). Blind-reviewed by Claude in Cowork on 2026-11-13 with control pairs (`reviews/M3-01-blind-review.md`): 593 of 600 test labels agreed, all 60 validation and 40 control pairs agreed; 3 test pairs had a time word fixed (h2-136, h2-522, h2-559). **Both files frozen 2026-11-13.**

SHA-256, frozen 2026-11-13:
- `test-hard-2.jsonl`: `4bf37d6af7dd8cca037862b06425410a381f6075a4403de4c55361729f03a71d` (as first written: `a57843b6...720d3b63`)
- `valid-hard-2.jsonl`: `f819f98758fa70f59759ec15c20f2049366b05f886e2dc77ff3a72aa84cc18f6` (unchanged by the review)
- `situation-plan.jsonl`: `1164c03a843b912687dee0937237585b9d555add3a28fa1e114c621573536de3`

## Why these sizes

The M3 bar asks for 0.03 more recall than v1. With 300 contradictions that is 9 pairs, about twice the 4-pair wobble between two forms of the same model (lesson A2). M2's test set had 150 contradictions, so its bar was 5 pairs. The validation set has 100 contradictions, so one false alarm moves precision by about 0.01 instead of 0.02 (lesson A5).

## Made up, all of it

Every person and every memory was made up. There is no real personal data. No name word of any person in `data/m1/` or `data/m2/` appears anywhere in either file. Each of the 800 people appears in one pair only, and all 1,602 name words are different. Aeverum's data and the name Mara Lindqvist are never used.

## Format

The M2 format: `id`, `person`, `a` and `b` (each `text` and `written_at`, YYYY-MM-DD), `label`, `type`, `hard` (the trap), `reason`, `source` (always `synthetic`) and `label_by` (always `claude-code`). Ids are `h2-001` to `h2-600` and `v2-001` to `v2-200`. The model never sees `hard`, `label`, `type` or `reason`.

## Counts

| Label | Type | Test | Valid |
|---|---|---|---|
| contradiction | (none) | 300 | 100 |
| non_contradiction | update | 100 | 33 |
| non_contradiction | compatible_detail | 100 | 33 |
| non_contradiction | merely_related | 100 | 34 |
| unclear | (none) | 0 | 0 |

## The traps

The same 11 traps as `data/m2/test-hard.jsonl`. The test set has twice M2's count of each, with the same shares of same-day pairs and change words. The validation set has the same mix at a third of the test set's size. "Same day" means both memories carry the same date. "Change word" means one of: now, moved, switched, since, stopped, later, replaced.

| Trap | Label and type | Test pairs | Same day | Change word | Valid pairs | Same day | Change word |
|---|---|---|---|---|---|---|---|
| `explain_away` | contradiction | 80 | 40 | 0 | 27 | 13 | 0 |
| `paraphrase` | contradiction | 60 | 30 | 2 | 20 | 10 | 1 |
| `small_difference` | contradiction | 60 | 30 | 0 | 20 | 10 | 0 |
| `change_word_contradiction` | contradiction | 50 | 30 | 50 | 17 | 10 | 17 |
| `far_apart_no_change` | contradiction | 50 | 0 | 0 | 16 | 0 | 0 |
| `indirect_update` | update | 50 | 0 | 0 | 17 | 0 | 0 |
| `update_with_overlap` | update | 50 | 0 | 46 | 16 | 0 | 15 |
| `looks_like_clash` | compatible_detail | 60 | 30 | 4 | 20 | 10 | 2 |
| `near_identical` | compatible_detail | 40 | 20 | 8 | 13 | 6 | 3 |
| `rule5_bait` | merely_related | 50 | 26 | 6 | 17 | 9 | 2 |
| `near_identical` | merely_related | 50 | 26 | 6 | 17 | 9 | 2 |

Updates are never same-day, because the newer memory of an update is always written later. Each trap is defined, with an example, in `writing-brief.md`.

## How it was written

1. **Slots by program.** `build/skeleton.py` (Python `random`, seed 20261111) made 800 slots. Each slot fixes its set, trap, label and type, whether it is same-day, whether it has a change word, and both dates. Dates run from 2023-01-01 to 2026-09-30. Gaps follow M2's design, so the two sets can be compared: other contradictions are 3 days to 8 weeks apart, far-apart pairs 4 to 18 months, updates 1 week to about 30 months, and the other non-contradictions 3 days to 3 months. Each of 8 writers got 100 slots, with every trap in each writer's share.
2. **A situation plan first (lesson C1).** Eight helper agents (Claude, w1 to w8), each with its own topic area, gave every slot a topic and a one-phrase clash or change. The areas: home and housing; food and drink; work and study; family, friends and pets; sport and the outdoors; arts, music and crafts; travel and places; belongings, routines and tastes. Two writers cannot reach for the same idea when their areas differ. The plan was checked against itself and against every pair in the non-frozen files of `data/m1/` and `data/m2/` with the cross-pair measure applied to each line's topic and clash (`build/check_plan.py`). It flagged 138 plan lines, and all were read by hand. 104 were the same idea as an old pair or another plan line, such as left- or right-handed, a house number 14 or 41, or a suitcase weighed in kilos and pounds. Each went back to its writer for a new idea before any pair was written. The other 34 only share a topic. The two frozen test files were not opened for this; the checker's walls cover them on the written pairs (below). The final plan is `situation-plan.jsonl`, 800 lines, each with its set, trap, label, type, person, culture, topic, clash, dates and writer.
3. **Names.** Each writer suggested 300 names from its own cultures (West, East and Southern Africa, the Arab world, Iran, Turkey and Central Asia, South Asia, East and Southeast Asia, Europe, the Caucasus, the Americas and the Pacific). Each name word was screened with the checker's own name wall against every file in `data/m1/` and `data/m2/` (`build/screen_names.py`, `build/build_names.py`). About 60% of the first suggestions were already taken. Free first names and surnames from the same culture were paired, and 100 names from each writer's region were drawn and spread over all 800 slots at random (seed 20261111). So a person's culture says nothing about the topic or the trap.
4. **Writing.** Each writer wrote its 100 pairs from its slice of the plan, in two batches of 50, with `writing-brief.md` (reading rules 1 to 5, the traps, the date rules) and a validator (`build/validate.py`) that checks the slot was followed exactly, including the change-word rule. No writer saw another writer's slice, any model's answers, or any data file. Claude Code reread every pair of every batch against reading rules 1 to 5, with the dates and the order in mind, before the writer wrote the next batch. That also checked that dates fit the time words (lesson C4): for example, "switched in March" has both memories in the same March, and "gave up glasses last year" is dated two years apart. The rereads sent back 7 pairs. 5 repeated ideas from earlier sets or another writer ("can't swim but works as a lifeguard", "never threaded a needle but mends by hand", "never been on a plane"). 1 contradiction could read as an update without dates. 1 had two sellers who could be one person.
5. **The cross-pair check, then rewrites.** The new check (below) compared all 800 pairs with each other and with every earlier file. The first run flagged 140 pairs. Writers rewrote them with a new idea or different specifics, and Claude Code reread every rewrite. A second run flagged 22, a third 4. Claude Code rewrote the last 4 itself, and the final run found none. In all, 154 of the 800 pairs were rewritten after their first draft, so about that many no longer follow the topic or clash their plan line gives. The plan stays as it was committed, as a record of the starting point.
6. **Two renames.** Two people's name words clashed with ordinary words in other pairs. A surname "Two Bulls" blocked the word "Two", and a first name "Danish" blocked the language. They became "Bettelyoun" and "Faraz", both screened like every other name.
7. **Order and ids.** Each file was ordered one pair at a time with the rule built in that no label comes more than 3 times in a row, the same rule as M2 (LESSONS 2026-10-21), using Python's `random` with **seed 20261111** (`build/assemble.py`). The longest run of one label, and of one type, is 3 in both files. Ids follow the final order, so an id says nothing about the label or trap.

No model was run on any pair, and no pair was picked or dropped by how a model answers (D-039).

## The cross-pair check (lesson C1)

M2's writers kept reaching for the same everyday ideas, and the old checker could not see it, because it only caught near word-for-word copies. The new check, `scripts/check_dataset.py --cross-pair`, compares every pair with every other pair in the file, and with every pair in every file given with `--against` or `--distinct-from`. It fails if any two pairs score above the limit, **0.44**. Two old files given to it are never compared with each other.

**The measure.** For each pair, take the words of both memories. Leave out the person's names and 107 little words ("the", "has", "never" and so on), and cut plain endings ("drives" and "driving" both become "driv"). Some words also add a group word, so the same idea said with different words still matches: "beef", "pork" and "lamb" all add MEAT, "brother" and "twin" add SIBLING, "hatchback" and "van" add VEHICLE (14 small groups in `IDEA_GROUPS`). Each word gets a weight: the fewer pairs use it, the more it weighs (TF-IDF, counted over all files given). The score is the cosine between the two pairs' weight lists: 0 means no shared words, 1 means the same words. Pairs in a frozen test file (`--against`) are named by id only, never by their text.

**How the limit was chosen** (`scripts/cross_pair_calibration.py` reruns it). The check was tried on M2's 1,068 training and validation pairs. LESSONS 2026-10-27 names the repeated ideas found there by hand. Searching the final files found four of them still present: "vegetarian eats meat" (5 pairs), "only child, but has a brother or sister" (5), "never owned a car, but drives one" (3), "bakery night shift, but works days" (2).

| Limit | Links flagged | Pairs flagged | Hand-found ideas caught |
|---|---|---|---|
| 0.40 | 219 | 261 of 1,068 | 4 of 4 (11 of 15 pairs) |
| **0.44** | **146** | **193 of 1,068** | **4 of 4 (8 of 15 pairs)** |
| 0.48 | 95 | 142 of 1,068 | 3 of 4 (the car idea is missed) |

0.44 is the highest limit that still catches all four hand-found ideas (the car idea's best link scores 0.45). To see how many of the other flags are real, 40 of the 146 links were drawn at random (seed 20261111) and read: **33 are the same idea** (sibling counts, left- or right-handed, chess on two weeknights, kilos against pounds, a front door blue or green, and so on), and 7 share a topic but not an idea. So most of what it flags at 0.44 are repeats that nobody had found by hand.

**What it does not catch.** It reads words, not meaning. The same idea told in entirely different words still passes; only 1 of the 10 possible links inside "vegetarian eats meat" scores above 0.44. That is why the situation plan comes first: it stops repeats before any writing, and this check catches what slips through.

**Tests** (`tests/test_check_dataset.py`, made-up pairs in a temporary folder): two different ideas pass; the same idea in other words fails, in one file and across a `--distinct-from` file; a frozen file's pair is named by id with none of its text; two old files are not compared with each other; without `--cross-pair` the old behaviour is unchanged; the word handling and the group words work as described, and the group words are what lift the made-up "vegetarian" pairs above the limit. With the limit raised to 0.99, or the scoring switched off, the tests fail.

## Checks

`scripts/check_dataset.py` prints PASS on both files:

```
uv run python scripts/check_dataset.py data/m3/test-hard-2.jsonl --against data/m2/test-hard.jsonl --against data/m1/test.jsonl --distinct-from data/m1/train.jsonl --distinct-from data/m1/valid.jsonl --distinct-from data/m2/train-hard.jsonl --distinct-from data/m2/valid-hard.jsonl --distinct-from data/m2/rl-pool.jsonl --distinct-from data/m2/rl-pool-always-wrong.jsonl --distinct-from data/m2/dropped-unclear.jsonl --distinct-from data/m3/valid-hard-2.jsonl --cross-pair
uv run python scripts/check_dataset.py data/m3/valid-hard-2.jsonl --against data/m2/test-hard.jsonl --against data/m1/test.jsonl --distinct-from data/m1/train.jsonl --distinct-from data/m1/valid.jsonl --distinct-from data/m2/train-hard.jsonl --distinct-from data/m2/valid-hard.jsonl --distinct-from data/m2/rl-pool.jsonl --distinct-from data/m2/rl-pool-always-wrong.jsonl --distinct-from data/m2/dropped-unclear.jsonl --distinct-from data/m3/test-hard-2.jsonl --cross-pair
```

That covers the fields, labels, types, traps, dates (memory b never earlier, updates always later), names in both memories, the same-day and change-word shares, sentence patterns, `source` and `label_by`. It also checks the name walls against every file in `data/m1/` and `data/m2/` and the other new file, no repeats or near copies of any of their pairs, and the cross-pair check over all 4,344 pairs in those files: no two score above 0.44. The frozen test files were read only by the checker's walls, never opened or fingerprinted.

## Files

- `test-hard-2.jsonl`, `valid-hard-2.jsonl`: the two sets.
- `situation-plan.jsonl`: the plan, committed before any pair was written.
- `writing-brief.md`: what every writer read.
- `build/`: the scripts that drew the slots, screened names, checked the plan, validated each writer's file, ran the walls and ordered the files. They ran in the session's scratch folder, where the writers' working files stay (not in git).
