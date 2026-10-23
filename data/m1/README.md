# M1 data: memory contradiction judge

**Final counts (2026-10-22, D-030).** After the second blind review, the 31 `unclear` pairs it still disputed were dropped. The files as used for training and published are: `train.jsonl` 1,522 pairs (697 contradiction, 255 update, 253 compatible_detail, 270 merely_related, 47 unclear), SHA-256 `a41de541d2a25c564009e232b2d75ccb714c5a668dcd30ef73004fae778231b1`; `valid.jsonl` 99 pairs (46, 17, 16, 18, 2), SHA-256 `0476bc57e19460a10f1a3299f17e570b0d7926cb893d6ce5d57d49f356e042d3`; `test.jsonl` unchanged. The sections below tell the story in order, so their counts are the ones from each stage. Details: `reviews/M1-06-blind-review-r2.md`.

## The files

- `test.jsonl`: the test set. 200 pairs of memories, one pair per line. The model is scored on these and never trained on them. Frozen (SHA-256 in `STATE.md`).
- `spotcheck-test.md`: 20 pairs from the test set, picked at random with seed 7, for Elias to check by hand.
- `train.jsonl`: the training set. 1,550 pairs the model learns from (Step 6; 1,500 before the M1-06b fixes below).
- `valid.jsonl`: the validation set. 102 more pairs, used during training only to watch for overfitting (see below). Never trained on.
- `spotcheck-train.md`: 20 pairs from the training set, picked with seed 11 before the M1-06b fixes, so some labels on it are out of date. Not used (D-028: a separate agent reviews the training set, not Elias).
- `train-writing-brief.md`: the exact instructions the writers of the training and validation pairs followed.

Each line of `test.jsonl` looks like this:

```json
{"id": "t-001", "person": "...", "a": {"text": "...", "written_at": "2025-11-08"}, "b": {"text": "...", "written_at": "2025-11-08"}, "label": "contradiction", "type": null, "reason": "..."}
```

`label` is `contradiction`, `non_contradiction` or (in the training and validation sets only) `unclear`. `type` is empty (`null`) for a contradiction and for unclear, and for a non-contradiction it is `update`, `compatible_detail` or `merely_related`. In every file, memory `a` is the earlier one (or written the same day). The meaning of each, and the reading rules, are in `milestones/M1-contradiction-judge/SPEC.md`.

## Label counts

| Label | Type | Pairs |
|---|---|---|
| contradiction | | 100 |
| non_contradiction | update | 34 |
| non_contradiction | compatible_detail | 33 |
| non_contradiction | merely_related | 33 |
| **Total** | | **200** |

There are 200 different people, one per pair, with names from many cultures.

## How it was made

Every person and every memory is made up. Claude Code wrote them by hand on 2026-10-21, in 8 batches of about 25, and reread each batch against the reading rules before writing the next. No text comes from real people or from Aeverum.

Topics are everyday ones only: home, work, food, hobbies, travel, pets, routines, belongings, tastes and places.

The order was shuffled with seed 20261021, so that no label comes more than 3 times in a row.

## Revision after an independent check (2026-10-21)

An independent checker read all 200 pairs. It found no wrong labels, but the set was too easy: a model could score well with shortcuts (surface patterns) instead of reading. For example, every update had a change word ("now", "moved", "switched", "since", "stopped", "later", "replaced") and no other pair did, and many pairs were the same sentence with one word swapped.

Claude Code rewrote 86 pairs in place. Each kept its id, its label and its type, so the label counts and the order did not change. What changed:

- **Change words are in every label now.** 25 new contradictions use a change word but are still contradictions: both memories written the same day ("Kateryna now lives alone" / "Kateryna now shares the flat"), the change word is about something else, or both describe the same past change differently. 19 new compatible details and merely related pairs use a change word with no clash ("Noa switched to oat milk in her morning coffee" / "Noa drinks her afternoon coffee black"). 10 new updates say the change without any of those words ("Ji-woo's new flat is...", "After the renovation...", "As of January...", "These days... instead of...").
- **Dates are spread.** About half of each label and type has both memories on the same day, half on different days. Updates always have memory b later, now with gaps from 6 days to 26 months. No pair has memory b dated before memory a (t-068 was fixed).
- **Fewer templates.** No pair pattern (like "X's favourite thing is A" / "X's favourite thing is B") is used more than 3 times. Compatible details are no longer always "vague fact, then specific fact": some give the detail first, some match a count or a unit ("six goldfish and one catfish" / "seven fish"; "1.85 metres" / "a little over six feet"). Merely related pairs are no longer always "thing 1 / thing 2".
- **Near-identical pairs are not always contradictions.** 39 contradictions still differ by only one or two words, so 8 non-contradictions do too ("Yuki works from home on Fridays" / "Yuki works from home on Mondays").
- **One person per pair.** t-140, t-174 and t-179 (found by the checker), plus t-042 and t-194 (found while rewriting), had a memory about someone else, such as a brother or a teacher. Now both memories in every pair are about the named person.

A second Claude agent then reread all 86 rewritten pairs. It found no wrong labels. It found one unclear pair (t-147) and nine wording problems, and all ten were fixed.

| Label | Type | Same day | Different days | Has a change word |
|---|---|---|---|---|
| contradiction | | 49 | 51 | 26 |
| non_contradiction | update | 0 | 34 | 24 |
| non_contradiction | compatible_detail | 16 | 17 | 8 |
| non_contradiction | merely_related | 16 | 17 | 11 |

`scripts/check_dataset.py` checks the file. It stops if a field is missing, a label is wrong, an id or pair repeats, memory b is dated before memory a, an update's memory b is not later than memory a, or the person's first name is missing from either memory. It also prints how many pairs per label contain a change word.

```
uv run python scripts/check_dataset.py data/m1/test.jsonl
```

## The test set is never used for training

Once Elias has spot-checked it, the test set is frozen: it is never edited again, and it is never used for training. Its fingerprint (SHA-256) goes in `STATE.md`. The training set is written separately, in Step 6, and must not share any pair with this file.

## The training and validation sets (Step 6, 2026-10-22)

`train.jsonl` and `valid.jsonl` use the same format as the test set, with a third label, `unclear` (D-025): the text alone cannot settle the pair, because whether the memories clash depends on a fact neither one gives, such as the scope or the time ("is allergic to cats" / "has a cat"). Silent changes (a current fact stated differently months apart, with no change wording) are never `unclear`; they stay `contradiction` under reading rule 1. Since D-029, `unclear` also needs reading rules 4 and 5: an invented escape ("maybe she has two dogs") does not make a pair unclear, and different days, times or places with no exclusive word are not a contradiction.

| Label | Type | Train | Valid |
|---|---|---|---|
| contradiction | | 697 (45%) | 46 |
| non_contradiction | update | 255 | 17 |
| non_contradiction | compatible_detail | 253 | 16 |
| non_contradiction | merely_related | 270 | 18 |
| unclear | | 75 (5%) | 5 |
| **Total** | | **1,550** | **102** |

These are the counts after the M1-06b fixes (below). The first version had 675, 255, 248, 248 and 74 (1,500) in train and 45, 17, 16, 17 and 5 (100) in valid.

Every pair has its own made-up person: 1,652 different people, none sharing a first name or surname with anyone in the test set.

### How they were made

Claude Code wrote the brief (`train-writing-brief.md`) and split the writing across 8 Claude helper agents on 2026-10-22. Each wrote 200 pairs in 4 batches of 50, with its own range of cultures for names and its own topic focus (food and music; pets, gardens and sport; home; work and study; travel and transport; hobbies and films; clothes and belongings; routines and past events). Each reread every batch against the reading rules and ran the checker after it. Then Claude Code:

- reread every pair the writers marked as unsure (about 50) and changed two: one `unclear` pair got the missing fact made explicit, and one ("has never driven a car" / "drove a golf cart") became `merely_related`;
- searched all 1,600 for contradictions that could look like updates once the dates are hidden (change wording in the newer memory only); all 5 found were real contradictions;
- added a change word to 7 `unclear` pairs, so every label has them;
- gave one repeated person a new surname;
- read a random 40 pairs (seed 5): no wrong labels found.

The 100 validation pairs were drawn from the 1,600 with seed 20261022, by label and type (45, 17, 16, 17, 5), and the rest became the training set. Both files were then shuffled with the same seed and numbered `tr-0001` to `tr-1500` and `v-001` to `v-100`.

Hard cases, from the checker (train): 213 of 675 contradictions contain a change word; 108 of 255 updates (42%) use none; 67 compatible details, 76 merely related pairs and 11 unclear pairs contain one. About half of each label is written on the same day (contradiction 347 of 675, compatible detail 128 of 248, merely related 134 of 248, unclear 36 of 74); updates never are.

### Order and dates are shuffled at training time, not here

The model will meet pairs in any order and sometimes without dates (D-026). The files keep the earlier memory as `a`. `src/judge/format.py` turns pairs into chat examples and, with seed 20261022, shows the later memory first in exactly half and leaves the dates out of exactly a third. So every update in these files can be recognised from its wording alone, and no reason mentions "first", "second" or a date.

### Validation set and overfitting

During training, the model is scored now and then on `valid.jsonl`. If it keeps getting better on the training pairs but worse on the validation pairs, it is overfitting: memorising its practice examples instead of learning the rule. Then training should stop earlier. The validation set is never trained on, and the test set is never looked at during training at all.

### The walls, checked by `scripts/check_dataset.py`

```
uv run python scripts/check_dataset.py data/m1/train.jsonl --against data/m1/test.jsonl --distinct-from data/m1/valid.jsonl
uv run python scripts/check_dataset.py data/m1/valid.jsonl --against data/m1/test.jsonl --distinct-from data/m1/train.jsonl
```

Both print PASS. Besides the earlier checks, the checker now stops if:
- any first name or surname of a test-set person appears anywhere in the file (whole word);
- a pair is a near copy of a test pair, or repeats or nearly copies a pair of the other file (in either order). Near copy means more than 60% shared words: the words of both memories, without the person's name, shared divided by all the different words of the two pairs together;
- Mara or Lindqvist appears;
- a sentence pattern is used more than 3 times in any block of 100 pairs (a memory shares at least 60% of its words with memories in 3 or more other pairs of its block);
- a label or type with at least 20 pairs has change words in under 10% of them, fewer than 10% of updates are said without one, or fewer than 35% or more than 65% of a label or type (updates aside) is same-day;
- among `unclear` reasons, the same start (the first 3 words, with the person's names counted as one word) is used more than 3 times in any 100 (added in M1-06b).

### Fix of the `unclear` pairs (M1-06b, 2026-10-22)

A blind review by a separate agent (`reviews/M1-06-blind-review.md`) found the contradiction and non-contradiction pairs clean, but about half the `unclear` pairs could be decided, and 73 of 74 `unclear` reasons started the same way. Reading rules 4 and 5 were added (D-029), and Claude Code then:

- reread all 79 `unclear` pairs: 28 stay `unclear`, 25 became `contradiction`, 5 `compatible_detail` and 21 `merely_related`;
- applied rule 5 to every pair of both files: 2 more contradictions (tr-0068, v-054) became `merely_related`;
- wrote 52 new `unclear` pairs (tr-1501 to tr-1550, v-101, v-102) where a truly missing fact decides it and both readings are natural, such as "is vegan" / "takes honey in her tea";
- rewrote every `unclear` reason so no two start the same way, and added that rule to the checker.

The files grew to 1,550 and 102 pairs instead of losing 50 good pairs, and the label mix stayed at 45 / 50 / 5. Every change, pair by pair: `reviews/M1-06-fixes.md`.

SHA-256 after the fixes: `train.jsonl` `22d2ff6f9f563b8712b96bf8ea6dfd2e2f01826a10ce8a367bf3ccab291bdacd`, `valid.jsonl` `5f61a3f36feed7ccb95a6b245a1e41443b2d8dd8218fcd4f5b67245ada31fc19` (not frozen; Claude in Cowork re-checks the `unclear` pairs first).

SHA-256 of the first version, as written: `train.jsonl` `a910ed814096c6f7ffbd3803cc93049977391ee48d72102e6e0fcfd71e65619d`, `valid.jsonl` `ef439b8d3e6b7ccf40055e632c6e91d1619868ae7f76ea87adc2a48fd57a2393` (not frozen; the review may still change pairs).
