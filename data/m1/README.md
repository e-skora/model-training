# M1 data: memory contradiction judge

## The files

- `test.jsonl`: the test set. 200 pairs of memories, one pair per line. The model is scored on these and never trained on them.
- `spotcheck-test.md`: 20 pairs from the test set, picked at random with seed 7, for Elias to check by hand.

Each line of `test.jsonl` looks like this:

```json
{"id": "t-001", "person": "...", "a": {"text": "...", "written_at": "2025-11-08"}, "b": {"text": "...", "written_at": "2025-11-08"}, "label": "contradiction", "type": null, "reason": "..."}
```

`label` is `contradiction` or `non_contradiction`. `type` is empty (`null`) for a contradiction, and for a non-contradiction it is `update`, `compatible_detail` or `merely_related`. The meaning of each, and the reading rules, are in `milestones/M1-contradiction-judge/SPEC.md`.

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
