# M1 data: memory contradiction judge

## The files

- `test.jsonl`: the test set. 200 pairs of memories, one pair per line. The model is scored on these and never trained on them.
- `spotcheck-test.md`: 20 pairs from the test set, picked at random, for Elias to check by hand.

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

Most pairs are made to look alike, so the judge has to read carefully:

- Contradictions where only one small thing differs: a day, a colour, a number or a place. 40 have both memories written on the same day, and 60 on different days with no change described. Under reading rule 1, those 60 are still contradictions.
- Some contradictions use words that sound like a change but are not one, such as two "now" memories written on the same day, or a later memory that says "still" or "has always".
- All 34 updates describe a change ("moved", "now", "switched", "later", "since then") and have the later date on memory b.
- Compatible details and merely related pairs share many words with each other.

The order was shuffled with seed 20261021, so that no label comes more than 3 times in a row.

`scripts/check_dataset.py` checks the file:

```
uv run python scripts/check_dataset.py data/m1/test.jsonl
```

## The test set is never used for training

Once Elias has spot-checked it, the test set is frozen: it is never edited again, and it is never used for training. Its fingerprint (SHA-256) goes in `STATE.md`. The training set is written separately, in Step 6, and must not share any pair with this file.
