# M1-06b: fixes after the blind review of the training set (2026-10-22)

**Who:** Claude Code, branch `m1-06-training-set`, from `prompts/M1-06b-fix-unclear.md`.
**Input:** `reviews/M1-06-blind-review.md` and reading rules 4 and 5 in the SPEC (D-029).
**No GPU runs.** The test set was not touched (SHA-256 still `6e2a0832...f6e574b`).

## Result in short
- Every `unclear` pair was reread against rules 4 and 5: 79 in all (74 train, 5 valid). **28 stay `unclear`; 51 were relabelled.**
- Rule 5 was applied to the whole of both files: **2 more pairs** went from `contradiction` to `merely_related`.
- **52 new `unclear` pairs** were written (50 train, 2 valid), so `unclear` is back to about 5%.
- Every `unclear` reason now starts differently. The checker has a new rule for this, and both files pass.

## 1. The `unclear` pairs, one by one

Test used for each pair: keep `unclear` only if a fact that is truly missing decides it, and neither reading is more natural. An escape someone invents ("maybe two dogs") does not count (rule 4). Different days, times or places with no exclusive word are not a contradiction (rule 5).

### Train: 74 pairs, 25 stay `unclear`, 49 relabelled

| Group | Pairs | Now |
|---|---|---|
| Review said contradiction | tr-0472, tr-0279, tr-1110, tr-0371, tr-0182, tr-0871, tr-1441, tr-0953, tr-0430, tr-0948, tr-0122, tr-0408, tr-0363, tr-0714, tr-0808, tr-1353, tr-1285, tr-0607, tr-0324, tr-1309, tr-0663 (21) | `contradiction` |
| Review said contradiction, Claude Code disagrees | tr-1019 | `merely_related` (see below) |
| Review said compatible or merely related | tr-0002, tr-1060, tr-0169, tr-0236, tr-0501 (5) | `compatible_detail` |
| | tr-0071, tr-0102, tr-1434, tr-0698, tr-0247, tr-1014, tr-0108 (7) | `merely_related` |
| Review agreed `unclear`, but rule 4 decides it | tr-0420 (no second camera is given), tr-0957 (an hour's drive is not near the sea) | `contradiction` |
| Review agreed `unclear`, but rule 5 decides it (different times or places, no exclusive word) | tr-0262, tr-0513, tr-0731, tr-1034, tr-1067, tr-1140, tr-1273, tr-1323 (8) | `merely_related` |
| Review agreed `unclear`, but the activity itself shows the natural reading | tr-0551 (mows someone's grass), tr-1138 (streams over mobile data), tr-1205 ("meets for coffee"), tr-1212 (a forklift is not a car), tr-1388 (new brushes after giving the old ones away) | `merely_related` |
| Stay `unclear` | tr-0032, tr-0067, tr-0096, tr-0149, tr-0171, tr-0203, tr-0219, tr-0228, tr-0309, tr-0429, tr-0587, tr-0742, tr-0874, tr-0875, tr-0914, tr-0926, tr-0989, tr-1093, tr-1213, tr-1216, tr-1369, tr-1410, tr-1450, tr-1453, tr-1471 (25) | `unclear`, reason rewritten |

Train relabels: **23 to `contradiction`, 5 to `compatible_detail`, 21 to `merely_related`.**

**Where Claude Code disagrees with the reviewer: tr-1019.** "Ye-jin has the whole of August off from the publishing house" and "Ye-jin is running a summer writing workshop for teenagers in August." The time off is from one employer, not from all work. Running a separate workshop in that free month is the natural reading, not an invented escape, so this is `merely_related`, not a contradiction.

**Close call kept with the reviewer: tr-0247** (morning shift at a hotel front desk / school drop-off at eight). A hotel morning shift often starts before eight, so this could be `unclear`. It follows the reviewer (`merely_related`) because neither memory gives the shift's hours or uses an exclusive word, which is rule 5's case.

### Valid: 5 pairs, 3 stay `unclear`, 2 relabelled
- v-026 (glasses "only for reading" / puts them on to drive at night): `contradiction` under rule 4 (no invented second pair).
- v-058 (piano lessons at home in the afternoons / at the market stall from noon to six on weekdays): `contradiction`. The stall covers every weekday afternoon, and "only at weekends" is an invented escape.
- v-013, v-037, v-052 stay `unclear`, with new reasons.

## 2. Rule 5 across the whole file

Claude Code read every pair that could be a rule-5 case:
- every contradiction where both memories name a day, time or season (about 100 pairs);
- every contradiction with no exclusive word in either memory (201 pairs);
- every compatible or merely related pair that has an exclusive word ("only", "always", "never", "every day", "once a week"; 34 pairs).

Two contradictions were the "works from home on Fridays / on Mondays" pattern with no exclusive word. Both are now `merely_related`:
- tr-0068: runs 10 km every Tuesday and Thursday / does his 10 km runs on Mondays and Wednesdays.
- v-054: bakes sourdough every Saturday / bakes his sourdough on Sunday afternoons.

The rest stay as they were. They either have an exclusive word ("only", "one weekly class", "never") or they give two values for one fixed thing, such as the time a club meets or the place a group meets. Those are contradictions under rules 1 and 4.

## 3. New `unclear` pairs

52 new pairs: tr-1501 to tr-1550 and v-101, v-102. Each meets the review's standard: one fact that is truly missing decides it, and both readings are natural. The kinds:
- **Does X count as Y?** Honey for a vegan, a WhatsApp group as social media, a ride-hailing car as a taxi, chapati as bread, vaping as smoking, a stepdaughter as a child, a polytechnic diploma as a degree.
- **Which one was it?** Lamb or pork sausage at a barbecue with both on the grill; a paper or an e-book from the library; the ferry or a plane to Zanzibar.
- **How strong is it?** A fear of dogs and an old pug, a cat allergy and a Siberian cat, "hates spicy food" and a mild curry.
- **A vague word.** "Close" and a 25-minute walk; "short" and 1.65 metres; "early" and half past seven.
- **Before or now?** A favourite swimming spot before a knee operation, a favourite dish before a new diet.

They follow every wall in the checker: no test-set name, no near copies of test or train/valid pairs, no sentence pattern more than 3 times in any block of 100. Every new person has a first name and surname used nowhere else in the three files. Of the 50 new train pairs, 8 contain a change word and 27 are same-day, so `unclear` keeps its hard-case shares.

**Totals went up instead of pairs being removed.** Keeping exactly 1,500 and 100 would have meant deleting 50 good pairs that were already written and reviewed. Adding the new pairs keeps the label mix at 45 / 50 / 5 anyway.

## 4. The reasons
Before: 73 of 74 `unclear` reasons started "It depends on whether..." or "Whether they clash depends on...". Now no two of the 75 train `unclear` reasons start with the same three words (names counted as one word), and the same holds for the 5 in valid. The 51 relabelled pairs got new reasons that fit their new label.

New checker rule (`scripts/check_dataset.py`, `check_reasons`): among `unclear` reasons in file order, no start of 3 words (the person's names count as "NAME") may be used more than 3 times in any 100. New test: `test_unclear_reasons_with_one_start_four_times_fail` (4 fail, 3 pass).

## 5. New label counts

| Label | Type | Train before | Train now | Valid before | Valid now |
|---|---|---|---|---|---|
| contradiction | | 675 (45%) | 697 (45%) | 45 | 46 (45%) |
| non_contradiction | update | 255 | 255 | 17 | 17 |
| non_contradiction | compatible_detail | 248 | 253 | 16 | 16 |
| non_contradiction | merely_related | 248 | 270 | 17 | 18 |
| unclear | | 74 (5%) | 75 (5%) | 5 | 5 (5%) |
| **Total** | | **1,500** | **1,550** | **100** | **102** |

## 6. What the checker printed (both PASS)

```
File: data/m1/train.jsonl
Pairs: 1550
  contradiction: 697 (45%)
  non_contradiction: 778 (50%)
    update: 255
    compatible_detail: 253
    merely_related: 270
  unclear: 75 (5%)
Different people: 1550
Pairs with a change word (now, moved, switched, since, stopped, later, replaced) / same day:
  contradiction: 213 of 697 / 360 of 697
  update: 147 of 255 / 0 of 255
  compatible_detail: 67 of 253 / 131 of 253
  merely_related: 81 of 270 / 143 of 270
  unclear: 14 of 75 / 38 of 75
Sentence patterns: none used more than 3 times in any block of 100 pairs
Unclear reasons: no start of 3 words used more than 3 times in any 100
Wall: no names of the 398 name words in data/m1/test.jsonl, no near copies of its pairs
Wall: no repeated or near-copied pairs from data/m1/valid.jsonl
PASS

File: data/m1/valid.jsonl
Pairs: 102
  contradiction: 46 (45%)
  non_contradiction: 51 (50%)
    update: 17
    compatible_detail: 16
    merely_related: 18
  unclear: 5 (5%)
Different people: 102
Pairs with a change word (now, moved, switched, since, stopped, later, replaced) / same day:
  contradiction: 12 of 46 / 24 of 46
  update: 10 of 17 / 0 of 17
  compatible_detail: 6 of 16 / 9 of 16
  merely_related: 7 of 18 / 7 of 18
  unclear: 0 of 5 / 3 of 5
Sentence patterns: none used more than 3 times in any block of 100 pairs
Unclear reasons: no start of 3 words used more than 3 times in any 100
Wall: no names of the 398 name words in data/m1/test.jsonl, no near copies of its pairs
Wall: no repeated or near-copied pairs from data/m1/train.jsonl
PASS
```

`uv run pytest`: 58 passed.

## Noticed, not fixed (outside this task)
The reasons of other labels follow templates too. In train, all 255 update reasons start "The newer memory...", and 457 of 697 contradiction reasons start "One says [name]...". The model learns to write reasons from these, so it may learn the template. The prompt for this task only asked about `unclear`, so they were left alone.

`data/m1/spotcheck-train.md` was drawn before these fixes, and some of its pairs now have other labels. STATE.md already says to ignore it (D-028).
