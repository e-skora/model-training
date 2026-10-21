# Spot-check: M1 test set (revised)

**What to do (about 10 minutes):** read each of the 20 pairs below. Each one shows two memories about one person, the answer Claude Code gave, and why.

- If you agree with the answer, leave the line `Elias: OK` as it is.
- If you disagree, change that line to `Elias: WRONG, <why>`, for example `Elias: WRONG, these can both be true`.

Edit this file in place and save it. You do not need to commit anything. Then tell Claude in the Cowork project "spot-check done".

How to judge: two memories that say different things are true at the same moment are a contradiction, even if they were written on different dates. If the later memory says something changed, it is an update, not a contradiction. The change can be said plainly ("moved", "now", "switched") or indirectly ("her new flat", "after the renovation", "as of June"). A change word alone does not make an update: two "now" memories written the same day can still clash. If both can be true together, it is not a contradiction.

These 20 pairs were picked at random from `data/m1/test.jsonl` with seed 7: 10 contradictions and 10 non-contradictions (Python: `rng = random.Random(7)`, then `rng.sample(contradictions, 10)`, then `rng.sample(non_contradictions, 10)`). A plain draw of 20 with seed 7 gave only 7 contradictions, so the two groups are drawn separately. 10 of them are new hard cases, marked "Hard case" below.

## 1. t-010: Layla Hassan

- Memory A (written 2025-10-16): Layla's cat Simsim is afraid of the vacuum cleaner.
- Memory B (written 2025-11-22): Layla's cat Simsim loves to sit on the vacuum cleaner while it runs.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: A cat that is afraid of the vacuum cleaner would not sit on it while it runs, and no change is described.

Elias: OK

## 2. t-012: Cyrus Farahani

- Memory A (written 2026-05-01): Cyrus gets up before seven on weekdays.
- Memory B (written 2026-05-01): Cyrus gets up around 6:30 on weekdays.
- Label: non_contradiction
- Type: compatible_detail
- Reason: Around 6:30 is before seven, so the second memory gives a closer time and both are true.

Elias: OK

## 3. t-016: Noa Levi

- Memory A (written 2026-05-02): Noa switched to oat milk in her morning coffee.
- Memory B (written 2026-05-18): Noa drinks her afternoon coffee black.
- Label: non_contradiction
- Type: merely_related
- Reason: Milk in the morning and black coffee in the afternoon are two separate habits that can both be true.
- Hard case: has a change word, but nothing clashes

Elias: OK

## 4. t-018: Xin Li

- Memory A (written 2025-08-06): Xin owns a lot of books.
- Memory B (written 2025-08-06): Xin owns about 400 books, mostly science fiction.
- Label: non_contradiction
- Type: compatible_detail
- Reason: The second memory adds how many books and what kind, and both can be true.

Elias: OK

## 5. t-019: Grace Mwangi

- Memory A (written 2025-10-03): Grace grew up in Nakuru.
- Memory B (written 2026-03-12): Grace grew up in Eldoret.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Both say where Grace grew up and name different towns, which cannot both be true.

Elias: OK

## 6. t-024: Chidi Okeke

- Memory A (written 2025-09-25): Chidi's front door key has a green rubber cover.
- Memory B (written 2025-09-25): Chidi's front door key has a red rubber cover.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Both describe the same key on the same day and give different cover colours.

Elias: OK

## 7. t-026: Eleni Georgiou

- Memory A (written 2025-08-12): Eleni has long brown hair.
- Memory B (written 2025-12-03): Eleni's hair is short these days; she had it cut in October.
- Label: non_contradiction
- Type: update
- Reason: The later memory says Eleni had her hair cut, so short hair replaces long hair.
- Hard case: update said without the usual change words

Elias: OK

## 8. t-041: Giorgos Nikolaidis

- Memory A (written 2025-06-01): Since the kayak club closed, Giorgos stores his orange kayak in his parents' garage.
- Memory B (written 2025-06-01): Giorgos stores his orange kayak on a rack at the marina.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Written the same day, they put Giorgos's one orange kayak in two different places.
- Hard case: contradiction with a change word, but not an update

Elias: OK

## 9. t-053: Rashid Hamdan

- Memory A (written 2025-02-27): Rashid owns a grey cat called Shadow.
- Memory B (written 2025-12-11): Rashid gave Shadow to his sister when he moved, and since then Rashid has had no pets.
- Label: non_contradiction
- Type: update
- Reason: The later memory says Rashid gave the cat away, so it replaces the earlier one.

Elias: OK

## 10. t-057: Hamid Karimi

- Memory A (written 2025-06-28): Hamid loves Italian food.
- Memory B (written 2025-06-28): Hamid's favourite Italian dish is mushroom risotto.
- Label: non_contradiction
- Type: compatible_detail
- Reason: The second memory names Hamid's favourite dish within Italian food, and both can be true.

Elias: OK

## 11. t-085: Ifeoma Nwosu

- Memory A (written 2026-02-19): On Friday the needle on Ifeoma's sewing machine broke, and she replaced it herself at home.
- Memory B (written 2026-02-19): The needle that broke on Ifeoma's sewing machine on Friday was replaced at a repair shop, because Ifeoma did not want to do it herself.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: They give two different accounts of who replaced the same broken needle.
- Hard case: contradiction with a change word, but not an update

Elias: OK

## 12. t-093: Nikolai Sorokin

- Memory A (written 2025-05-06): Nikolai's grandmother taught him to play chess.
- Memory B (written 2025-09-23): Nikolai taught himself chess from a book, and nobody in his family plays.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: They disagree about how Nikolai learned chess and whether anyone in his family plays.

Elias: OK

## 13. t-104: Noor Jaber

- Memory A (written 2025-12-04): Noor first moved abroad in 2021, to Doha.
- Memory B (written 2026-02-18): Noor first moved abroad in 2023, to Dubai.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Both describe Noor's first move abroad and give a different year and city.
- Hard case: contradiction with a change word, but not an update

Elias: OK

## 14. t-105: Vikram Sethi

- Memory A (written 2025-08-13): Vikram plays cricket.
- Memory B (written 2025-08-13): Vikram plays cricket for the Sunday club team as a bowler.
- Label: non_contradiction
- Type: compatible_detail
- Reason: The second memory adds the team and Vikram's role, and both can be true.

Elias: OK

## 15. t-110: Ravi Menon

- Memory A (written 2025-01-15): Ravi drinks masala chai every afternoon.
- Memory B (written 2025-02-05): These days, on his doctor's advice, Ravi drinks jasmine tea in the afternoon instead of chai.
- Label: non_contradiction
- Type: update
- Reason: The later memory says Ravi drinks jasmine tea instead of chai these days, so it replaces the earlier habit.
- Hard case: update said without the usual change words

Elias: OK

## 16. t-128: Farah Rahimi

- Memory A (written 2025-01-27): Farah has commuted by train since the new line opened.
- Memory B (written 2025-02-03): Farah's train journey to work takes twenty-five minutes.
- Label: non_contradiction
- Type: compatible_detail
- Reason: The second memory adds how long Farah's train commute takes, and both can be true.
- Hard case: has a change word, but nothing clashes

Elias: OK

## 17. t-137: Jakub Nowak

- Memory A (written 2026-02-02): Jakub now works from home three days a week.
- Memory B (written 2026-02-02): Jakub's desk at home faces the window.
- Label: non_contradiction
- Type: merely_related
- Reason: Both are about Jakub working at home, as separate facts with no clash.
- Hard case: has a change word, but nothing clashes

Elias: OK

## 18. t-141: Sofia Ferreira

- Memory A (written 2025-04-02): Sofia stopped eating meat in 2018 and has not eaten any since.
- Memory B (written 2025-07-19): Sofia ate roast lamb at her cousin's wedding in 2023.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Eating lamb in 2023 clashes with Sofia not eating any meat since 2018.
- Hard case: contradiction with a change word, but not an update

Elias: OK

## 19. t-156: Samira Said

- Memory A (written 2025-05-08): Samira's morning walk always goes through the park.
- Memory B (written 2025-07-02): Samira's morning walk always goes along the river and never through the park.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: One says the walk always goes through the park and the other says it never does.

Elias: OK

## 20. t-169: Kamala Pillai

- Memory A (written 2025-10-24): Kamala's garden has had no trees since the storm last year.
- Memory B (written 2025-10-24): Kamala sits under the mango tree in her garden every afternoon.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: On the same day, one says Kamala's garden has no trees and the other has a mango tree in it.
- Hard case: contradiction with a change word, but not an update

Elias: OK
