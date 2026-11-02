# M2-04: check of the RL pool and blind review of the always-wrong pairs (Claude in Cowork, 2026-10-27)

**Files checked** (branch `m2-04-pool-and-reward` at `be82d97`):
- `data/m2/rl-pool.jsonl`: 314 pairs, SHA-256 `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77`.
- `data/m2/rl-pool-always-wrong.jsonl`: 9 pairs.
- `results/m2/pool/v1-samples.jsonl`: v1's 8 answers for each of the 1,018 candidates.

## 1. Tests and rescoring
- **Tests:** 240 pass (`uv run pytest`, run by Claude in Cowork on the Mac).
- **Reward table:** the values in `src/judge/reward.py` match the SPEC table (D-040) and D-046 line by line.
- **Every reward rescored:** Claude in Cowork wrote its own copy of the table from the SPEC and D-046 (not imported from `reward.py`) and rescored all 8,144 answers. All 323 stored reward lists (314 pool, 9 always-wrong) match exactly. Same counts as the summary: 262 mixed, 747 always right, 9 always wrong, 9 invalid answers.
- **Pool rule (D-046):** all 262 mixed pairs are in the pool; the other 52 are all always-right pairs; the always-wrong list is exactly the 9 pairs whose 8 rewards are all the same and wrong.
- **Walls:** the candidate ids are exactly the 971 M2 hard training pairs plus M1's 47 `unclear` training pairs; no pool id is in `valid-hard.jsonl`.
- **Strict reading:** one answer is readable JSON but invalid by the scorer (tr2-0228: `"type": "unclear"`, which is not a type). Counting it invalid is right.
- **Cost:** $0.86 billed by Modal for the three runs (`runs/RUNLOG.md`).

## 2. Blind review of the 9 always-wrong pairs
- **How:** two separate Claude agents (Sonnet) judged 20 pairs each with `reviews/blind-brief.md`: the 9 always-wrong pairs mixed with 11 always-right pool pairs (seed 20261027), so a judge that agrees with everything would show. They saw only the id, the person and the two memories, in the order and with the dates v1 was shown, never the labels, traps or v1's answers. The second agent got the pairs in reverse order.
- **Controls:** both agents gave the file's label and type on all 11.

| Pair | Trap | File label | v1 said (8 of 8) | Judge 1 | Judge 2 | Result |
|---|---|---|---|---|---|---|
| tr2-0015 | true_unclear | unclear | contradiction | unclear (not sure) | unclear (not sure) | label right |
| tr2-0303 | looks_like_clash | compatible_detail | contradiction | compatible_detail | compatible_detail | label right |
| tr2-0309 | paraphrase | contradiction | unclear | contradiction | contradiction | label right |
| tr2-0384 | update_with_overlap | update | contradiction | update | update | label right |
| tr2-0407 | small_difference | contradiction | compatible_detail | contradiction | contradiction | label right |
| tr2-0655 | explain_away | contradiction | compatible_detail | contradiction | contradiction (not sure) | label right |
| tr2-0773 | paraphrase | contradiction | unclear | contradiction | contradiction | label right |
| tr2-0913 | looks_like_clash | compatible_detail | contradiction | compatible_detail | compatible_detail | label right |
| tr-1527 | m1_unclear | unclear | compatible_detail | compatible_detail | compatible_detail | **disputed** |

## Decision (D-047)
- **tr-1527 is disputed** ("Rasmus gets up early" / "his alarm goes off at half past seven"): both judges, sure, call it compatible_detail. As in D-030 and D-045, a disputed `unclear` pair is dropped, not relabelled: it is never added to the RL pool. `data/m1/train.jsonl` is published and stays as it is.
- **The other 8 labels are right.** v1 is wrong, not the data. They stay out of the pool anyway: with 8 equal rewards, GRPO has nothing to learn from them. They are kept as a watch list of v1's hardest blind spots.
