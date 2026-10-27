# M2 Step 1: blind review of the hard test set (Claude in Cowork, 2026-10-26)

**File:** `data/m2/test-hard.jsonl`, 300 pairs, SHA-256 `d9e8e5f3a53369981d11c835a0a4bf68b9b105ae0c8dd78b2d02df967645ff82`, from branch `m2-01-hard-test` at `100e59e`.

## How it was checked
1. **Checker and tests, run by Claude in Cowork on the Mac:**
   - `scripts/check_dataset.py` with `--against data/m1/test.jsonl` and `--distinct-from` for M1 train and valid: PASS.
   - Counts are as in prompt M2-01.
   - 300 different people; no names or near copies from any M1 file.
   - 139 tests pass.
2. **Blind judging of all 300 pairs:**
   - Two separate Claude agents each judged 150 pairs. They saw only the person and the two memories, never the labels, `hard` tags or reasons.
   - They followed one brief: labels, reading rules 1 to 5, and a second judgment with the dates hidden.
   - The order was shuffled with seed 5.

## Result
- **Label:** same as the file on 296 of 300 (98.7%).
- **Type:** same on every pair where the label matched.
- **Dates hidden:** only those same 4 pairs changed label, so the rest are decidable from text alone (D-038).
- **Close calls:** the judges marked 12 pairs as not sure. All 12 match the file's label except the 4 below.

## The 4 disagreements: all kept as written
All 4 are `near_identical` `merely_related` pairs. The judges read a swapped word as a clash:

| Pair | Memories | File | Judges |
|---|---|---|---|
| h-020 | son plays the oud / son plays the piano | merely_related | contradiction |
| h-023 | brother teaches maths / brother studies maths | merely_related | contradiction |
| h-106 | keeps bees / keeps hens, behind the summer house | merely_related | contradiction |
| h-262 | learned the tar as a child / learned the setar as a child | merely_related | contradiction |

- **Why they stay:** both memories can be true, and neither has an exclusive word ("only", "just", "never"). Reading rule 3 makes them non-contradictions, the same pattern as M1's "She hikes" / "She owns hiking boots".
- **What the disagreement shows:** the trap works. A careful reader, and so likely a model, is pulled toward calling a swapped word a clash.
- **No label changed.** No pair was found wrong.

## Noted, not changed
- 15 pairs are about a relative of the person ("Chinwe's sister Nneka ..."). This is realistic for a memory app, and all 15 were judged the same as the file.
- h-196 (grandmother born in Tripoli / in Sidon) stays a contradiction by reading rule 4: "maybe two grandmothers" is an invented escape, and the judges agreed.

## Decision
Frozen on 2026-10-26 as written. The SHA-256 is recorded under "Frozen files" in `STATE.md`. Never edit it, and never train on it.
