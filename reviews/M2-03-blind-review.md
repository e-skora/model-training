# M2 Step 3: blind review of the hard training and validation pairs (Claude in Cowork, 2026-10-27)

**Files as written** (branch `m2-03-training-pairs` at `efd3251`):
- `train-hard.jsonl`: 1,000 pairs, SHA-256 `e3069a5f...14aa`.
- `valid-hard.jsonl`: 100 pairs, SHA-256 `a93ab3c6...261a`.

## How it was checked
1. **Checker and tests, run by Claude in Cowork on the Mac:** both files PASS against the frozen hard test set, the M1 test set and the other sets; 163 tests pass.
2. **Blind judging of 330 pairs** by three separate Claude agents, 110 each:
   - all 100 validation pairs;
   - all 30 training `unclear` pairs;
   - 200 other training pairs, chosen at random with seed 27.

   The agents saw only the person and the two memories, never the labels, tags or reasons. They followed the brief of the M2-01 review, changed only to say that some pairs may truly be `unclear`.

## Result

| Group | Pairs | Same label as the file |
|---|---|---|
| Training, not `unclear` | 200 | 199 |
| Validation, not `unclear` | 97 | 96 |
| `unclear` (30 training, 3 validation) | 33 | **1** |

- **Decided pairs:** every type matched wherever the label matched. The 2 disagreements are kept as written:
  - **tr2-0947:** two "now" statements 10 days apart, each claiming a season in the other position, so they clash.
  - **va2-008:** a judo bronze at fifteen and a wrestling bronze at seventeen can both be true, by reading rule 3.
- **`unclear` pairs:** the reviewers called 32 of the 33 decidable. They split between `non_contradiction` (22) and `contradiction` (10). Most follow one pattern: "X never does A", then an event that may or may not be A (ramen that may be pork, punch that may hold alcohol, an opera that may have been seen on a screen). For many, one reading is clearly more natural, so by reading rule 4 they are not `unclear`.

## Decision (D-045)
- **The 32 disputed `unclear` pairs are dropped, not relabelled**, as in M1 (D-030). They are kept in `data/m2/dropped-unclear.jsonl`, SHA-256 `cca2440878e0fa46762690141aa7a963e2eb0c91273811d91f153122f03bdec7`, so nothing is lost.
- **Why:** `unclear` pairs that a careful reader can decide would reward the dodge, and the dodge is v1's main miss on the hard set (`results/m2/baseline/summary.md`).

## Files after the decision
- `train-hard.jsonl`: 971 pairs, 1 `unclear` (the one both sides agree on), SHA-256 `8c2abddb93b21a0850c000c2919ee0bb11277ecf3b9e272d55cf9703110ca856`.
- `valid-hard.jsonl`: 97 pairs, 0 `unclear`, SHA-256 `dd29c90d5343c5e7b59208de90f0334a7b63b2849c7941f990bf38d18320b384`.

The RL pool takes its `unclear` share from M1's 47 `unclear` training pairs instead. Those already passed two blind reviews (D-030).
