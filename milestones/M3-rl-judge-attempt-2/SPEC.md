# M3: Judge v3, the second attempt at beating v1 with RL

Written by Claude in Cowork, 2026-11-11 (D-061, D-063). Every step below carries the fixes in `docs/m2-lessons-and-fixes.md`; the codes in brackets (A2, C2, ...) point to the entry each step comes from.

## What and why
M2 trained v1 further with RL (GRPO) and got v2. v2 was better on most numbers but missed the bar by one pair (138 of 150 against 139), so it is not called better (D-041). M3 tries again with what M2 taught us. The goal is the same: a judge that beats v1 on a fresh hard test set, in v1's exact answer format (Style A, D-040), so Aeverum could swap it in with nothing else changing.

## Before anything: the lessons checklist
Every M3 prompt starts by reading `docs/m2-lessons-and-fixes.md` and names the entries it applies. At the end of M3 that file gets a new section for M3's lessons (the iteration loop).

## Steps

| Step | What | Who | Lessons |
|---|---|---|---|
| 1 | Write hard test set 2 (600 pairs) and hard validation set 2 (200 pairs) from a situation plan; cross-pair check; blind review with controls; freeze | Claude Code (prompt M3-01), then Claude in Cowork | A2, A3, A5, C1, C3, C4 |
| 2 | Score v1's and v2's adapters once on both new sets: the M3 baseline | Claude Code | A4 |
| 3 | Write new hard training pairs, weighted to the weak traps; `unclear` candidates checked by Elias | Claude Code, Elias (about 15 minutes), Claude in Cowork | B4, C1, C2, C3 |
| 4 | Build the round-1 pool from v2's answers: mixed pairs only, weak traps in both orders | Claude Code | B1, B2, B4 |
| 5 | Reuse M2's training script with the smallest changes: start from v2, two rounds with a fresh pool between, early stop; trial; one full Codex review of the changes, then narrow ones | Claude Code; Codex reviews | B2, B3, B5, D1, D2, D6, E1, E2, F1 |
| 6 | Fix the choice rule and the pass rule in code before any result | Claude in Cowork, then Claude Code | A1, A6, D4 |
| 7 | Full run (Elias approves the cost first), choose the checkpoint on validation set 2 | Claude Code | A5, F2 |
| 8 | Score the chosen model once on test set 2 and on the M1 test set | Claude Code, then Claude in Cowork rescores | A1, A2, A4, A6 |
| 9 | Publish what M3 taught us, with Elias's own sections; update the lessons file | Claude Code, Elias, Claude in Cowork | G3 |

## Step 1: the new sets (A2, A5, C1, C4)
- **Hard test set 2:** 600 pairs, 600 different people, 300 contradictions. The same 11 trap groups as M2, each with twice M2's count, so the two sets can be compared. No `unclear` pairs, as in M2.
- **Hard validation set 2:** 200 pairs, 100 contradictions, same trap mix. Never trained on.
- **Why 600:** the bar (+0.03 recall) is then 9 pairs, about twice the 4-pair difference between two forms of the same model (A2).
- **A situation plan first (C1):** one list gives every pair its person, topic and what changes, and each writer gets its own slice. Writers never see any model's answers.
- **Checks:** the existing checker, plus a new cross-pair check over every file that fails on pairs too alike. Names walled off from every earlier file. Dates must fit the time words (C4).
- **Blind review (C3):** a separate agent that never saw the brief judges all 600 test pairs and a sample of the validation pairs, mixed with control pairs. Then freeze.

## Step 3: training pairs and the human check (B4, C2)
- About 800 new hard training pairs. Half go to the three weak traps (`looks_like_clash`, `far_apart_no_change`, `paraphrase`), and the rest follow M2's mix. Same situation plan and cross-pair check.
- **`unclear` pairs, with Elias in the loop (C2):** Claude Code writes 45 candidates to the written test (one act, one missing detail, both answers ordinary). A blind agent judges each one. Elias marks each keep or drop on a one-page sheet (about 15 minutes), with the agent's view shown next to each pair. Up to 30 kept pairs are used. No `unclear` pair is used without Elias's keep.
- M2's training pairs stay available as candidates for the pool.

## Step 4 and 5: pool and training (B1, B2, B4, B5, D2)
- **Start from v2** (B5). The bar is still measured against v1.
- **Round 1 pool:** v2 answers every candidate 8 times at temperature 1.0, as in M2. Keep only pairs where the answers disagree. Pairs in the weak traps go in both orders (as written and swapped).
- **Round 2 pool:** after round 1, the round-1 model answers the candidates again and the pool is rebuilt the same way (B2).
- **Early stop:** a round stops and saves if the share of pairs whose 8 answers all score the same stays above 0.7 for 40 steps.
- **Same reward** (fingerprint unchanged), same pins, same sampling, same answer checks as M2 (B3, E1).
- **Smallest change to `scripts/train_grpo.py`** (D2): reuse every guard Codex accepted in M2. One full Codex review of the diff, then narrow reviews of fixes only. The Codex prompt says what each new check must withstand (D1).

## Step 6 to 8: choosing and the bar (A1, A2, A5, A6)
- **Choice:** D-056 and D-057, with validation set 2 in place of the M2 validation set and step 0 meaning v2. Coded before the run.
- **The bar, coded and committed before any test answer**, against v1's adapter on test set 2:
  1. Worst-case recall at least v1's + 0.03 (9 of 300 contradictions), worst-case precision no more than 0.01 below v1's.
  2. On the M1 test set, precision and recall each no more than 0.02 below v1's, in every version.
  3. 0 invalid answers, and 0 answers that break v1's format.
- **Reported with the bar, not part of it:** the paired count (pairs v1 misses and v3 gets, against the reverse), v2's scores, and per-trap results.
- **If it misses,** that is a result, written up plainly, as in M2.

## Elias's part
- Step 3: the `unclear` sheet, about 15 minutes.
- Step 7: yes or no to the full run's cost, if it is over $1 (D-019).
- Step 9: his own sections of the write-up.

## Cost (Claude's estimate from M2's bills)
- Baselines: about $0.50. Pool sampling, two rounds: about $1.50 to $2.50. Trial: about $0.20. Training, two rounds: about $1.50 to $2.50. Choice: about $0.80. Final scoring: about $0.30.
- **About $5 to $7 for all of M3**, inside Modal's free $30 a month (D-014). M2 cost about $3.50 to $4.
