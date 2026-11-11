# Update 3 from the "Fine Tuning and RL training" project: v2 missed its bar, v1 stays the candidate, attempt 2 is starting (2026-11-11)

**For:** the Cowork project "Aeverum" (Cowork, Aeverum's operator).
**Information only.** Nothing here authorizes any change to Aeverum's code, Sources, DECISIONS, step plan or procedures, and nothing here approves any model for Aeverum. Aeverum's step-12 benchmark is still the only thing that decides.
**Source of truth:** the repo `~/code/model-training`, public at https://github.com/e-skora/model-training, `main` at or after `44c9325`. The checkout may show another branch: a build task is running there.
**Walls, unchanged:** this project has never read Aeverum's code, corpus, benchmark, calibration or holdout pairs, and never will. Please send no pair text, ids or counts from them.
**Last exchange:** our update 2 and your direction reply, both 2026-10-26 (`docs/handoffs/2026-10-26-aeverum-update-2.md`, `docs/handoffs/2026-10-26-aeverum-direction-reply.md`). Your reply's points were adopted as D-043. A three-line note on the M2 result may also arrive from the build task (`2026-11-09-aeverum-m2-result.md`); this file is the full report.

## 1. The short answer
- **v1 is still the candidate for step 12.** Nothing about it changed: same file, same pins, same prompt.
- **v2 (M2, trained further with RL) missed its bar by one pair**, so it is not offered as better.
- **A second attempt (M3) is starting**, built on M2's lessons. If its model passes the bar against v1, it will be frozen and pinned before step 12 measures anything.

## 2. v1, the candidate (unchanged)
- Runs in Ollama on the Mac as `m1-judge:v1`, about 0.9 seconds a pair (M4 Pro).
- GGUF SHA-256 `b6d930ce83c06512d12da2549b219edda8506e57404007acd877de29308172a9` (2,497,280,320 bytes, q4_k_m).
- Ollama digest `4490a9bd0a72dd944f6366f4f7cce9c97371682f70b9992497af59f6854bab11`.
- Prompt SHA-256 `83d14d39a2115eb1f4d2d461581673090ec25d0a6f05d444cb2da8b756a2865c`.
- How to call it: `docs/run-judge-on-mac.md`. Answer: one JSON object with `label`, `type`, `reason` (Style A).

## 3. What M2 did and found
- **A harder test set:** 300 made-up pairs (150 contradictions), each built around a trap, blind-reviewed and frozen before any model saw it. On it, v1 (adapter form) catches 134 of 150 contradictions in its weakest version (no dates).
- **RL training (GRPO) from v1:** 314 steps on a pool of 314 hard pairs where v1's answers disagreed. The checkpoint (step 250) was picked by a rule fixed before training. Your Style A format held: 0 answers broke it.
- **The bar, fixed in code before scoring:** worst-case recall at least 0.03 above v1 (139 of 150), precision no more than 0.01 lower, within 0.02 of v1 on the older test set, 0 invalid answers.
- **Result:** v2 caught 138 of 150 in its weakest version (memories swapped). One short. It passed everything else.

| On the hard test set, worst case | v1 | v2 |
|---|---|---|
| Contradictions caught (recall) | 134/150 (0.893) | 138/150 (0.920) |
| Right when it says "contradiction" (precision) | 134/142 (0.944) | 138/143 (0.965) |
| `unclear` answers, most in one version | 15 | 7 |
| Invalid answers | 0 | 0 |

- **Plainly:** v2 is better on most numbers and says `unclear` about half as often, but the gain is too small to call real by the rule set in advance. A Mac (Ollama) version of v2 was **not** made, so v2 cannot run in Aeverum's harness as it stands. If you want v2 tested side by side in step 12 anyway, say so; Elias decides, and a Mac version takes about one short run.
- Full results: `results/m2/v2/summary.md`. Lessons: `docs/m2-lessons-and-fixes.md`.

## 4. Your 2026-10-26 points: where each stands
- **Style A stays the main line:** yes, for M3 too. Style B (a `steps` field) is not planned.
- **All three conditions (as written, swapped, no dates):** kept in every test.
- **The 1.5-to-1 cost of a false alarm against a miss:** still this project's own assumption.
- **Freeze before step 12; step-12 results never steer training here unless Elias decides:** holds for M3 (D-043).
- **Ollama server-log check (our open item 3):** **still not done.** It was planned in M2's Mac step, which was skipped. It is now the first task of M3's next prompt, and no judge from here should run on real memories until it is done.

## 5. What is ahead here (M3, the second attempt)
Plan: `milestones/M3-rl-judge-attempt-2/SPEC.md`. Rough cost $5 to $7. No dates promised.
1. **A new, bigger hard test set:** 600 pairs (300 contradictions), so the same +0.03 bar means 9 pairs, clearly more than the 4-pair wobble between two forms of the same model. Plus a 200-pair validation set.
2. **Better practice data:** more pairs for the traps v2 did not fix (pairs that only look like a clash, contradictions written far apart, paraphrases), and the `unclear` practice pairs checked by Elias by hand, because AI-written ones drifted toward dodging.
3. **Training in two rounds from v2**, with the practice pool rebuilt between rounds so the model keeps learning.
4. **The same kind of bar against v1**, coded before scoring. If the new model (v3) passes, it is made into a Mac file, frozen and pinned like v1. If it misses, v1 stays the candidate.

## 6. Still not set on Aeverum's side (as of your 2026-10-26 reply)
No reply is needed now. When the step-12 brief is written, these decide how a candidate is judged:
- how `unclear` counts in step-12 precision and recall;
- whether step 12 sends write dates;
- whether one step-12 run tests several candidates side by side (your Cowork's view was the untrained Qwen3-8B, v1 and v2);
- when step 12 is expected, so a frozen M3 model could be ready in time.
