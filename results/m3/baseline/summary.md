# M3 baseline: v1 and v2 on test set 2 and validation set 2

Date: 2026-11-14. Claude Code, prompt `prompts/M3-02-baseline.md`, branch `m3-02-baseline`. Lessons applied: A2, A4, A6, F2.

**The short answer:** v2 already clears the bar M3 set for a new model. On test set 2, v1 catches at worst 268 of 300 contradictions. The M3 bar asks for 277, and v2 catches 286 at worst. v2 also raises fewer false alarms than v1 in every version. v1's misses are again mostly `unclear` dodges: 68 of its 90 missed contradictions across the three versions. These are facts only. Nothing here is chosen by them; the M3 rule is coded in Step 6.

## How it ran

- **Same script, same way as M2 (A4):** `scripts/eval_trained.py --model v1|v2 --test <set> --out <folder>`. v1 is the epoch-2 adapter `eskora/m1-contradiction-judge-qwen3-4b-lora` at `1d9ebbf6`. v2 is step-250 of `eskora/m2-contradiction-judge-qwen3-4b-grpo-lora` at `0964d031`, with its adapter fingerprints checked on the GPU before loading. Both sit on the same base model, `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at `7744afa8`. Both use the same prompt (SHA-256 `83d14d39...`), always pick the most likely next word (greedy), and answer every pair in three versions: as written, swapped (memory B shown first), and with no dates.
- **No scoring code changed.** The script took both new files as they are, because both are listed in `STATE.md`'s "Frozen files" and pass its fingerprint check (test `4bf37d6a...`, valid `f819f987...`, recorded in each run's `meta.json`).
- **Smoke run first:** v2 on 8 pairs of `data/m1/valid.jsonl`, 3 versions, in a scratch folder outside `results/`. All 24 answers matched step-250's saved validation answers word for word (`results/m2/grpo/valid/step-250.m1_valid.*`).
- **Each set answered once by each model:** `v1/test`, `v1/valid`, `v2/test`, `v2/valid` under `results/m3/baseline/`. The first v1 test run failed when the Mac lost its internet connection part way through. The job then stopped, and its answers never reached the Mac, so nothing was saved and nobody saw them. It was run again once the connection was back. That is the only repeat.
- **Report:** `scripts/m3_baseline_report.py` (committed at `4b55a35`, before any answer existed, with tests on made-up answers), output in `report.txt` and `report.json`. It prints counts, never reduced fractions (A6). M2's numbers come from `results/m2/v2/compare.json`, so no M2 or M1 test file was opened.

## The main table

"Caught/said" is precision: how often "contradiction" was right when the model said it. "Caught/real" is recall: how many real contradictions it found. "Format breaks" counts answers that `judge.reward.read_answer` rejects, meaning they break v1's exact answer format.

**Test set 2** (600 pairs, 300 contradictions):

| Model | Version | Caught / said (precision) | Caught / real (recall) | `unclear` | Invalid | Format breaks |
|---|---|---|---|---|---|---|
| v1 | as is | 273/283 (0.965) | 273/300 (0.910) | 21 | 0 | 0 |
| v1 | swapped | 269/279 (0.964) | 269/300 (0.897) | 28 | 0 | 0 |
| v1 | no dates | 268/280 (0.957) | 268/300 (0.893) | 25 | 0 | 0 |
| v2 | as is | 290/299 (0.970) | 290/300 (0.967) | 2 | 0 | 0 |
| v2 | swapped | 286/294 (0.973) | 286/300 (0.953) | 7 | 0 | 0 |
| v2 | no dates | 286/293 (0.976) | 286/300 (0.953) | 3 | 0 | 0 |

**Validation set 2** (200 pairs, 100 contradictions):

| Model | Version | Caught / said (precision) | Caught / real (recall) | `unclear` | Invalid | Format breaks |
|---|---|---|---|---|---|---|
| v1 | as is | 95/99 (0.960) | 95/100 (0.950) | 4 | 0 | 0 |
| v1 | swapped | 93/96 (0.969) | 93/100 (0.930) | 6 | 0 | 0 |
| v1 | no dates | 92/95 (0.968) | 92/100 (0.920) | 6 | 0 | 0 |
| v2 | as is | 98/102 (0.961) | 98/100 (0.980) | 0 | 0 | 0 |
| v2 | swapped | 97/99 (0.980) | 97/100 (0.970) | 2 | 0 | 0 |
| v2 | no dates | 97/101 (0.960) | 97/100 (0.970) | 0 | 0 | 0 |

Every pair in both sets can be decided, so every `unclear` answer is a miss or a wrong answer. Both models gave 0 invalid answers and 0 format breaks in all 2,400 answers each.

## What the M3 bar will be (a record only)

"Worst case" means the lowest of the three versions. The rule itself is coded in Step 6.

| | v1 on test set 2 | The bar | v2 now |
|---|---|---|---|
| Worst-case recall | 268/300 (no dates) | **at least 277/300** (v1 + 0.03 of 300 = 9 more caught) | 286/300 |
| Worst-case precision | 268/280 (no dates, 0.957) | **at least 663/700 (0.9471)** (v1 - 0.01). At exactly 277 caught, at most 292 "contradiction" answers, so at most 15 false alarms | 286/294 (0.973) |

## Per trap on test set 2, v1 against v2

Right labels out of the trap's pairs, in each version (as is / swapped / no dates). The last column adds up the `unclear` answers over all three versions.

| Trap | Right answer | Pairs | v1 | v2 | `unclear`, v1 / v2 |
|---|---|---|---|---|---|
| explain_away | contradiction | 80 | 71 / 72 / 72 | 79 / 79 / 80 | 25 / 2 |
| paraphrase | contradiction | 60 | 48 / 49 / 46 | 58 / 56 / 56 | 35 / 8 |
| small_difference | contradiction | 60 | 57 / 55 / 57 | 57 / 57 / 56 | 5 / 1 |
| change_word_contradiction | contradiction | 50 | 47 / 45 / 45 | 46 / 45 / 45 | 0 / 0 |
| far_apart_no_change | contradiction | 50 | 50 / 48 / 48 | 50 / 49 / 49 | 3 / 1 |
| looks_like_clash | compatible_detail | 60 | 53 / 54 / 56 | 52 / 55 / 56 | 2 / 0 |
| near_identical | compatible_detail | 40 | 40 / 39 / 40 | 40 / 39 / 40 | 0 / 0 |
| near_identical | merely_related | 50 | 49 / 45 / 48 | 49 / 48 / 49 | 3 / 0 |
| rule5_bait | merely_related | 50 | 50 / 50 / 50 | 50 / 50 / 50 | 0 / 0 |
| indirect_update | update | 50 | 48 / 49 / 43 | 50 / 50 / 49 | 1 / 0 |
| update_with_overlap | update | 50 | 50 / 49 / 49 | 50 / 50 / 49 | 0 / 0 |

- v2's gain sits almost all in **explain_away** and **paraphrase**, the two traps where v1 says `unclear` most (60 of v1's 74 `unclear` answers).
- **looks_like_clash** stays where it was in M2: v1 and v2 both get 163 of 180 answers right.
- **change_word_contradiction** is the one trap where v1 is ahead, by one answer (137 against 136 of 150).

## Pair by pair on test set 2

| Version | v1 wrong, v2 right | v1 right, v2 wrong | Contradictions only: v1 misses, v2 catches | Contradictions only: v2 misses, v1 catches |
|---|---|---|---|---|
| as is | 21 | 3 | 19 | 2 |
| swapped | 23 | 0 | 17 | 0 |
| no dates | 26 | 1 | 19 | 1 |

The pair ids are in `report.json` under `paired`.

## How this differs from M2's hard set (facts only)

M2's numbers are from `results/m2/v2/compare.json` (M2 hard set, 300 pairs, 150 contradictions).

| | M2 hard set | M3 test set 2 |
|---|---|---|
| v1 worst-case recall | 134/150 (0.893) | 268/300 (0.893) |
| v1 worst-case precision | 134/142 (0.944) | 268/280 (0.957) |
| v2 worst-case recall | 138/150 (0.920) | 286/300 (0.953) |
| v2 worst-case precision | 138/143 (0.965) | 286/294 (0.973) |
| v2's lead in worst-case recall | 4 pairs of 150 | 18 pairs of 300 |
| `unclear` answers, all 3 versions, v1 / v2 | 42 / 18 (of 900 answers) | 74 / 12 (of 1,800 answers) |

- **v1** is the same at its worst (0.893 recall on both sets) and raises a little fewer false alarms on test set 2.
- **v2** is better on test set 2 than on M2's set. Its lead over v1 is 18 pairs here, against 4 in M2.

**Harder or easier, per trap.** Share of right answers over all three versions, M2 hard set to test set 2:

| Trap | v1 | v2 |
|---|---|---|
| far_apart_no_change (contradiction) | 0.840 to 0.973 | 0.893 to 0.987 |
| near_identical (merely_related) | 0.867 to 0.947 | 0.987 to 0.973 |
| rule5_bait (merely_related) | 0.973 to 1.000 | 0.987 to 1.000 |
| looks_like_clash (compatible_detail) | 0.889 to 0.906 | 0.889 to 0.906 |
| small_difference (contradiction) | 1.000 to 0.939 | 1.000 to 0.944 |
| change_word_contradiction (contradiction) | 0.920 to 0.913 | 0.933 to 0.907 |
| near_identical (compatible_detail) | 1.000 to 0.992 | 1.000 to 0.992 |
| update_with_overlap (update) | 1.000 to 0.987 | 1.000 to 0.993 |
| explain_away (contradiction) | 0.925 to 0.896 | 0.967 to 0.992 |
| paraphrase (contradiction) | 0.833 to 0.794 | 0.878 to 0.944 |
| indirect_update (update) | 0.960 to 0.933 | 0.973 to 0.993 |

- **Easier for both:** far_apart_no_change (by a lot), rule5_bait, and looks_like_clash (a little).
- **Harder for both:** small_difference (no misses in M2, about 6% wrong now), change_word_contradiction, and, a little, near_identical (compatible_detail) and update_with_overlap.
- **Mixed:** near_identical (merely_related) is easier for v1 and a little harder for v2. explain_away, paraphrase and indirect_update are harder for v1 and easier for v2.

So test set 2 is not simply harder or easier than M2's set. v1's overall worst case is the same on both.

## Time and cost (F2)

"Inside" is the time the job spent loading and answering on the GPU. "Waiting" is the wall time left over: waiting for a GPU plus starting the container. Billed costs come from `modal billing report --for today -r h --tz local --show-resources --json` (https://modal.com/docs/sdk/py/latest/Workspace#billingreport). It was read at 14:06 PST, when the last run had just ended, so its 14:00 hour may still grow by a cent or so.

| Run | App | Wall | Inside (load + answer) | Waiting | Billed |
|---|---|---|---|---|---|
| Smoke, v2, 8 M1 valid pairs | ap-9ZTCNAtbPQYcSxEX3jYTdL | 7.6 min | 1.1 min (0.9 + 0.3) | 6.5 min | $0.02 |
| v1, validation set 2 | ap-KiFFPdHPQNhHZmGIgwD0WQ | 6.7 min | 5.7 min (1.1 + 4.5) | 1.1 min | $0.09 |
| v2, test set 2 | ap-IZNBWp6M2egC16DstDat4v | 14.6 min | 14.6 min (1.0 + 13.6) | 0.1 min | $0.21 |
| v2, validation set 2 | ap-UP0fSmQ5NdgUYEbO3E9KjR | 16.0 min | 5.4 min (0.9 + 4.5) | 10.5 min | $0.08 |
| v1, test set 2, first try (failed, nothing saved) | ap-41jZrgTMua3IpRmv3p7Eox | 25 min (12:25 to 12:50) | about 5.5 min of L4 billed | most of the rest | $0.08 |
| v1, test set 2 | ap-y2G9feeqZSUvR7O6Bax75e | 18.4 min | 14.8 min (1.1 + 13.7) | 3.7 min | $0.21 |
| **Total** | | | | | **$0.70** |

Every run stayed under $1. The total is $0.10 over the $0.60 estimate, and the failed first try accounts for $0.08 of that. The four real runs started within 8 seconds of each other, so they waited for GPUs at the same time. The v2 validation run waited longest, 10.5 minutes.
