# M2 Step 6: v2 on the two frozen test sets

**Answer: v2 misses the D-041 bar by one pair.** On the hard test set, v2's worst-case recall is 138 of 150 (0.920). The bar was 139 of 150 (v1's 134 + 0.03 x 150 = 138.5, rounded up). Everything else passes: v2's worst-case precision on the hard set went up, v2 holds on the M1 test set in every version, and it gave 0 invalid answers. By the rule fixed before any score (D-060), the result is **MISS**. v2 is not published as better (D-041).

v2 is better than v1 in most numbers. It catches more contradictions in every version of both sets, it raises fewer false alarms on the hard set, and it says `unclear` about half as often. But the gain in its weakest version (swapped) is 4 pairs, and the bar asked for 5.

## How it was run

- **v2:** step-250 of the GRPO run, `eskora/m2-contradiction-judge-qwen3-4b-grpo-lora` at commit `0964d031073408707397ce7438fc7df7b8a772c0`, on v1's pinned base `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at `7744afa8`. Before answering, the GPU job checked the adapter files at that commit against the run's saved step-250 fingerprints (`adapter_model.safetensors` `7e69587c...63a1f7`, `adapter_config.json` `84bd07b5...0d6d6c`); both matched.
- **Same scoring as v1:** `scripts/eval_trained.py --model v2`, the same loader, prompt (fingerprint `83d14d39...865c`), greedy settings and answer loop. Nothing under `src/` changed; the reward fingerprint is still `769f58fe...025e7`.
- **Smoke run first:** 8 validation pairs in 3 versions, in a scratch folder. All 24 answers were word for word the same as step-250's saved validation answers from Step 5, so the copy on Hugging Face is the chosen model.
- **Each test set answered once,** by v2 only: hard set into `results/m2/v2/hard/` (900 answers), M1 set into `results/m2/v2/m1/` (600 answers). v1 was not run again; its saved answers are the baseline (`results/m2/baseline/trained-v1/`, `results/m1/trained-v1/`).
- **The rule** is `scripts/compare_v2.py`, committed and pushed at `1def5c5` before any GPU run. Its full output: `compare-output.txt`; its numbers: `compare.json`. (Its printout shows fractions in lowest terms, so 134/150 shows as 67/75; the counts below are the plain ones.)

## The D-041 items

| Item | Rule | v1 | Bar | v2 | Result |
|---|---|---|---|---|---|
| 1 | Hard set, worst-case recall at least v1 + 0.03 | 134/150 (0.893, no dates) | 139/150 (0.923) | 138/150 (0.920, swapped) | **MISS** |
| 1 | Hard set, worst-case precision at least v1 - 0.01 | 134/142 (0.944, no dates) | 0.934 | 138/143 (0.965, swapped) | PASS |
| 2 | M1 set, each version's precision and recall at least v1 - 0.02 | see table | | all 6 above the bar | PASS |
| 3 | 0 invalid answers, every version of both sets | | 0 | 0 | PASS |

Item 1 is a MISS, so overall: **MISS**.

Answers that break v1's exact format (no type or no reason, `judge.reward.read_answer`): 0 in every version of both sets.

## Every count

"Caught" is real contradictions the model called contradictions; "said" is every time it answered contradiction; "real" is the real contradictions in the set.

**Hard test set** (300 pairs, 150 contradictions)

| Model | Version | Caught / said | Precision | Recall | `unclear` | Invalid |
|---|---|---|---|---|---|---|
| v1 | as written | 138 / 144 | 0.958 | 0.920 | 12 | 0 |
| v1 | swapped | 136 / 142 | 0.958 | 0.907 | 15 | 0 |
| v1 | no dates | 134 / 142 | 0.944 | 0.893 | 15 | 0 |
| v2 | as written | 141 / 145 | 0.972 | 0.940 | 6 | 0 |
| v2 | swapped | 138 / 143 | 0.965 | 0.920 | 7 | 0 |
| v2 | no dates | 143 / 148 | 0.966 | 0.953 | 5 | 0 |

**M1 test set** (200 pairs, 100 contradictions)

| Model | Version | Caught / said | Precision | Recall | `unclear` | Invalid |
|---|---|---|---|---|---|---|
| v1 | as written | 95 / 96 | 0.990 | 0.950 | 3 | 0 |
| v1 | swapped | 96 / 101 | 0.950 | 0.960 | 6 | 0 |
| v1 | no dates | 90 / 92 | 0.978 | 0.900 | 3 | 0 |
| v2 | as written | 96 / 97 | 0.990 | 0.960 | 2 | 0 |
| v2 | swapped | 98 / 103 | 0.951 | 0.980 | 1 | 0 |
| v2 | no dates | 92 / 94 | 0.979 | 0.920 | 1 | 0 |

Format errors (a known label but a broken type or reason): 0 for both models in every version of both sets.

## Per trap on the hard set

Right labels out of the trap's pairs, in each version (as written / swapped / no dates), from `judge.score.score_by_trap`.

| Trap (right answer) | Pairs | v1 | v2 |
|---|---|---|---|
| explain_away (contradiction) | 40 | 38 / 38 / 35 | 38 / 39 / 39 |
| paraphrase (contradiction) | 30 | 25 / 25 / 25 | 26 / 26 / 27 |
| small_difference (contradiction) | 30 | 30 / 30 / 30 | 30 / 30 / 30 |
| change_word_contradiction (contradiction) | 25 | 24 / 22 / 23 | 24 / 23 / 23 |
| far_apart_no_change (contradiction) | 25 | 21 / 21 / 21 | 23 / **20** / 24 |
| indirect_update (update) | 25 | 25 / 24 / 23 | 25 / 24 / 24 |
| update_with_overlap (update) | 25 | 25 / 25 / 25 | 25 / 25 / 25 |
| near_identical (compatible_detail) | 20 | 20 / 20 / 20 | 20 / 20 / 20 |
| looks_like_clash (compatible_detail) | 30 | 26 / 28 / 26 | 26 / 28 / 26 |
| near_identical (merely_related) | 25 | 22 / 21 / 22 | 25 / 24 / 25 |
| rule5_bait (merely_related) | 25 | 25 / 23 / 25 | 25 / 24 / 25 |

What this shows, plainly:
- v2's biggest gains are on contradictions a reader could explain away when the dates are gone (35 to 39), on paraphrased contradictions, and on not calling "merely related" pairs contradictions (near_identical, 22 to 25).
- The one place v2 is worse than v1 is far-apart pairs with the memories swapped (21 to 20). That version is v2's weakest, and it is why item 1 misses by one pair.
- `looks_like_clash` did not move at all: 4 of 30 still wrong as written and with no dates.

## Cost and time

| Run | Inside the GPU job | Start to finish | Modal app |
|---|---|---|---|
| Smoke, 8 validation pairs x 3 | load 65 s, answer 17 s | 2.8 min | `ap-gqxPFLDqSHQ8XTTb9czumY` |
| Hard test set, 300 pairs x 3 | load 61 s, answer 380 s | 19.8 min (most of the rest was waiting for a GPU) | `ap-LrRGHjxLB8uYLxGbVhUf8r` |
| M1 test set, 200 pairs x 3 | load 63 s, answer 265 s | 9.2 min | `ap-psIUmSMEKwKHKMyjzWU2B8` |

Billed by Modal (`modal billing report`): smoke $0.02, hard set $0.11, M1 set $0.08; **$0.22 in all**. Lines in `runs/RUNLOG.md`.

## What happens next

By D-041, M2 still ends: the "what RL changed" note says plainly what RL did and did not change, and v2 is not published as better. Step 7 (GGUF for the Mac) waits for Elias. Claude in Cowork rescores v2 from these saved answers on its own and merges.
