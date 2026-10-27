# M2 baseline: v1 and the untrained models on the frozen hard test set

Date: 2026-10-26. Test set: `data/m2/test-hard.jsonl`, 300 pairs (150 contradictions), SHA-256 `d9e8e5f3...45ff82`, checked against `STATE.md` before each run. Each model answered it once, in all three versions: as is, swapped (memory B shown first), and no dates.

**The short answer:** v1 is good on the hard set but not near the top. At its worst it misses 12 (Mac) to 16 (adapter) of the 150 contradictions, and most of those misses are `unclear` dodges. So the M2 bar (+0.03 worst-case recall, about 5 more pairs caught) is reachable.

## How each model ran

| Model | Where | How | Folder |
|---|---|---|---|
| v1 on the Mac | Ollama 0.24.0, Apple M4 Pro | `m1-judge:v1`, GGUF SHA-256 `b6d930ce...72a9`, `scripts/run_ollama.py --final` | `m1-judge-v1/` |
| v1 adapter | Modal, L4 | epoch-2 adapter (D-031) at commit `1d9ebbf6...59b9` on `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`, `scripts/eval_trained.py --test` | `trained-v1/` |
| Qwen3-8B, untrained | Modal, L4 | `unsloth/Qwen3-8B` at `946bc9ac...`, thinking off, `scripts/run_baseline.py --test` | `qwen3-8b/` |
| Qwen3-4B-Instruct-2507, untrained | Modal, L4 | `unsloth/Qwen3-4B-Instruct-2507` at `99206368...`, `scripts/run_baseline.py --test` | `qwen3-4b-instruct-2507/` |

All four used the same prompt (SHA-256 `83d14d39...2865c`) and always picked the most likely next word (greedy, temperature 0). Each folder holds the raw answers (`<name>.<version>.jsonl`) and a settings file (`<name>.meta.json`) that records the test file and its SHA-256. Scored with `judge.score` (`uv run python -m judge.score data/m2/test-hard.jsonl <as_is> <swapped> <no_dates>`).

Before the real runs, each path was tried on 8 validation pairs from `data/m1/valid.jsonl`, written to a scratch folder outside `results/`. All four worked.

## The main table

| Model | Version | Precision | Recall | Accuracy | Type accuracy | Unclear | Invalid answers |
|---|---|---|---|---|---|---|---|
| **v1 on the Mac** | as is | **0.95** | **0.93** | **0.94** | 0.98 (of 142) | 8 | **0** |
| **v1 on the Mac** | swapped | **0.95** | **0.94** | **0.95** | 0.97 (of 143) | 6 | **0** |
| **v1 on the Mac** | no dates | **0.94** | **0.92** | **0.93** | 0.97 (of 141) | 10 | **0** |
| v1 adapter | as is | 0.96 | 0.92 | 0.94 | 0.99 (of 143) | 12 | 0 |
| v1 adapter | swapped | 0.96 | 0.91 | 0.92 | 0.96 (of 141) | 15 | 0 |
| v1 adapter | no dates | 0.94 | 0.89 | 0.92 | 0.99 (of 141) | 15 | 0 |
| Qwen3-8B untrained | as is | 0.93 | 0.93 | 0.93 | 0.64 (of 140) | 0 | 0 |
| Qwen3-8B untrained | swapped | 0.74 | 0.99 | 0.82 | 0.63 (of 98) | 0 | 0 |
| Qwen3-8B untrained | no dates | 0.90 | 0.97 | 0.93 | 0.78 (of 134) | 0 | 0 |
| Qwen3-4B untrained | as is | 0.92 | 0.84 | 0.85 | 0.82 (of 130) | 0 | 24 |
| Qwen3-4B untrained | swapped | 0.73 | 0.90 | 0.74 | 0.67 (of 88) | 0 | 17 |
| Qwen3-4B untrained | no dates | 0.89 | 0.93 | 0.85 | 0.82 (of 115) | 0 | 19 |

Exact counts (precision, recall):
- v1 on the Mac: as is 139/147, 139/150; swapped 141/148, 141/150; no dates 138/147, 138/150.
- v1 adapter: as is 138/144, 138/150; swapped 136/142, 136/150; no dates 134/142, 134/150.
- Qwen3-8B: as is 140/150, 140/150; swapped 148/200, 148/150; no dates 145/161, 145/150.
- Qwen3-4B: as is 126/137, 126/150; swapped 135/184, 135/150; no dates 140/157, 140/150.

The words: **precision** is how often "contradiction" was right when the model said it. **Recall** is how many of the 150 real contradictions it caught. **Type accuracy** is how often it named the right kind of non-contradiction. **Unclear** counts answers that said the text cannot settle it; every pair here can be settled, so each one is a miss or a wrong answer. One pair moves recall by about 0.007.

What the untrained rows show:
- **Qwen3-8B catches almost everything, because it calls almost everything a contradiction.** With the order swapped it raised 52 false alarms (precision 0.74). It also gets the type wrong about a third of the time.
- **Qwen3-4B still breaks the answer format:** 17 to 24 invalid answers per version.
- **The hard set is harder than the M1 test set**, but for the untrained 8B as is only a little (0.93 / 0.93 here, 0.95 / 0.96 on M1). Its traps bite most when the order is swapped.

## v1 on the Mac, trap by trap

"Right" counts pairs with the right label, out of the trap's pairs. On a contradiction trap this is the recall; on any other trap, every wrong answer is a false alarm or an `unclear`.

| Trap | Right answer | Pairs | Right, as is | Right, swapped | Right, no dates | `unclear` answers (all 3 versions) |
|---|---|---|---|---|---|---|
| paraphrase | contradiction | 30 | 25 | 26 | 25 | 14 |
| far_apart_no_change | contradiction | 25 | 22 | 23 | 23 | 5 |
| explain_away | contradiction | 40 | 38 | 39 | 37 | 3 |
| change_word_contradiction | contradiction | 25 | 24 | 23 | 24 | 2 |
| small_difference | contradiction | 30 | 30 | 30 | 29 | 0 |
| near_identical | merely_related | 25 | 21 | 22 | 21 | 0 |
| looks_like_clash | compatible_detail | 30 | 26 | 28 | 26 | 0 |
| indirect_update | update | 25 | 25 | 24 | 24 | 0 |
| rule5_bait | merely_related | 25 | 25 | 24 | 25 | 0 |
| near_identical | compatible_detail | 20 | 20 | 20 | 20 | 0 |
| update_with_overlap | update | 25 | 25 | 25 | 25 | 0 |

**v1's weakest traps:**
1. **paraphrase** (recall 0.83 to 0.87). The clash needs one step of everyday sense ("terrified of dogs" and "cuddles puppies at the shelter"). v1 answers `unclear` and invents a way out.
2. **near_identical merely_related** (21 to 22 of 25). One word is swapped ("oud" and "piano"), and v1 reads it as a clash. The blind reviewers fell for the same pairs (`reviews/M2-01-blind-review.md`).
3. **looks_like_clash** (26 to 28 of 30). The same thing is said two ways ("Bheki" and "Bhekizizwe", "fifteen minutes" and "a quarter of an hour"), and v1 calls it a clash.
4. **far_apart_no_change** (recall 0.88 to 0.92). Again mostly `unclear` dodges.

The adapter on Modal has the same weak traps, a little worse: far_apart_no_change recall is 0.84 in every version, with 12 `unclear` answers.

## v1 on the M1 test set and on the hard set

| v1 | Version | M1 test set: precision / recall | Hard set: precision / recall |
|---|---|---|---|
| on the Mac | as is | 0.98 / 0.96 | 0.95 / 0.93 |
| on the Mac | swapped | 0.94 / 0.96 | 0.95 / 0.94 |
| on the Mac | no dates | 0.98 / 0.94 | 0.94 / 0.92 |
| adapter | as is | 0.99 / 0.95 | 0.96 / 0.92 |
| adapter | swapped | 0.95 / 0.96 | 0.96 / 0.91 |
| adapter | no dates | 0.98 / 0.90 | 0.94 / 0.89 |

M1 numbers: `results/m1/gguf-v1/summary.md` and `results/m1/trained-v1/summary.md`. On the hard set, recall falls by 0.01 to 0.05 in every row, and precision by 0.03 to 0.04, except swapped precision, which stays about the same.

## The pairs v1 gets wrong most often

v1 on the Mac got 8 pairs wrong in all three versions. The 5 below are one per trap, taking the first in file order for each trap. The other three are h-233 and h-239 (paraphrase, both `unclear`) and h-262 (near_identical, "tar" and "setar" called a clash). v1's reason is quoted from the "as is" version.

1. **h-142, paraphrase (right answer: contradiction).** "Kweku is terrified of dogs." / "Kweku spends weekends volunteering at an animal shelter, cuddling the puppies." v1: `unclear`, "Fear of dogs can be mild or strong, and the text does not say which."
2. **h-278, far_apart_no_change (contradiction).** "Huda drives a white minivan." / "Huda does not have a driving licence." v1: `non_contradiction`, "Not having a licence means Huda cannot drive, so the minivan is either hers or borrowed." (`unclear` in the other two versions.)
3. **h-191, explain_away (contradiction).** "Solomiya is the oldest of three sisters." / "Solomiya's big sister Daryna is getting married in the spring." v1: `non_contradiction` (compatible_detail), "Both say Solomiya is the oldest of three sisters, and one adds that her big sister Daryna is getting married."
4. **h-020, near_identical (merely_related).** "Shadi's son plays the oud." / "Shadi's son plays the piano." v1: `contradiction`, "One says Shadi's son plays the piano, the other that he plays the oud." The blind reviewers said contradiction too; it stays `merely_related` by reading rule 3 (no "only").
5. **h-029, looks_like_clash (compatible_detail).** "Sizwe's best friend is Bheki." / "Sizwe's best friend, Bhekizizwe, sat next to Sizwe all through primary school." v1: `contradiction`, "They name different people as Sizwe's best friend." (The pair's written reason: Bheki is a short form of Bhekizizwe.)

The adapter got 12 pairs wrong in all three versions, including 7 of these 8. Eight of its 12 are `unclear` on a real contradiction, with an invented escape ("Veganism can mean no meat at all or just no meat from animals that are not farmed").

## Room for M2

D-041's bar: v2's worst-case recall (the lowest of the three versions) must rise by at least 0.03, with worst-case precision no more than 0.01 lower. v2's adapter is compared with v1's adapter, because RL starts from that adapter.

| v1 | Worst-case recall | Worst-case precision | Misses at worst | Needed for +0.03 |
|---|---|---|---|---|
| **adapter (the M2 comparison)** | **0.89** (134/150, no dates) | **0.94** (134/142, no dates) | 16 | 139/150, 5 more caught |
| on the Mac | 0.92 (138/150, no dates) | 0.94 (138/147, no dates) | 12 | 143/150, 5 more caught |

**Verdict: the bar is reachable, and v1 is not near the top.**
- At its worst the adapter misses 16 contradictions. 14 of those are `unclear` answers, not wrong labels. The M2 reward gives `unclear` -0.5 on a pair that can be decided (D-040), so this is exactly the habit RL is set up to change.
- The bar needs 5 of those 16 back, without many new false alarms: worst-case precision must stay above about 0.934 (v1's exact 134/142 is 0.944). The adapter raises 6 to 8 false alarms per version now.
- **Caution:** 5 pairs is still a small gain. v1 on the Mac and v1 as the adapter are the same model in two forms, yet their worst-case recall differs by 4 pairs (138 and 134). A gain of only 5 pairs would be real but close to that wobble.

## Cost

GPU price: Modal L4, $0.000222 a second (https://modal.com/pricing, checked 2026-10-26). Modal bills while a container loads and answers. "Inside the function" is the time the model spent loading and answering; "wall" is the whole run as seen from the Mac, including waiting for a GPU and starting the container. The real bill is between the two.

| Run | Inside the function | Wall (upper bound) |
|---|---|---|
| Smoke, v1 adapter, 8 pairs | 1.6 min | 6.3 min |
| Smoke, Qwen3-4B, 8 pairs | 1.3 min | 3.6 min |
| Smoke, Qwen3-8B, 8 pairs | 1.0 min | 4.7 min |
| Hard set, Qwen3-8B | 11.0 min | 12.5 min |
| Hard set, v1 adapter | 7.2 min | 17.4 min |
| Hard set, Qwen3-4B | 8.9 min | 25.8 min |
| **Total** | **31.0 min, about $0.41** | **70.2 min, about $0.94** |

The longest single run cost at most $0.34, under the $1 stop (D-019). The Mac run used no GPU on Modal (900 answers in 11.8 minutes, 0.79 seconds each).
