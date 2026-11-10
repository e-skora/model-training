# M2-08: publish what M2 taught us

**Where:** Claude Code (the desktop Code panel), **new session**, folder `~/code/model-training`, branch picker **`main`**, worktree box **unchecked**. Make the branch `m2-08-publish` from `main`.

Copy everything below the line into Claude Code.

---

Read `STATE.md` first, then `milestones/M2-rl-judge/SPEC.md`, `DECISIONS.md` (D-033, D-035, D-041, D-061, D-062), all of `LESSONS.md`, `results/m2/v2/summary.md`, `results/m2/grpo/full-run-summary.md`, and the `reviews/M2-*` files.

**Goal:** M2 Step 8. v2 missed the bar by one pair (D-041). Elias's rule for this publish (D-061): **the most important thing is to show what we learned, publish something useful that shows why the experiment was valuable, and make the second attempt use those lessons.** So the lessons write-up is the main piece, not the model. Everything stays **private**. Claude in Cowork checks it and makes it public. No GPU, no Modal, no model runs. Step 7 (the Mac version) is skipped (D-062).

## 1. The main piece: `docs/m2-what-we-learned.md` (public)

Write it for a reader who builds AI tools and has never seen this repo. Plain words, short sections, real numbers, and each claim linked to the file in the repo that shows it. Sections:

1. **What we tried, in five lines.** v1 judges whether two memories contradict. M2 trained it further with RL (GRPO) on harder pairs and set a bar before scoring.
2. **What happened.** The bar and the result: a miss by one pair (138 of 150 against 139). Show the v1-against-v2 table for both test sets, and the per-trap table.
3. **What RL changed, and what it did not.** It caught more contradictions in every version, raised fewer false alarms on the hard set, and answered `unclear` half as often. It did not fix `looks_like_clash`, and it got slightly worse on far-apart pairs when swapped. Late in training the 8 answers to a pair mostly agreed (`frac_reward_zero_std` about 0.8), so the last steps taught little. Steps 250, 300 and 314 scored the same.
4. **What we learned (the core).** Take these from `LESSONS.md`, the reviews and the results, not from memory. Group them into a few headings. Each lesson gets one sentence on what happened and one on what to do instead. At least: fixing the bar and the checkpoint rule before any score; a harder test set before training; one word ("invalid") meaning two things; guarding a run up to its last use (choose, upload); checks that count calls at every point; telling reviewers what a check must withstand (D-053); review loops that keep finding issues in newly added recovery code, and limiting the scope of a review; reading the installed library source instead of trusting a note; defaults a library fills in quietly; trial timing (the slow first step); cost estimates against the real bills.
5. **What it cost.** Every GPU run in M2 from `runs/RUNLOG.md`, the total, and against the estimate.
6. **What we would do differently: the plan for attempt 2.** A short numbered list. Each item names the lesson it comes from. These are proposals; Elias decides. Cover at least: a pool rebuilt from pairs the current model still gets wrong sometimes, since an easy pool stops teaching; more weight on the weak traps; whether to start from v2 or v1; a new hard test set written and frozen before any training; and whether the bar should be measured with more pairs, since one pair decided it.

## 2. Useful pieces others can reuse (public)

- **New private dataset repo `eskora/memory-contradiction-pairs-hard`:** `test-hard.jsonl` (byte for byte the frozen file; check it only through `judge.frozen`, never with `shasum`), `train-hard.jsonl`, `valid-hard.jsonl`, `rl-pool.jsonl`, and a card in `hf/memory-contradiction-pairs-hard/`. The card presents the hard test set as a small benchmark: the 11 trap groups and their counts, how it was written and blind-reviewed (D-039, D-042, D-045), and v1's and v2's scores on it as baselines. CC BY 4.0.
- **v2's card** in `hf/m2-contradiction-judge-qwen3-4b-grpo-lora/README.md`, uploaded to that private repo. Its first lines say it is an experiment that missed the bar fixed before scoring, by one pair, and that v1 is still the recommended model (link v1's repos). It links to `docs/m2-what-we-learned.md`. It covers how it was trained and chosen, the tables, and how to load it on the pinned base.
- **`README.md` on GitHub:** add M2: the result in two lines, a link to `docs/m2-what-we-learned.md` near the top, and the two new Hugging Face links. Point to the reusable parts: the reward function, the checkpoint-choice rule, `scripts/compare_v2.py`, and the run checks in `scripts/train_grpo.py`. Keep M1 as it is.

## 3. Lessons carried into attempt 2 (internal)

- Add the "what RL changed" note to `LESSONS.md` (SPEC item 5), pointing to the public write-up.
- Write `docs/attempt-2-inputs.md`: section 6's list, plus every open item, known limit and unchecked thing from M2 (for example the choose stage waiting 35 minutes for no known reason). Claude in Cowork reads this file first when it writes the attempt-2 plan.

## 4. Also

- **`scripts/prepublish_check.py`:** extend it to the new write-up, the two new repos and the README. It fails if public text names Aeverum or Mara Lindqvist, has an em dash, or if a repo is not private. Save its output in `hf/prepublish-check.txt`.
- **LinkedIn draft:** `docs/posts/m2-linkedin-draft.md`, about 150 words, in the style of `docs/posts/m1-linkedin-draft.md` and D-035. Lead with what was learned, be honest about the miss, link the write-up. No em dashes. Elias edits and posts it himself.
- **Aeverum note:** `docs/handoffs/2026-11-09-aeverum-m2-result.md`, with a copy in `~/code/Aeverum-Auditer/.handoffs/`. Three lines: v2 missed, v1 (`m1-judge:v1`) stays the candidate for step 12, and a second attempt is starting. Add its SHA-256 to STATE's open items.

**Never:** make anything public, change anything under `src/` or `data/`, or run a model.

**Finish:** run the tests, update `STATE.md`, add new terms to `GLOSSARY.md`, then commit and push `m2-08-publish`. Do not merge. Then tell Elias in plain words what is ready and that his next step is to tell Claude in Cowork.
