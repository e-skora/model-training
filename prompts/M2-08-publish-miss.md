# M2-08: publish what M2 taught us

**Where:** Claude Code (the desktop Code panel), **new session**, folder `~/code/model-training`, branch picker **`main`**, worktree box **unchecked**. Make the branch `m2-08-publish` from `main`.

Copy everything below the line into Claude Code.

---

Read `STATE.md` first, then `docs/m2-lessons-and-fixes.md`, `milestones/M3-rl-judge-attempt-2/SPEC.md`, `milestones/M2-rl-judge/SPEC.md`, `DECISIONS.md` (D-033, D-035, D-041, D-061, D-062, D-063), all of `LESSONS.md`, `results/m2/v2/summary.md`, `results/m2/grpo/full-run-summary.md`, and the `reviews/M2-*` files.

**Goal:** M2 Step 8. v2 missed the bar by one pair (D-041). Elias's rule for this publish (D-061): **the most important thing is to show what we learned, publish something useful that shows why the experiment was valuable, and make the second attempt use those lessons.** So the lessons write-up is the main piece, not the model. Everything stays **private**. Claude in Cowork checks it and makes it public. No GPU, no Modal, no model runs. Step 7 (the Mac version) is skipped (D-062).

## 1. The main piece: `docs/m2-what-we-learned.md` (public)

Write it for a reader who builds AI tools and has never seen this repo. Plain words, short sections, real numbers, and each claim linked to the file in the repo that shows it. Sections:

1. **Why I ran this.** Two or three sentences in Elias's voice, built only from his own recorded words: D-061 ("show what we learned and publish something useful to show how this experiment was valuable") and D-035 for how he is described. Do not invent feelings or facts about him.
2. **What we tried, in five lines.** v1 judges whether two memories contradict. M2 trained it further with RL (GRPO) on harder pairs and set a bar before scoring.
3. **What happened.** The bar and the result: a miss by one pair (138 of 150 against 139). The v1-against-v2 table for both test sets, and the per-trap table.
4. **What RL changed, and what it did not.** It caught more contradictions in every version, raised fewer false alarms on the hard set, and answered `unclear` half as often. It did not fix `looks_like_clash`, and it got slightly worse on far-apart pairs when swapped. Late in training the 8 answers to a pair mostly agreed (`frac_reward_zero_std` about 0.8), so the last steps taught little.
5. **What we learned, and what we changed because of it (the core).** Built from `docs/m2-lessons-and-fixes.md`, which is the source. A table with one row per lesson: what happened, the fix, and where attempt 2 applies it. Keep the public lessons to those a reader building AI tools could use (A1, A2, A5, B1, B2, B4, C1, C2, C3, D1, D2, E1, F1). Explain the human check on `unclear` pairs (C2) in two or three sentences. Do not copy the file whole; link it.
6. **What it cost.** Every M2 GPU run from `runs/RUNLOG.md`, the total (about $3.50 to $4), and estimates against the bills.
7. **Attempt 2.** Five lines on `milestones/M3-rl-judge-attempt-2/SPEC.md`: the bigger test set, the two-round training, the human check, and the same honest bar. Link it.
8. **My takeaway.** Two or three sentences in Elias's voice: a near miss judged by a rule fixed in advance is still a result, and every lesson became a fix in attempt 2 (D-061, D-063). Same rule: only what his recorded words support.

## 2. Useful pieces others can reuse (public)

- **New private dataset repo `eskora/memory-contradiction-pairs-hard`:** `test-hard.jsonl` (byte for byte the frozen file; check it only through `judge.frozen`, never with `shasum`), `train-hard.jsonl`, `valid-hard.jsonl`, `rl-pool.jsonl`, and a card in `hf/memory-contradiction-pairs-hard/`. The card presents the hard test set as a small benchmark: the 11 trap groups and their counts, how it was written and blind-reviewed (D-039, D-042, D-045), and v1's and v2's scores on it as baselines. CC BY 4.0.
- **v2's card** in `hf/m2-contradiction-judge-qwen3-4b-grpo-lora/README.md`, uploaded to that private repo. Its first lines say it is an experiment that missed the bar fixed before scoring, by one pair, and that v1 is still the recommended model (link v1's repos). It links to `docs/m2-what-we-learned.md`. It covers how it was trained and chosen, the tables, and how to load it on the pinned base.
- **`README.md` on GitHub:** add M2: the result in two lines, a link to `docs/m2-what-we-learned.md` near the top, and the two new Hugging Face links. Point to the reusable parts: the reward function, the checkpoint-choice rule, `scripts/compare_v2.py`, and the run checks in `scripts/train_grpo.py`. Keep M1 as it is.

## 3. Lessons file

- Add the "what RL changed" note to `LESSONS.md` (SPEC item 5), pointing to the public write-up and to `docs/m2-lessons-and-fixes.md`.

## 4. Also

- **`scripts/prepublish_check.py`:** extend it to the new write-up, the two new repos and the README. It fails if public text names Aeverum or Mara Lindqvist, has an em dash, or if a repo is not private. Save its output in `hf/prepublish-check.txt`.
- **LinkedIn draft:** `docs/posts/m2-linkedin-draft.md`, about 150 words, in the style of `docs/posts/m1-linkedin-draft.md` and D-035. Lead with what was learned, be honest about the miss, link the write-up. No em dashes. Elias edits and posts it himself.
- **Aeverum note:** `docs/handoffs/2026-11-09-aeverum-m2-result.md`, with a copy in `~/code/Aeverum-Auditer/.handoffs/`. Three lines: v2 missed, v1 (`m1-judge:v1`) stays the candidate for step 12, and a second attempt is starting. Add its SHA-256 to STATE's open items.

**Never:** make anything public, change anything under `src/` or `data/`, or run a model.

**Finish:** run the tests, update `STATE.md`, add new terms to `GLOSSARY.md`, then commit and push `m2-08-publish`. Do not merge. Then tell Elias in plain words what is ready and that his next step is to tell Claude in Cowork.
