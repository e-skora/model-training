# M2 lessons and their fixes

Written by Claude in Cowork, 2026-11-11, at Elias's request (D-063): for every lesson and every thing that did not go well in M2, the best way to avoid repeating it, and where that fix is built into attempt 2 (`milestones/M3-rl-judge-attempt-2/SPEC.md`). Sources: `LESSONS.md`, `reviews/M2-*`, `results/m2/`, `runs/RUNLOG.md`.

Each entry: **what happened**, **why**, **the fix**, **where it lives in attempt 2**. "Keep" means it worked and attempt 2 does it again on purpose.

## The iteration loop (new standing process)

Every attempt ends with this file updated: one entry per lesson, each with a fix and a named place where the fix is enforced (a SPEC step, a checker rule, a test, or a decision). The next plan opens with a checklist of those entries, and its prompts point to them. A lesson without a place to live is not finished. This file is the source for the public write-up's lessons section.

## A. Testing fairly

**A1. Set the bar and the choice rule before any score. (Keep)**
- What happened: D-041, D-056, D-057 and D-060 were fixed and coded before results. That is why "missed by one pair" is a clean result.
- Fix: keep. The pass rule is code, committed before the first test answer, and its tests use made-up answers.
- In attempt 2: Steps 6 and 8; D-063.

**A2. One pair decided the result, and the test set was too small to show the gain clearly.**
- What happened: the bar was +0.03 recall on 150 contradictions, so 5 pairs. The same model in two forms (Mac file and adapter) already differs by 4 pairs. A real gain of 4 pairs could not be told apart from a change of form.
- Why: the hard test set was sized for M2's writing effort, not for the size of gain we wanted to see.
- Fix: decide the smallest gain worth seeing first, then size the test set so that gain is clearly bigger than the wobble. Attempt 2's hard test set has **300 contradictions in 600 pairs**, so +0.03 is 9 pairs, about twice the 4-pair wobble. The report also gives a paired count: pairs v1 misses and the new model gets, against the reverse.
- In attempt 2: Step 1 (size), Step 8 (paired count reported next to the bar).

**A3. A test can wear out. (Keep)**
- Fix: a new frozen test set for attempt 2. M2's hard set is published, so it stops being a fair final test.
- In attempt 2: Step 1.

**A4. Compare like with like. (Keep)**
- Fix: adapter against adapter, both scored by the same script.
- In attempt 2: Steps 2 and 8.

**A5. The validation set was too small to separate checkpoints.**
- What happened: on 97 validation pairs one false alarm moves precision by about 0.02, so the choice rule needed loose limits, and steps 250, 300 and 314 tied exactly.
- Fix: a bigger validation set, 200 pairs with 100 contradictions, written with the test set but kept apart from it.
- In attempt 2: Step 1, Step 7.

**A6. The rule's printout hid the counts (67/75 instead of 134/150).**
- Fix: every report prints "caught of real" counts, checked on a made-up example before the real run.
- In attempt 2: Step 8 (a test checks the printed text).

## B. What the training taught the model

**B1. The model learns nothing from pairs it always gets right. (Keep)**
- Fix: the pool keeps only pairs where the starting model's 8 answers disagree.
- In attempt 2: Step 4.

**B2. The practice pool got too easy part way through.**
- What happened: the share of pairs whose 8 answers all scored the same rose from 0.45 to about 0.8. The last 60 or so steps taught almost nothing, and three checkpoints scored the same.
- Why: the pool was picked once, from v1, before training. As the model improved, more of the pool became "always right".
- Fix: train in **two rounds**. After round 1, the round-1 model answers the candidate pairs again, and round 2's pool is rebuilt from the pairs it still gets wrong sometimes. Also, a run stops early (and saves) if that share stays above 0.7 for 40 steps, which saves money. Before writing this, Claude Code reads the installed TRL 1.15.0 source to see whether it already has a filter for such groups (lesson E1), and uses it only if it does exactly this.
- In attempt 2: Step 4 and Step 5.

**B3. RL fixed dodging. (Keep)**
- Fix: keep the reward table as it is (D-040, D-046). Changing it would make attempt 2 harder to compare with M2.
- In attempt 2: Step 5 (reward fingerprint unchanged).

**B4. Some traps did not improve.**
- What happened: `looks_like_clash` did not move at all (4 of 30 still wrong). Far-apart contradictions got slightly worse when the memories were swapped (21 to 20).
- Why: the pool had few of these, and each pool row was shown in one fixed order.
- Fix: write more training pairs for the three weakest traps (`looks_like_clash`, `far_apart_no_change`, `paraphrase`), and show every pool pair for these traps in **both orders**. Choosing still uses the fixed rule, and per-trap results are reported only, so they do not steer the choice.
- In attempt 2: Steps 3 and 4.

**B5. Where to start.**
- Fix: start from v2. It is better than v1 on most numbers, and starting there continues the progress instead of repeating it. The bar is still measured against v1, the recommended model.
- In attempt 2: Step 5; D-063.

## C. Making the data

**C1. Many AI writers repeat the same ideas.**
- What happened: 11 helpers wrote 8 "vegetarian orders meat" pairs, 7 "never owned a car" pairs and more. It was caught only by rereading everything.
- Why: each writer got the same brief and reached for the same everyday ideas, and the checker compared each file only with itself.
- Fix, two parts. (1) **Prevent:** before any writing, one plan lists every pair's situation (person, topic, what changes), and each writer gets its own slice. No two writers can then reach for the same idea. (2) **Catch:** `scripts/check_dataset.py` compares every pair with every other pair across all files, flags pairs whose texts are too alike, and fails above a set count.
- In attempt 2: Steps 1 and 3.

**C2. AI-written `unclear` pairs drift toward dodging, and an AI check from the same brief is not independent.**
- What happened: the blind review rejected 32 of 33 new `unclear` pairs, after a helper agent had passed them all.
- Why: an `unclear` pair is a judgment call about what an ordinary reader can decide. The writer and the helper shared the same habit of seeing doubt where a person sees none.
- Fix: **a small human check.** Claude Code writes 45 `unclear` candidates to the written test from 2026-10-27 (one act, one missing detail, both answers ordinary). Elias marks each one keep or drop on a one-page sheet, about 15 minutes. Only kept pairs are used, and at most 30 are needed. A blind agent that never saw the brief also judges them, and its view is shown to Elias next to each pair. This is the most efficient fix because `unclear` pairs are few, and they are exactly where AI judgment failed. Every other kind of pair is fine with AI writing plus blind review.
- In attempt 2: Step 3; D-063.

**C3. Blind reviews with control pairs catch real problems. (Keep)**
- Fix: keep, for the test set, the validation set and a sample of the training pairs. The reviewer never sees the writing brief.
- In attempt 2: Steps 1 and 3.

**C4. Dates drawn by a program must fit the time words in the text. (Keep the check)**
- In attempt 2: Step 1 (the checker rule from M2 stays).

## D. Safety checks and reviews

**D1. A review never ends until you say what a check must withstand.**
- What happened: the reward fingerprint took 5 Codex rounds, because each round found a new deliberate trick.
- Fix: every safety check in a prompt says up front what it must catch (honest mistakes) and what is a known limit (deliberate tricks, D-053). The Codex prompt includes that line.
- In attempt 2: Step 5's Codex prompt.

**D2. Each fix added new code that the next review had to check.**
- What happened: the GRPO script also took 5 Codex rounds. Most new findings were in recovery code that earlier fixes had added.
- Fix: (1) **reuse** M2's accepted script and change as little as possible, so most of the code is already reviewed; (2) the smallest fix for each finding; (3) one full Codex review of the changes, then narrow reviews of the fixes only (D-058).
- In attempt 2: Step 5.

**D3. Guard a result to its last use. (Keep)**
- Fix: keep the fingerprints from training through choosing and uploading.
- In attempt 2: Steps 5 to 7 reuse that code.

**D4. One word meant two things ("invalid").**
- Fix: a rule never uses a word alone. It names the function that decides it (for example `judge.reward.read_answer`).
- In attempt 2: every rule in the SPEC.

**D5. Tests read the real home folder through an import. (Fixed; keep)**
- Fix: `tests/conftest.py` stays. New test files go under it.

**D6. Checks moved away from what they guarded, and a fake of the whole job hid a bug.**
- Fix: a check sits at the last point before the effect, and tests drive the real job code, not a fake of it.
- In attempt 2: Step 5's Codex prompt.

## E. Tools

**E1. A protection we counted on in Unsloth never ran on our path.**
- Fix: for any setting or patch we rely on, read the installed source of the pinned version, and log what the run really used.
- In attempt 2: Step 5 (same pins as M2; any new pin means a new source read).

**E2. Unsloth quietly filled in a setting (`epsilon_high`).**
- Fix: set every value explicitly, and the run report copies every "Unsloth:" log line.
- In attempt 2: Step 5.

**E3. Check the exact model's card before suggesting a feature. (Keep)**

## F. Time and money

**F1. The slow first step and counting the saves twice made early estimates wrong.**
- Fix: estimate from the steady step rate plus one-off start-up, from the trial, and compare every estimate with the real bill afterwards.
- In attempt 2: Step 5 trial, every RUNLOG line.

**F2. Wall time can be much longer than GPU time (waiting for GPUs).**
- What happened: the M2 choice waited 35 minutes instead of about 7, and nobody checked why.
- Fix: every run logs the time spent waiting for a GPU separately from the work.
- In attempt 2: Steps 5 and 7.

## G. Working together

**G1. Commits to `main` while a branch is open cause merge conflicts in the control files.**
- Fix: keep both sides and the date order (2026-11-09 lesson). Claude in Cowork commits control files on the branch when Claude Code is idle.

**G2. A one-word approval must be quoted with its scope at once. (Keep)**
- Fix: every approval goes into DECISIONS with the question and Elias's exact words.

**G3. Elias wants the publish to show what we learned (D-061).**
- Fix: the lessons table is the core of the write-up. Claude writes the personal sections in Elias's voice from his recorded words only (D-064), so publishing does not wait on his time.
- In M2-08 and in attempt 2's Step 9.
