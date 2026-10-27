# Continuation: M2 (judge v2 with reinforcement learning), after prompt M2-04

**Where:** the Cowork project "Fine Tuning and RL training", **new chat**.
**Folders:** `~/code/model-training` must be connected. Also connect `~/code/Aeverum-Auditer` if an Aeverum handoff needs reading; read only.

Copy everything below the line into that chat. Nothing else is needed: Elias already pasted M2-04 into Claude Code, and this session reads its results from disk.

---

This is a new session of the "Fine Tuning and RL training" project. We are in M2: improving the contradiction judge with reinforcement learning (GRPO). I already pasted prompt M2-04 into Claude Code on 2026-10-27; do not ask me to paste it or its report. Read its work from disk, on the branch `m2-04-pool-and-reward`. If it is not finished yet (no "M2-04 written" line in that branch's `STATE.md`), say so in one line and wait.

**Open the session the usual way.**
1. Read, in this order:
   - `STATE.md`;
   - `milestones/M2-rl-judge/SPEC.md`;
   - `DECISIONS.md` D-036 to D-046;
   - `LESSONS.md` (the 2026-10-26 and 2026-10-27 entries);
   - `results/m2/baseline/summary.md`;
   - `reviews/M2-01-blind-review.md` and `reviews/M2-03-blind-review.md`.
2. Check the files against git (`main` and the branch `m2-04-pool-and-reward`). The files win over this prompt.
3. Tell me in one to three plain sentences where things stand and the next action.

**Where things stand (check it against the files):**
- **M1 is done and public:** judge v1, `m1-judge:v1` in Ollama.
- **M2 Step 1:** a 300-pair hard test set, frozen (SHA-256 in `STATE.md`).
- **M2 Step 2:** v1's adapter at its worst, without dates, catches 134 of 150 contradictions (precision 0.94). 14 of its 16 misses are `unclear` dodges. The bar (D-041) is 139 of 150, with precision no more than 0.01 lower. Compare adapter with adapter only.
- **M2 Step 3 part A:** `data/m2/train-hard.jsonl` (971 pairs) and `data/m2/valid-hard.jsonl` (97 pairs), after a blind review. The 32 disputed `unclear` pairs were dropped (D-045). The RL pool's `unclear` share is M1's 47 checked `unclear` pairs.
- **M2-04 (already pasted into Claude Code; read its results from the branch):**
  - Part A: the reward function `src/judge/reward.py`, following the D-040 and D-046 tables.
  - Part B: the RL pool. v1 answers each candidate 8 times, and pairs whose rewards differ join the pool.
- **Aeverum:** its direction reply was adopted as D-043. Far-apart pairs with no change shown stay contradictions (D-044, Elias). v2 must be frozen and pinned with five fingerprints before Aeverum's step 12, and step-12 results never steer training here. The Ollama server-log check happens in M2 Step 7.

**What to do next, in order:**
1. **Check M2-04 on disk.** Run the tests on the Mac. Compare the coded reward table with D-040 and D-046. Rescore a sample of pool rewards. Check the pool counts and the cost in `runs/RUNLOG.md`.
2. **Have the always-wrong pairs judged blind,** with `reviews/blind-brief.md`. They may have wrong labels. Fix or drop any that are wrong, and record it in a review file.
3. **Write a Codex review prompt for the reward function.** Codex runs in the ChatGPT desktop app, new session, in the folder `~/code/model-training`, and writes its review to `reviews/M2-04-codex-review.md`. Then write a fix prompt if it finds real problems, check the fixes, and merge.
4. **Write prompt M2-05: GRPO training (Style A only, D-040).**
   - A short trial run, then the full run, starting from v1's adapter with Unsloth on Modal.
   - The rule for picking the best checkpoint (a saved copy of the model partway through training), written before any result, on `valid-hard.jsonl`.
   - Watch the reward-hacking counts in every log.
   - Check current Unsloth and TRL GRPO settings against their docs.
   - The full run will likely cost over $1, so the prompt states the estimate and asks Elias first (D-019).
   - Codex reviews the training code before the full run.

**How to work** (also in the project instructions):
- **Language:** plain enough for a child to follow. Define new terms.
- **Length:** concise, with no summary at the end.
- **Git:** run it through osascript on the Mac, never from the sandbox. When Claude Code has a branch checked out, commit control files to `main` with `.git/cowork-commit.sh` (temporary index; run it with `/bin/bash`). See LESSONS 2026-10-27.
- **Every prompt for Elias:** a file in `prompts/` and a file card. Say where it goes, new or same session, the folder, the branch picker, and the worktree box.
- **Blind reviews:** separate agents, labels hidden, `reviews/blind-brief.md`.
- **Questions:** ask only what is Elias's to decide, in one batch in chat, never in pop-up boxes.
- **No em dashes** anywhere public. **Nothing important lives only in chat.**
