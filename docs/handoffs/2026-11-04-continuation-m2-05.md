# MODEL-TRAINING: CONTINUATION: Claude Code is running M2-05 (GRPO code, tests, trial run) · 2026-11-04

**Where:** the Cowork project "Fine Tuning and RL training", **new chat**. `~/code/model-training` must be connected.

Copy everything below the line into that chat. Nothing else is needed.

---

This is a new session of the "Fine Tuning and RL training" project. You are continuing M2: improving the contradiction judge with reinforcement learning (GRPO). Read state from files and git, not from any earlier chat. The working rules are in the project instructions, `AGENTS.md` and `LESSONS.md`; do not restate them.

**Why we're here:** v1 catches 134 of 150 contradictions on the hard test set at its worst (no dates), mostly missing them by dodging with `unclear`. M2's bar is 139 (D-041). GRPO training is how we try to get there, and M2-05 is its first real GPU step.

**Where we are (check it against the files):**
- `main` is at `ee3ce32` (or later). M2-04 is merged: the reward function, the RL pool (314 pairs, SHA-256 `2c5c3bf2...dac4e77`) and the reward fingerprint `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`, after five Codex reviews (the last ACCEPT). 523 tests passed on the Mac at `088aca0`.
- I pasted prompt `prompts/M2-05-grpo-trial.md` into Claude Code on 2026-11-04. Do not ask me to paste it or its report. At 17:30 that day its branch `m2-05-grpo` was local only, with no commits yet.
- Approved (D-055): writing M2-05 and its trial run, each Modal run under $1. **Not approved:** the full training run. It waits for Codex's review of the training code and my yes to the cost in `results/m2/grpo/full-run-plan.md`.
- The checkpoint rule is fixed before any result (D-056). The fingerprint only has to catch honest changes; tricks written to fool it are known limits (D-053).

**First action:** read `STATE.md`, then check the branch `m2-05-grpo` on disk and on GitHub. If Claude Code has not finished (no "Step 5 part 1 written" line in that branch's `STATE.md`), say so in one line and wait. If it has, check its work on the Mac: run the tests, compare the settings and the sampling the trial really used with M2-05's Part B rules and the pool's, rescore a few trial rewards, check the cost in `runs/RUNLOG.md`, and check that no file under `src/` and no saved pool file changed. Then check `prompts/M2-05-codex-review.md` before I paste it, and tell me in one to three plain sentences where things stand.

**Watch-outs:**
- Claude Code may still be working in the checkout. Do not touch its files; commit control files to `main` with `.git/cowork-commit-files.sh <files>` (LESSONS 2026-10-27). Once its branch is idle, commit control files on that branch instead (LESSONS 2026-11-04).
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`.

**Read before acting:** `STATE.md`; `prompts/M2-05-grpo-trial.md`; `DECISIONS.md` D-053 to D-056. **Read if needed:** `milestones/M2-rl-judge/SPEC.md`, `results/m2/pool/summary.md`, `reviews/M2-04-fixes.md`.
