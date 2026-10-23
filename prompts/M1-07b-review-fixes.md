# Prompt M1-07b: Fix the Codex review findings for Step 7

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `m1-07-train`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training` on branch `m1-07-train`. Read `AGENTS.md` first and follow it, then `STATE.md` and the review `reviews/M1-07-codex-review.md`. This task fixes the review's findings before Step 7 merges.

**Rules for this task**
- **No GPU or Modal runs, and no new training or test-set run.**
- Keep every saved answer file and score exactly as it is.
- Do not start Step 8.

**Fix these (finding numbers are from the review)**
1. **A smoke run can overwrite final results.**
   - Smoke runs get their own default names, or they require a separate empty folder.
   - Before any remote work starts, `scripts/eval_trained.py` refuses to replace an existing answer or settings file.
   - Add offline tests for both.
2. **A new choice can leave the upload pointing at the old choice.**
   - When `choose` replaces the choice, drop the old upload record, unless it provably belongs to the same choice and run.
   - Final scoring stops before loading the test set or calling Modal if `choice.chosen` and `upload.checkpoint` differ.
   - Add offline tests for this sequence.
3. **Selection doesn't check fingerprints.**
   - Before selecting, check the validation file's SHA-256 and the prompt SHA-256 against `train.meta.json`, and stop on any mismatch. Check the saved run settings the same way, instead of quietly using today's script constants.
   - Compute the training-data fingerprints from the same bytes that get loaded, before sending them to Modal.
   - Add offline tests showing that a mismatch stops before any remote call.
4. **The claim that the baseline used identical base files isn't verified.** Gather the evidence that exists without a GPU:
   - Use the Hugging Face API to read the commit history of `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`. Was `7744afa8566e264af1a92a806d8d9aae00cc7c78` the latest commit on 2026-10-22, when the baselines ran?
   - Check the pinned Unsloth 2026.10.7 source to see how it maps `unsloth/Qwen3-4B-Instruct-2507` to that 4-bit copy.
   - Then rewrite `results/m1/trained-v1/summary.md` line 78 so it says exactly what the evidence proves. If it is still an inference, say so plainly. Fix the test comment in `tests/test_train_data.py` so it doesn't claim to prove the weights are identical.
5. **Wording, nice to have.** In `summary.md`, say that compared with the untrained 8B, recall is lower both as is (0.95 vs 0.96) and without dates (0.90 vs 0.96).

**Close**
- Commit `reviews/M1-07-codex-review.md` as it is.
- Add `reviews/M1-07-fixes.md`, listing each finding, what changed, and the test that covers it.
- Update `STATE.md`: review fixed; next action is that Claude in Cowork checks and merges Step 7.
- Add any lesson to `LESSONS.md`.
- Run `uv run pytest` until it passes.
- Commit in small steps and push the branch. Do not merge.

**Report back** in plain language: what changed for each finding, the test count, what the base-file evidence showed, and confirmation that no saved answer or score changed.
