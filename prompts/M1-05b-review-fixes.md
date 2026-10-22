# Prompt M1-05b: Fix the Codex review findings

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `m1-05-baseline`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training` on branch `m1-05-baseline`. Read `AGENTS.md` first and follow it, then `STATE.md`, `DECISIONS.md` (D-021 to D-024) and the review `reviews/M1-05-codex-review.md`. This task fixes the review's findings before Step 5 merges. **No GPU or Modal runs.** Do not start any other step.

**The scoring rule is settled (D-024): do not change what precision and recall mean.**
- Precision and recall stay standard. That is the definition Aeverum's bar uses (02 §8.4).
- An invalid answer is neither a "yes" nor a "no". It lowers recall when the right answer was contradiction, and it lowers accuracy, but it does not enter precision. In Aeverum, an unreadable answer becomes "needs manual review", not a false alarm.
- Because of that, the invalid-answer count is always reported next to precision and recall, and the training goal requires 0 invalid answers.
- Fix the wording wherever it says invalid answers "count as wrong" in every score. Today that is `results/m1/baseline/summary.md` and the column notes. Say exactly what they lower.

**Fix these (finding numbers are from the review)**
1. **Finding 1:** keep the standard measures and the current baseline numbers. Correct the wording as above. Add tests that mix correct contradiction answers, false alarms, invalid answers whose right label is contradiction, and invalid answers whose right label is non-contradiction, and check each measure.
2. **Finding 2:** add one shared frozen-file loader. It checks the SHA-256 against the "Frozen files" entry in `STATE.md` and stops before any scoring or model run if they differ. Use it in both the scorer command and `scripts/run_baseline.py`. Keep the pure `score()` function usable with made-up examples. Add a temporary-file test showing that a changed byte stops scoring.
3. **Finding 3:** when reading an answer, reject `NaN`, `Infinity` and `-Infinity`, and reject repeated field names. Add tests for both.
4. **Finding 4:** add a separate "format errors" count: answers with a valid label that still break the required fields (`type` must be JSON `null` for contradictions and one of the three types otherwise, and `reason` must be non-empty text). Report it as its own column. It does not change label accuracy.
5. **Finding 5:** reject duplicate and unknown answer IDs with a clear message. Missing answers still count as wrong. Add tests.
6. **Finding 6:** in `scripts/run_baseline.py`, set `num_beams=1` explicitly. Save the generation settings, chat-template options, model and tokenizer revisions, and a SHA-256 of the first rendered prompt in each run's `.meta.json` from now on. Add an offline test with fake model and tokenizer objects that checks these settings are passed through. Do not rerun the GPU job. In the summary, say the saved baseline runs predate this metadata.
7. **Rescore without the GPU.** Rerun the scorer on the three saved answer files and update `results/m1/baseline/summary.md`, including the new "format errors" column. Precision, recall and accuracy must come out the same as before. If any of them changes, stop and explain why.
8. **Close.** Commit `reviews/M1-05-codex-review.md` as it is. Add a short `reviews/M1-05-fixes.md` that lists each finding, what you changed, and the test that covers it. Update `STATE.md` (review fixed; next action: Claude in Cowork merges Step 5). Add any lesson to `LESSONS.md`. Run `uv run pytest` until it passes. Commit in small steps, push the branch, do not merge.

**Report back** in plain language: what changed for each finding, the test count, the rescored table, and confirmation that precision, recall and accuracy did not change.
