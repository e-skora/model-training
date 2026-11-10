# Prompt M2-05d review: Codex checks the three fixes to its M2-05c review (narrow review, D-058)

**Where:** Codex (ChatGPT desktop app), **new session**, working directly in the folder `~/code/model-training` on your Mac.
**Branch:** `m2-05-grpo` (the folder is already on it; do not switch).

Copy everything below the line into Codex.

---

You are reviewing code in `~/code/model-training`, which is on branch `m2-05-grpo`. Read `AGENTS.md` first. **This is a read-only review:**
- Do not switch branches.
- Do not edit, commit or delete anything.
- Do not start any GPU or Modal job, and do not use the network.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`, and never run a model on them.
- The only file you write is `reviews/M2-05d-codex-review.md`.

**Scope (D-058 in `DECISIONS.md`, Elias's decision).** This review checks only the three fixes to your M2-05c review, and whether they broke anything. Do not review the rest of the branch again; your three earlier reviews covered it, and their fixes held. If you happen to see something outside this scope that is clearly `must fix` under D-053, list it under a separate heading "Outside the scope"; it does not decide the verdict.

Context: your review `reviews/M2-05c-codex-review.md` (at `7be1643`) said CHANGES REQUIRED with three must-fix findings in `scripts/train_grpo.py`:
1. the generation check accepted right padding;
2. recording the bill could strand an upload retry;
3. the upload could overwrite a file another writer added after its check.

Claude Code fixed them in the commits after `d447e0b`. What changed and which test covers each: `reviews/M2-05c-fixes.md`.

**How to rate what you find (D-053).** A problem that ordinary, honest code or an ordinary run could cause by accident is `must fix`. A problem that needs code written on purpose to fool a check is a `known limit`; it does not block. Anything else is `nice to have`. Say why for each.

Review `git diff d447e0b..HEAD` and check:
1. **Finding 1.** Retry your probe: the real wrappers with a fake trainer that pads on the right. Does it stop now? Can the new rule stop a correct, left-padded run?
2. **Finding 2.** Retry your two probes: a bill recorded between two tries of one upload, once after the repo was made and the upload failed, once after the upload landed and the local record was lost. Does the retry now resume or recover? Can leaving the two bill fields out of the uploaded record ever let a different run or choice pass?
3. **Finding 3.** Retry your probe: another writer adds a file between the check and the commit. Is it refused now, both during the copy and in the last moment before the commit (`parent_commit`)? Is the repo checked as private before sending on every path? Judge `parent_commit` from the pinned `huggingface_hub` 1.33.0 source if you can read it, and say what you could not check.
4. **The tests.** `reviews/M2-05c-fixes.md` says each of 5 new guards was broken on purpose and its test then failed. Is that right? Do any new tests touch the live environment, the network or a frozen test file?
5. **Nothing else changed.** No file under `src/` changed (reward fingerprint `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`), and the saved pool files, `results/m2/pool/` and `results/m2/grpo/trial.meta.json` keep their bytes (`git diff 2460e46..HEAD --stat`).
6. **Rerun:** run `uv run pytest` and report the count. Leave out the same 25 tests as last time and say what you left out.

Write `reviews/M2-05d-codex-review.md` in plain language a non-engineer can follow:
- a verdict line at the top: `ACCEPT` or `CHANGES REQUIRED`, decided by the three fixes only. Known limits alone do not make it `CHANGES REQUIRED`.
- then each finding, with the file and line, what is wrong, why it matters, and the fix
- mark each finding `must fix`, `nice to have` or `known limit`
- if you find nothing, say so
