# Prompt M2-05e review: Codex checks the one fix to its M2-05d review (narrow review, D-058)

**Where:** Codex (ChatGPT desktop app), **new session**, working directly in the folder `~/code/model-training` on your Mac.
**Branch:** `m2-05-grpo` (the folder is already on it; do not switch).

Copy everything below the line into Codex.

---

You are reviewing code in `~/code/model-training`, which is on branch `m2-05-grpo`. Read `AGENTS.md` first. **This is a read-only review:**
- Do not switch branches.
- Do not edit, commit or delete anything.
- Do not start any GPU or Modal job, and do not use the network.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`, and never run a model on them.
- The only file you write is `reviews/M2-05e-codex-review.md`.

**Scope (D-058 in `DECISIONS.md`, Elias's decision).** This review checks only the one fix to your M2-05d review, and whether it broke anything. Do not review the rest of the branch again. If you happen to see something outside this scope that is clearly `must fix` under D-053, list it under a separate heading "Outside the scope"; it does not decide the verdict.

Context: your review `reviews/M2-05d-codex-review.md` (at `cfd104d`) said CHANGES REQUIRED with one must-fix finding: on the fresh-repo path, a writer who lands after the repo is made and before the first lookup could become the upload's parent commit, and the upload could overwrite that writer's file. The fix and its test: `reviews/M2-05d-fixes.md`.

**How to rate what you find (D-053).** A problem that ordinary, honest code or an ordinary run could cause by accident is `must fix`. A problem that needs code written on purpose to fool a check is a `known limit`; it does not block. Anything else is `nice to have`. Say why for each.

Review `git diff cfd104d..HEAD` and check:
1. Retry your probe: another writer adds `adapter_config.json` right after the repo is made, before the first lookup. Does the upload now stop, send nothing and leave the file intact? Can a retry resume into that repo?
2. Is the commit the new check looks at the same one the pre-send check and `parent_commit` use?
3. Can the new check stop a correct upload, for example because of the `.gitattributes` file Hugging Face adds?
4. Is the known limit in `reviews/M2-05d-fixes.md` (a writer who changes only `.gitattributes`) rated correctly?
5. Does the new test fail with the check turned off? Does it touch the live environment, the network or a frozen test file?
6. No file under `src/` changed (reward fingerprint `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`), and the saved pool files, `results/m2/pool/` and `results/m2/grpo/trial.meta.json` keep their bytes (`git diff 2460e46..HEAD --stat`).
7. Run `uv run pytest` and report the count. Leave out the same 25 tests as last time and say what you left out.

Write `reviews/M2-05e-codex-review.md` in plain language a non-engineer can follow:
- a verdict line at the top: `ACCEPT` or `CHANGES REQUIRED`, decided by this fix only. Known limits alone do not make it `CHANGES REQUIRED`.
- then each finding, with the file and line, what is wrong, why it matters, and the fix
- mark each finding `must fix`, `nice to have` or `known limit`
- if you find nothing, say so
