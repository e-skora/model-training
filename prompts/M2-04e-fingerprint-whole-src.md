# Prompt M2-04e: The reward fingerprint covers the whole `src/` folder

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `m2-04-pool-and-reward`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`, on the branch `m2-04-pool-and-reward`. Read `AGENTS.md` first and follow it. Then read:
- `STATE.md`;
- `DECISIONS.md` D-048 and D-049 (D-049 replaces D-048's import walk);
- `reviews/M2-04d-codex-review.md` (Codex's third review: three must-fix findings and one nice-to-have);
- `reviews/M2-04-fixes.md` and `results/m2/pool/reward-recheck.md`;
- `src/judge/reward.py` and `tests/test_reward.py`.

This task fixes the findings of `reviews/M2-04d-codex-review.md` by changing what the reward fingerprint (`judge.reward.reward_sha256()`) covers. Rules for this task:
- No GPU or Modal run, and no network use.
- No change to any reward value or rule.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`.
- Do not change any saved result: `results/m2/pool/v1-samples.jsonl`, its `.meta.json`, `data/m2/rl-pool.jsonl`, `data/m2/rl-pool-always-wrong.jsonl` and `results/m2/pool/summary.md` keep their bytes. Check this with `git diff` before you finish.
- Step 5 (training) is not approved. Do not start it.

## What to build (D-049)

1. **Cover the whole `src/` folder.** `reward_sha256()` becomes the SHA-256 of every file under `src/`: each file's path relative to `src/`, then its bytes, in sorted order. Skip only `__pycache__` folders and `.pyc` files. Remove the import walk (`imported_modules` and its helpers) if nothing else needs it. Keep `fingerprinted_files()` (or a function like it) so a test and the recheck can list what is covered.
2. **Stop 1, loading code by name.** If any file under `src/` uses `importlib`, `runpy`, `__import__`, `exec`, `eval`, `compile` or `builtins` (as a name, an attribute, `import x` or `from x import y`), or changes `sys.path`, stop with a clear error naming the file and line. Read the code with Python's `ast` module. Words inside strings and comments do not count, so the stop list itself in `reward.py` must not trigger it.
3. **Stop 2, project code outside `src/`.** If any file under `src/` imports a top-level name that matches a `.py` file or a folder in the repo root, `scripts/` or `tests/`, stop with a clear error.
4. **Today's code must pass both stops.** If a stop blocks a file that is there today (for example `load.py` or `modal_image.py`), stop and report it instead of loosening the rule.
5. **Docstring:** say exactly what is covered, the two stops, and what is not covered (Python, outside packages, data and settings files, changes in memory while running, wrappers outside `src/`).

## Tests (in `tests/test_reward.py`)

Build every test in a temporary copy of `src/` (`tmp_path`), run in a fresh Python, never by editing the real files. Each new test must fail on today's code (commit `0d39055`) and pass on the new code; run each against today's code first and record the result.
- Codex finding 1: `runpy.run_module("judge.policy")` and `runpy.run_path(...)` each stop with an error. Also `from runpy import run_module`.
- Codex finding 2: a folder `src/policy/` with no `__init__.py`, used by `from policy.reader import read_fields` and by `import policy.reader`: changing `policy/reader.py` moves the fingerprint.
- Codex finding 3: `judge/sub/policy.py`, loaded first by another import, then called as `judge.sub.policy.read_fields(...)` inside the scorer, so the points change: changing that file moves the fingerprint. Show the points change too, so the test proves the helper is used.
- Any new file anywhere under `src/` moves the fingerprint; a change inside `__pycache__` does not.
- `sys.path.insert(...)` in a file under `src/` stops with an error.
- An import of a name that matches a folder in `scripts/` or the repo root stops with an error.
- The fingerprint is the same when run twice and from another folder.

Keep the existing tests that still apply. Rewrite or remove the ones that only test the import walk, and say which.

## Offline recheck

- In `results/m2/pool/reward-recheck.md`, add the new fingerprint as a new row. Keep the earlier rows as they are. Rerun its check with today's code: all 8,144 answers must still rescore to the 323 stored reward lists, and the pool SHA-256 must stay `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77`.
- Step 5 records this final fingerprint.

## Close

- `reviews/M2-04d-codex-review.md` is already committed by Claude in Cowork; do not change it.
- Add a section for the third review to `reviews/M2-04-fixes.md`: each finding, what changed, which test covers it. Also fix Codex's nice-to-have (finding 4): in the second-review section, say that 44 tests failed on the old code for the reasons listed, and 2 failed only because they test new behaviour (`test_fingerprint_reads_files_not_memory`, `test_fingerprint_covers_these_files_today`). Add a line; do not rewrite the earlier text.
- Make `GLOSSARY.md` ("Reward fingerprint") and `STATE.md` say exactly what is covered and what is not.
- Add a lesson to `LESSONS.md`.
- Write the Codex re-review prompt `prompts/M2-04f-codex-review.md` in the same shape as `prompts/M2-04d-codex-review.md`: read-only, new session, folder `~/code/model-training`, branch `m2-04-pool-and-reward`, writes only `reviews/M2-04f-codex-review.md`. It must ask Codex to retry all three of its reproductions, check both stops, and try to change points without moving the fingerprint or tripping a stop.
- Update `STATE.md`'s next action: Elias pastes `prompts/M2-04f-codex-review.md` into Codex.
- Run `uv run pytest` on the Mac. Commit in small steps and push the branch. Do not merge.

**Report back** in plain language:
- the final fingerprint and how many files it covers;
- what each new test changes, and that it failed on today's code;
- any file today's code had to change to pass the stops;
- the recheck result;
- the number of tests that pass;
- confirmation that no saved result file changed;
- anything that is Elias's to do (expected: paste the Codex prompt).
