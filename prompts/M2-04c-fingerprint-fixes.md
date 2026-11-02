# Prompt M2-04c: The reward fingerprint covers whole source files

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `m2-04-pool-and-reward`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`, on the branch `m2-04-pool-and-reward`. Read `AGENTS.md` first and follow it. Then read:
- `STATE.md`;
- `DECISIONS.md` D-040, D-046 and D-048;
- `reviews/M2-04-codex-review.md`, `reviews/M2-04-fixes.md` and `reviews/M2-04b-codex-review.md`;
- `results/m2/pool/reward-recheck.md`;
- `src/judge/reward.py` and `tests/test_reward.py`.

This task fixes the two must-fix findings of `reviews/M2-04b-codex-review.md`. Both are about the reward fingerprint (`judge.reward.reward_sha256()`), the code that says which reward a training run used. Rules for this task:
- No GPU or Modal run.
- No change to any reward value or rule.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`.
- Do not change or rewrite any saved result: `results/m2/pool/v1-samples.jsonl`, its `.meta.json`, `data/m2/rl-pool.jsonl`, `data/m2/rl-pool-always-wrong.jsonl` and `results/m2/pool/summary.md` keep their bytes. Check this with `git diff` before you finish.

## What to build (D-048)

1. **Replace the name walk with whole source files.** `reward_sha256()` becomes the SHA-256 of the full text of `src/judge/reward.py` and of every `judge` module it imports, followed through their imports, with each file's path, in sorted order.
   - Find the imports by reading each file's code with Python's `ast` module (`import judge.x`, `from judge import x`, `from judge.x import y`, relative imports). Do not keep a hand-made list.
   - Today that should be `judge.reward`, `judge.score`, `judge.prompt` and `judge.frozen`. Say if it finds others.
   - If one of those files uses `importlib`, `__import__`, `exec` or `eval`, stop with a clear error instead of skipping it.
   - Remove `rule_parts` and `describe` if nothing else uses them.
   - Python and outside packages are not covered (D-048); say so in the docstring.
2. **This covers both findings.** Finding 1 (default values, captured values, helpers reached through a module) and finding 2 (`GRPOReward.__call__`) are in those files, so any change to them moves the fingerprint.

## Tests (add to `tests/test_reward.py`)

Each test must fail on the current code and pass on the new code. Run each against the current code first to prove it fails. Build each test in a temporary folder with made-up module files (`tmp_path`), never by editing the real `src/judge` files:
- Codex's default-value case: `STRICT = True` used as a default argument; changing only `STRICT` moves the fingerprint.
- A captured value (a closure) changing.
- A helper in another `judge` module, reached as `module.helper(...)`, changing.
- `GRPOReward.__call__` changing (for example, doubling the points) while `reward` and `completion_text` stay the same.
- A new `judge` module imported later is covered without editing a list.
- A file that uses `importlib` or `exec` stops with an error.
- The fingerprint is the same when run twice, and does not depend on the folder the tests run from.

Keep the existing fingerprint tests that still apply.

## Offline recheck

- Update `results/m2/pool/reward-recheck.md`: add the final fingerprint as a new row, keep the earlier rows as they are, and rerun its check with today's code. All 8,144 answers must still rescore to the 323 stored reward lists, and the pool SHA-256 must stay `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77`.
- Step 5 records the final fingerprint.

## Close

- Add the fixes to `reviews/M2-04-fixes.md` under a new heading for the second review: each finding, what changed, which test covers it.
- Fix the claims in `GLOSSARY.md` ("Reward fingerprint"), `STATE.md` and `reviews/M2-04-fixes.md` so they say exactly what is covered and what is not.
- Add a lesson to `LESSONS.md`.
- Write the Codex re-review prompt `prompts/M2-04d-codex-review.md` in the same shape as `prompts/M2-04b-codex-review.md`: read-only, new session, folder `~/code/model-training`, branch `m2-04-pool-and-reward`, writes only `reviews/M2-04d-codex-review.md`. It must ask Codex to retry both findings' reproductions and to try to change points without moving the fingerprint.
- Update `STATE.md`'s next action: Elias pastes `prompts/M2-04d-codex-review.md` into Codex.
- Run `uv run pytest` on the Mac. Commit in small steps and push the branch. Do not merge.

**Report back** in plain language:
- the final fingerprint and the files it covers;
- what each new test changes and that it failed on the old code;
- the recheck result;
- the number of tests that pass;
- confirmation that no saved result file changed;
- anything that is Elias's to do (expected: paste the Codex prompt).
