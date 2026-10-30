# Prompt M2-04d review: Codex checks the fixes to its M2-04b review

**Where:** Codex (ChatGPT desktop app), **new session**, working directly in the folder `~/code/model-training` on your Mac.
**Branch:** `m2-04-pool-and-reward` (the folder is already on it; do not switch).

Copy everything below the line into Codex.

---

You are reviewing code in `~/code/model-training`, which is on branch `m2-04-pool-and-reward`. Read `AGENTS.md` first. **This is a read-only review:**
- Do not switch branches.
- Do not edit, commit or delete anything.
- Do not start any GPU or Modal job, and do not use the network.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`.
- The only file you write is `reviews/M2-04d-codex-review.md`.

Context: your review `reviews/M2-04b-codex-review.md` said CHANGES REQUIRED, with two must-fix findings about the reward fingerprint (`judge.reward.reward_sha256()`): (1) default values, captured values and helpers reached through a module could change points without moving it; (2) `GRPOReward.__call__` was outside it. Claude in Cowork chose to replace the name walk with whole source files (`DECISIONS.md` D-048). Claude Code did this in the commits after `4b2d9d6`. What changed and which test covers each: the "Second review" section of `reviews/M2-04-fixes.md`. The offline recheck: `results/m2/pool/reward-recheck.md`.

Review `git diff 4b2d9d6..HEAD` and check:
1. **Finding 1:** retry your reproductions, in temporary folders only: the `STRICT = True` default value, the captured `allow` flag, and `policy.read_fields(items)` in another module. Does each one now move the fingerprint?
2. **Finding 2:** retry your reproduction: change `GRPOReward.__call__` to double its points, with `reward` and `completion_text` unchanged. Does it move the fingerprint?
3. **Try to break it:** try to change the points without moving the fingerprint, and without using a way that `reviews/M2-04-fixes.md` lists as not covered. Do `fingerprinted_files()` and `imported_modules()` find every `judge` module (`import judge.x`, `from judge import x`, `from judge.x import y`, relative imports, imports inside a function, subfolders)? Can they loop forever, skip a file silently, or depend on the folder or the computer? Does `judge/__init__.py` belong in the list?
4. **The stops:** does the fingerprint stop with a clear error for `importlib`, `builtins`, `__import__`, `exec`, `eval`, `import judge` alone, a star import from `judge`, an import with no source file, and project code beside `judge`? Is any stop wrong, or will one get in the way of real work? Is any plain way of loading code missing?
5. **The claims:** do `GLOSSARY.md` ("Reward fingerprint"), `STATE.md`, `reviews/M2-04-fixes.md` and the docstring of `reward_sha256()` say exactly what is covered and what is not? `STATE.md` and the older table in `reviews/M2-04-fixes.md` keep their old entries as history; check that the newer text clearly replaces them. Is the list of what is not covered complete and honest, including that a change made in memory no longer shows?
6. **The tests:** do the new tests in `tests/test_reward.py` test the real thing? The fix note says 46 fail on the old `reward.py`, the stability test cannot fail there, and 14 rewritten older tests pass on both. Is that right? Do any tests touch the real `src/judge` files, the live environment or the network?
7. **The recheck:** run the script in `results/m2/pool/reward-recheck.md` and confirm its numbers, including the final fingerprint `4d64ca4c54a353e3cb60522cfd49ed5dd38a14ade5ec55f8e87a98ee15f8a42a`.
8. **Nothing else changed:** no reward value, saved sample, pool row, always-wrong row or summary differs from `4b2d9d6`. `results/m2/pool/v1-samples.jsonl`, its `.meta.json`, `data/m2/rl-pool.jsonl`, `data/m2/rl-pool-always-wrong.jsonl` and `results/m2/pool/summary.md` keep their bytes.
9. **Rerun:** run `uv run pytest` and report the count.

Write `reviews/M2-04d-codex-review.md` in plain language a non-engineer can follow:
- a verdict line at the top: `ACCEPT` or `CHANGES REQUIRED`
- then each finding, with the file and line, what is wrong, why it matters, and the fix
- mark each finding `must fix` or `nice to have`
- if you find nothing, say so
