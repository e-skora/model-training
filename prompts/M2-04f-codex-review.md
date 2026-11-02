# Prompt M2-04f review: Codex checks the fixes to its M2-04d review

**Where:** Codex (ChatGPT desktop app), **new session**, working directly in the folder `~/code/model-training` on your Mac.
**Branch:** `m2-04-pool-and-reward` (the folder is already on it; do not switch).

Copy everything below the line into Codex.

---

You are reviewing code in `~/code/model-training`, which is on branch `m2-04-pool-and-reward`. Read `AGENTS.md` first. **This is a read-only review:**
- Do not switch branches.
- Do not edit, commit or delete anything.
- Do not start any GPU or Modal job, and do not use the network.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`.
- The only file you write is `reviews/M2-04f-codex-review.md`.

Context: your review `reviews/M2-04d-codex-review.md` said CHANGES REQUIRED, with three must-fix findings about the reward fingerprint (`judge.reward.reward_sha256()`): (1) `runpy` loaded project code without being followed or stopped; (2) a folder beside `judge` with no `__init__.py` escaped; (3) a nested module already loaded was missed. There was one nice-to-have: (4) the note on the 46 old-code failures. Claude in Cowork chose to replace the import tracing with the whole `src/` folder plus two stops (`DECISIONS.md` D-049). Elias allowed exactly `re.compile` (D-050). Claude Code added two more stops of its own (D-051). Claude Code did this in the commits after `dd96f9c`. What changed and which test covers each: the "Third review" section of `reviews/M2-04-fixes.md`. The offline recheck: `results/m2/pool/reward-recheck.md`.

Review `git diff dd96f9c..HEAD` and check:
1. **Finding 1:** retry your reproductions, in temporary folders only: `runpy.run_module("judge.policy")`, `runpy.run_path(...)` and `from runpy import run_module`. Does each one now stop with a clear error naming the file and line?
2. **Finding 2:** retry your reproduction: `policy/reader.py` beside `judge` with no `__init__.py`, used by `from policy.reader import read_fields` and by `import policy.reader`. Does changing it now move the fingerprint?
3. **Finding 3:** retry your reproduction: `judge/sub/policy.py`, loaded first by another import, then called as `judge.sub.policy.read_fields(...)` inside the scorer. Does changing it now move the fingerprint?
4. **Stop 1:** does it stop on `importlib`, `runpy`, `__import__`, `exec`, `eval`, `compile`, `builtins` and `__builtins__`, as a name, an attribute, `import x` and `from x import y`, and on any change to `sys.path`? Do words inside strings and comments stay out of it? Is `re.compile` the only `compile` allowed, or can another `compile` pass as `re.compile`?
5. **Stop 2:** does it stop on an import of a name that is a `.py` file or a folder in the repo root, `scripts/` or `tests/`? Is any stop wrong, or will one get in the way of real work?
6. **The two added stops (D-051):** a link under `src/`, and `judge` not in a folder called `src`. Are they right? Could either give a wrong answer without an error?
7. **Try to break it:** try to change the points without moving the fingerprint and without tripping a stop, using a way that `reviews/M2-04-fixes.md` does not list as not covered. Does it read every file under `src/`, in an order that does not depend on the computer? Can two different folders give the same value? Is the list of skipped files exactly `__pycache__` folders and `.pyc` files? Is any plain way of loading code missing from the stop list?
8. **The claims:** do `GLOSSARY.md` ("Reward fingerprint"), `STATE.md`, `reviews/M2-04-fixes.md` and the docstring of `reward_sha256()` say exactly what is covered and what is not? `STATE.md`, the older sections of `reviews/M2-04-fixes.md` and `results/m2/pool/reward-recheck.md` keep their old entries as history; check that the newer text clearly replaces them. Is the list of what is not covered complete and honest?
9. **The tests:** do the new tests in `tests/test_reward.py` test the real thing? The fix note says 96 of 195 fail on the code at `0d39055`, 22 of those only because the old error names no line, and three tests that cannot fail there were each shown to fail on a scratch copy broken on purpose. Is that right? Were the removed tests right to go? Do any tests touch the real `src/judge` files, the live environment or the network?
10. **The recheck:** run the script in `results/m2/pool/reward-recheck.md` and confirm its numbers, including the final fingerprint `1ffce9909b4cef446921fd0ecd0185eebef4b881b1d3c1fb39648edad42c2381`.
11. **Nothing else changed:** no reward value, saved sample, pool row, always-wrong row or summary differs from `dd96f9c`. `results/m2/pool/v1-samples.jsonl`, its `.meta.json`, `data/m2/rl-pool.jsonl`, `data/m2/rl-pool-always-wrong.jsonl` and `results/m2/pool/summary.md` keep their bytes. The only file under `src/` that changed is `src/judge/reward.py`.
12. **Rerun:** run `uv run pytest` and report the count.

Write `reviews/M2-04f-codex-review.md` in plain language a non-engineer can follow:
- a verdict line at the top: `ACCEPT` or `CHANGES REQUIRED`
- then each finding, with the file and line, what is wrong, why it matters, and the fix
- mark each finding `must fix` or `nice to have`
- if you find nothing, say so
