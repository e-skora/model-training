# Prompt M2-04h review: Codex checks the fixes to its M2-04f review

**Where:** Codex (ChatGPT desktop app), **new session**, working directly in the folder `~/code/model-training` on your Mac.
**Branch:** `m2-04-pool-and-reward` (the folder is already on it; do not switch).

Copy everything below the line into Codex.

---

You are reviewing code in `~/code/model-training`, which is on branch `m2-04-pool-and-reward`. Read `AGENTS.md` first. **This is a read-only review:**
- Do not switch branches.
- Do not edit, commit or delete anything.
- Do not start any GPU or Modal job, and do not use the network.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`.
- The only file you write is `reviews/M2-04h-codex-review.md`.

Context: your review `reviews/M2-04f-codex-review.md` said CHANGES REQUIRED, with three must-fix findings about the reward fingerprint (`judge.reward.reward_sha256()`): (1) `sys` under another name changed where imports look, and `__path__` did the same; (2) resolving links first hid a linked `judge` folder; (3) `re` could be taken by a `match` pattern or a star import. There was one nice-to-have: (4) a regular file named `__pycache__` was skipped. Claude in Cowork turned the stops into allow-lists (`DECISIONS.md` D-052). An allow-list says what may pass, and everything else stops. Claude Code settled the points D-052 left open in D-054. Claude Code did this in the commits after `61616b6`. What changed and which test covers each: the "Fourth review" section of `reviews/M2-04-fixes.md`. The offline recheck: `results/m2/pool/reward-recheck.md`.

**How to rate what you find (D-053, Elias's ruling).** The fingerprint is meant to catch honest changes: an edit to the reward code, a helper, a setting, or a file someone forgot about. It is not a defence against code written on purpose to fool it.
- A bypass that ordinary, honest code could cause by accident is `must fix`.
- A bypass that needs code written to fool the check is a `known limit`. List it, but it does not block Step 5.
- Say for each finding why you rated it that way.

Review `git diff 61616b6..HEAD` and check:
1. **Finding 1:** retry your reproductions, in temporary folders only: `runtime = sys; runtime.path.append(...)` with the helper in `plugins/policy.py`, and `__path__.append(...)` in `judge/__init__.py` with `from judge.policy import read_fields` in the scorer. Does each one now stop with a clear error naming the file and line, before a fingerprint is returned?
2. **Finding 2:** retry your reproduction: `strict/src/judge` and `loose/src/judge` both linking to `external/src/judge`, with a different `policy.py` beside each link. Does it now stop? Does the new test reach the fingerprint's own check, not stop earlier in a test helper?
3. **Finding 3:** retry your reproductions: `match Worker(): case re:` and `import re; from judge.other import *`. Does each one now stop?
4. **Finding 4:** retry your reproduction: a regular file at `src/__pycache__`. Does it now move the fingerprint? Does a change inside a `__pycache__` folder still not move it?
5. **The `sys` allow-list:** try to get past it. Can any other use of `sys` pass: another attribute (`path`, `meta_path`, `path_hooks`, `modules`), a new name for it, passing it, or binding the name another way? Are `sys.argv`, `sys.exit`, `sys.stdout` and `sys.stderr` the only things that pass, and only when read?
6. **The `re` allow-list:** try to get past it. Is any kind of binding missed? The kinds to try: assignment, augmented or annotated assignment, `for`, `with ... as`, `except ... as`, walrus, all `match` patterns, `global`, `nonlocal`, function arguments, `def`, `class`, `import ... as re`, `del re`. Claude Code used a full `ast` check, not `symtable`, and it counts every text field as a binding unless it is known not to be one. Is that sound? Is any field wrongly counted as not binding?
7. **The `__path__` stop:** can any plain use pass?
8. **The link stops:** is the check made on the path Python imported, before anything is resolved? Can a link on the way to `src`, `src/judge` or `src/judge/reward.py` pass? Is the repo folder itself rightly left out of the check? Does the check for links inside `src/` still work, including for a link called `__pycache__` or `something.pyc`?
9. **Wrong stops:** do the allow-lists stop honest code they should not? Do today's `score.py`, `frozen.py` and `format.py` pass? D-054 made some strict choices: is any of them wrong, or likely to get in the way of real work?
10. **Try to break it:** try to change the points without moving the fingerprint and without tripping a stop, using a way that the "Fourth review" section of `reviews/M2-04-fixes.md` does not list as not covered. Could ordinary, honest code do it by accident? Rate each one by D-053.
11. **The claims:** do `GLOSSARY.md` ("Reward fingerprint"), `STATE.md`, `reviews/M2-04-fixes.md` and the docstring of `reward_sha256()` say exactly what passes, what stops and what is not covered? `STATE.md`, the older sections of `reviews/M2-04-fixes.md` and `results/m2/pool/reward-recheck.md` keep their old entries as history; check that the newer text clearly replaces them.
12. **The tests:** do the new tests in `tests/test_reward.py` test the real thing? The fix note says 142 of the 341 fail on the code at `61616b6`, 113 for the real reason and 29 only because the error points at a different line, and that the 8 new tests that cannot fail there were each shown to fail on a scratch copy broken on purpose. Is that right? Were the 4 changed expected lines, and the change to `PROBE`, right? Do any tests touch the real `src/judge` files, the live environment or the network?
13. **The recheck:** run the script in `results/m2/pool/reward-recheck.md` and confirm its numbers, including the final fingerprint `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`.
14. **Nothing else changed:** no reward value, saved sample, pool row, always-wrong row or summary differs from `61616b6`. `results/m2/pool/v1-samples.jsonl`, its `.meta.json`, `data/m2/rl-pool.jsonl`, `data/m2/rl-pool-always-wrong.jsonl` and `results/m2/pool/summary.md` keep their bytes. The only file under `src/` that changed is `src/judge/reward.py`.
15. **Rerun:** run `uv run pytest` and report the count. Four older test functions read `data/m1/test.jsonl`: three in `tests/test_prepublish_check.py` and `test_m1_saved_answers_rescore_to_the_same_numbers` in `tests/test_score.py`. Leave those out, as you did last time, and say what you left out.

Write `reviews/M2-04h-codex-review.md` in plain language a non-engineer can follow:
- a verdict line at the top: `ACCEPT` or `CHANGES REQUIRED`. Known limits alone do not make it `CHANGES REQUIRED`.
- then each finding, with the file and line, what is wrong, why it matters, and the fix
- mark each finding `must fix`, `nice to have` or `known limit`
- if you find nothing, say so
