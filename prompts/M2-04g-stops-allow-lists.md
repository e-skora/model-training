# Prompt M2-04g: The fingerprint stops become allow-lists

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `m2-04-pool-and-reward`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`, on the branch `m2-04-pool-and-reward`. Read `AGENTS.md` first and follow it. Then read:
- `STATE.md`;
- `DECISIONS.md` D-049 to D-052 (D-052 is new and sets this task's rules);
- `reviews/M2-04f-codex-review.md` (Codex's fourth review: three must-fix findings and one nice-to-have);
- `reviews/M2-04-fixes.md` and `results/m2/pool/reward-recheck.md`;
- `src/judge/reward.py` and `tests/test_reward.py`.

Rules for this task:
- No GPU or Modal run, and no network use.
- No change to any reward value or rule.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`.
- Do not change any saved result: `results/m2/pool/v1-samples.jsonl`, its `.meta.json`, `data/m2/rl-pool.jsonl`, `data/m2/rl-pool-always-wrong.jsonl` and `results/m2/pool/summary.md` keep their bytes. Check this with `git diff` before you finish.
- Do not change any file under `src/` other than `src/judge/reward.py`, so the old-method check still gives `a369e730...`.
- Step 5 (training) is not approved. Do not start it.

## What to build (D-052): say what may pass, stop everything else

1. **Finding 1, `sys`.** A file under `src/` may only `import sys` plainly (no alias, no `from sys import ...`) and use `sys.argv`, `sys.exit`, `sys.stdout` and `sys.stderr`. Any other use of the name `sys` stops: another attribute (including `path`, `meta_path`, `path_hooks`, `modules`), `runtime = sys`, passing `sys` to a function, or binding the name `sys` in any other way. Today's uses are `sys.exit` and `sys.argv` in `format.py` and `score.py`; they must pass.
2. **Finding 1, related: `__path__`.** Any use of `__path__` in a file under `src/` stops.
3. **Finding 2, links.** Before anything is resolved, check the path from the repo folder to `src/`, `src/judge` and `src/judge/reward.py` for links, part by part, using the path Python imported (`__file__`, not `resolve()`). A link anywhere on it stops. Keep the check for links found inside `src/`.
4. **Finding 3, `re.compile` (D-050).** `re.compile` passes only when, in that file, the name `re` is bound exactly once, by a plain top-level `import re`, and the file has no star import (`from x import *`). Any other binding of `re` stops, of any kind: assignment, augmented or annotated assignment, `for`, `with ... as`, `except ... as`, walrus (`:=`), `match` patterns (`case re:`, `case [*re]`, `case {**re}`, `case X() as re`), `global` and `nonlocal`, function arguments, `def`, `class`, `import ... as re`, `del re`. Use Python's `symtable` module or a full `ast` check; say which. Today's `score.py` and `frozen.py` must pass.
5. **Finding 4, `__pycache__`.** Skip `__pycache__` only when it is a folder, and skip what is inside it. A file named `__pycache__` is covered.
6. Update the docstring and `GLOSSARY.md` ("Reward fingerprint") to say each allow-list.

## Tests (in `tests/test_reward.py`)

Build every test in a temporary copy of `src/`, run in a fresh Python, never by editing the real files. Each new test must fail on today's code (commit `61616b6`) and pass on the new code; run each against today's code first and record the result.
- Codex's `runtime = sys; runtime.path.append(...)` case with an outside helper in `plugins/policy.py`: it stops with a file-and-line error before a fingerprint is returned. Also `sys.meta_path`, `sys.modules`, and passing `sys` to a function.
- Codex's `__path__.append(...)` case in `judge/__init__.py`: it stops.
- Codex's linked-`judge` layout (`strict/src/judge` and `loose/src/judge` both linking to `external/src/judge`, whose parent is also named `src`): it stops. The test must reach the fingerprint's own check, not stop earlier in a test helper.
- Codex's `match Worker(): case re:` example and the star-import example: each stops. Add one test per binding kind in point 4.
- A regular file named `src/__pycache__` moves the fingerprint; a change inside a `__pycache__` folder does not.
- `score.py` and `frozen.py` as they are today pass both stops.

## Offline recheck

- In `results/m2/pool/reward-recheck.md`, add the new fingerprint as a new row; keep the earlier rows. Rerun the check: all 8,144 answers must still rescore to the 323 stored reward lists, the old method must still give `a369e730...`, and the pool SHA-256 must stay `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77`.

## Close

- `reviews/M2-04f-codex-review.md` is already committed by Claude in Cowork; do not change it.
- Add a section for the fourth review to `reviews/M2-04-fixes.md`: each finding, what changed, which test covers it.
- Make `STATE.md` and `GLOSSARY.md` say exactly what passes and what stops.
- Add a lesson to `LESSONS.md`.
- Write the Codex re-review prompt `prompts/M2-04h-codex-review.md` in the same shape as `prompts/M2-04f-codex-review.md`: read-only, new session, folder `~/code/model-training`, branch `m2-04-pool-and-reward`, writes only `reviews/M2-04h-codex-review.md`. It must ask Codex to retry all its fourth-review reproductions and to try to get past each allow-list.
- Update `STATE.md`'s next action: Claude in Cowork checks `prompts/M2-04h-codex-review.md`, then Elias pastes it into Codex.
- Run `uv run pytest` on the Mac. Commit in small steps and push the branch. Do not merge.

**Report back** in plain language:
- the final fingerprint;
- what each new test changes, and that it failed on today's code;
- the recheck result;
- the number of tests that pass;
- confirmation that no saved result file and no `src/` file other than `reward.py` changed;
- anything that is Elias's to do (expected: nothing until Claude in Cowork checks the Codex prompt).
