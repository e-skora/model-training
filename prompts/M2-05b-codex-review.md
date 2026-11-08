# Prompt M2-05b review: Codex checks the fixes to its M2-05 review

**Where:** Codex (ChatGPT desktop app), **new session**, working directly in the folder `~/code/model-training` on your Mac.
**Branch:** `m2-05-grpo` (the folder is already on it; do not switch).

Copy everything below the line into Codex.

---

You are reviewing code in `~/code/model-training`, which is on branch `m2-05-grpo`. Read `AGENTS.md` first. **This is a read-only review:**
- Do not switch branches.
- Do not edit, commit or delete anything.
- Do not start any GPU or Modal job, and do not use the network.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`, and never run a model on them.
- The only file you write is `reviews/M2-05b-codex-review.md`.

Context: your review `reviews/M2-05-codex-review.md` (at `2460e46`) said CHANGES REQUIRED, with five must-fix findings about the GRPO training code `scripts/train_grpo.py`, and one nice-to-have:
1. checkpoint folders could be reused and the upload could differ from the choice;
2. choose and upload accepted an incomplete or stale record;
3. the sampling check could pass after the answer generation changed;
4. a model could win with answers that break v1's format;
5. importing the test script could read the real Modal config;
6. the cost and record details.

Claude Code fixed them in the commits after `2460e46`. What changed and which test covers each: `reviews/M2-05-fixes.md`. The format gate in the choice is recorded as D-057 in `DECISIONS.md`, waiting for Claude in Cowork to confirm. The corrected cost: `results/m2/grpo/full-run-plan.md`. No GPU job ran for the fixes. The saved trial (`results/m2/grpo/trial.meta.json`) is from the code at `2460e46` and was not rerun.

**How to rate what you find (D-053, as in your last review).**
- A problem that ordinary, honest code or an ordinary run could cause by accident is `must fix`.
- A problem that needs code written on purpose to fool a check is a `known limit`. List it, but it does not block the full run.
- Anything else worth doing is `nice to have`.
- Say for each finding why you rated it that way.

Review `git diff 2460e46..HEAD` and check:
1. **Finding 1.** Retry your probes in temporary folders only:
   - a training run under a reserved or reused name;
   - a retry after an interruption;
   - a second checkout;
   - an upload to a repo that already exists.

   Can any run write into another run's folder now? Do choose and upload check each adapter's file fingerprints before they use it? Does the upload refuse to replace another model, and record the commit its own upload returned? Is the "recovered" case, an earlier finished upload of the same adapter file, safe? Are local records written whole?
2. **Finding 2.** Retry your probe: an incomplete record with only step 50, no sampling evidence and a changed validation fingerprint, and a changed source fingerprint. Does choose now stop before any remote call? Does a candidate with missing or repeated answers stop the choice before anything is written? Can a failed choose be retried? Does upload recheck that the recorded counts still pick the recorded winner?
3. **Finding 3.** Retry your probe of the recording functions: two answers per prompt copy, and a filter that bans words. Does each now stop? The checks now sit at five places: `generate`, its settings, its word filters, its stop rules, and TRL's `_generate_single_turn`.
   - Can a generate call or a group of answers get past all of them?
   - Does a call that skips one recording point stop the run at the end of that step (`check_counts`)?
   - Is the full-run rule "every pool pair exactly 2 groups" right for TRL 1.15.0's sampler with 314 pairs, 2 pairs a step and 314 steps? Judge from the code you can read and say what you could not check.
   - Does a missing step metric stop the run?
4. **Finding 4.** Retry your probe: a candidate that answers with its label only. Is it out now? Is D-057's rule the same as the training reward's idea of a valid answer (`judge.reward.read_answer`)? Did the scorer's published counts stay as they were?
5. **Finding 5.** Does `tests/conftest.py` keep every test away from `~/.modal.toml`, the Modal and Hugging Face settings and the real Hugging Face cache, before any test imports Modal? Rerun your guarded test run with the real config path blocked.
6. **Finding 6.** Is the corrected cost in `full-run-plan.md` right (about 71 L4 minutes, about $1.02, still over $1)? Do the new records keep each stage's GPU seconds, the save cadence used and the upload's app id?
7. **The tests.** `reviews/M2-05-fixes.md` says each of 12 guards was broken on purpose and its test then failed, and that most new tests check functions that did not exist at `2460e46`. Is that right? Do any new tests touch the live environment, the network or a frozen test file?
8. **New problems.** Did the fixes break anything that worked at `2460e46`? Look first at the wrappers in the GPU job, which have not run on a GPU. Could one of them stop a correct run, for example by counting a generate call twice or by missing TRL's real call path? How soon would that show, and what would it cost?
9. **Nothing else changed.** No file under `src/` changed, so the reward fingerprint is still `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`. The saved pool files and `results/m2/pool/` keep their bytes, and so does `results/m2/grpo/trial.meta.json` (`git diff 2460e46..HEAD --stat`).
10. **Rerun:** run `uv run pytest` and report the count. Leave out the same tests as last time (the 13 cases that read `data/m1/test.jsonl` and the 12 in `tests/test_run_ollama.py` that start a local server) and say what you left out.

Write `reviews/M2-05b-codex-review.md` in plain language a non-engineer can follow:
- a verdict line at the top: `ACCEPT` or `CHANGES REQUIRED`. Known limits alone do not make it `CHANGES REQUIRED`.
- then each finding, with the file and line, what is wrong, why it matters, and the fix
- mark each finding `must fix`, `nice to have` or `known limit`
- if you find nothing, say so
