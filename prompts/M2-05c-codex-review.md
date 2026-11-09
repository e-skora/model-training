# Prompt M2-05c review: Codex checks the fixes to its M2-05b re-review

**Where:** Codex (ChatGPT desktop app), **new session**, working directly in the folder `~/code/model-training` on your Mac.
**Branch:** `m2-05-grpo` (the folder is already on it; do not switch).

Copy everything below the line into Codex.

---

You are reviewing code in `~/code/model-training`, which is on branch `m2-05-grpo`. Read `AGENTS.md` first. **This is a read-only review:**
- Do not switch branches.
- Do not edit, commit or delete anything.
- Do not start any GPU or Modal job, and do not use the network.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`, and never run a model on them.
- The only file you write is `reviews/M2-05c-codex-review.md`.

Context: your re-review `reviews/M2-05b-codex-review.md` (at `6c72da5`) said CHANGES REQUIRED, with three must-fix findings about the GRPO training code `scripts/train_grpo.py`, and one nice-to-have:
1. the sampling watch did not check the tokens that enter the model, nor the end tokens and length limit;
2. upload recovery could accept a different model, and an interrupted upload could strand the script;
3. an interruption after validation finished could force a second paid evaluation;
4. the cost plan should say when each check fires, and that 305 steps is an estimate, not a cap.

Claude Code fixed them in the commits after `6c72da5`. What changed and which test covers each: `reviews/M2-05b-fixes.md`. Where the end tokens and the length limit come from, with line numbers in the pinned TRL, Unsloth and transformers source: `results/m2/grpo/settings.md`, "Tightened after Codex's re-review". The cost plan: `results/m2/grpo/full-run-plan.md`. No GPU job ran for the fixes. D-057 (the format gate in the choice) is still waiting for Claude in Cowork; it is not part of this review.

**How to rate what you find (D-053, as in your last two reviews).**
- A problem that ordinary, honest code or an ordinary run could cause by accident is `must fix`.
- A problem that needs code written on purpose to fool a check is a `known limit`. List it, but it does not block the full run.
- Anything else worth doing is `nice to have`.
- Say for each finding why you rated it that way.

Review `git diff 6c72da5..HEAD` and check:
1. **Finding 1.** Retry your probe: the real wrappers (`wrap_generation`, `wrap_trainer_turn`) around a fake trainer that shortens each prompt on the way to `generate`, and a different or missing end token. Does each stop now? Also check:
   - Is the end-token list (151645 and 151643) really what the pool's `generate` used and what TRL 1.15.0 passes? Judge from the code and files you can read, and say what you could not check.
   - Is the length limit rule (padded prompt width plus 256) what transformers 5.17.0 builds?
   - Could any of the new checks stop a correct run? For example: a mask that is not one unbroken run on the real path, `input_ids` passed by position, a tensor shape the helpers do not read, or a second `generate` call inside one TRL step. How soon would that show, and what would it cost?
2. **Finding 2.** Retry your probes with the real `upload_checked` and a fake Hugging Face client:
   - an existing private repo with matching weights, a different `adapter_config.json` and no record;
   - a repo that was made, then the upload failed, then a retry;
   - a finished upload whose local record was lost.

   Is recovery now proven at one commit, for every file and the record? Can resuming ever upload into a repo this run did not make? Is the staged copy checked before sending, and the returned commit after? Is the remaining window (stopped between making the repo and noting its first commit) handled safely?
3. **Finding 3.** Retry your probe: the record write fails just after `valid/` is named. Does a normal retry finish without dispatching any worker, and give the same choice as a clean run? Can it ever use answers from another run, other checkpoint files, other validation files, another reward, or changed answer files? Is a `valid/` that does not check out ever used or overwritten?
4. **The nice-to-have.** Is "When each check stops the run" in `full-run-plan.md` right, and is the 305-step note honest?
5. **The tests.** `reviews/M2-05b-fixes.md` says each of 15 new guards was broken on purpose and its test then failed. Is that right? Do any new tests touch the live environment, the network or a frozen test file?
6. **New problems.** Did the fixes break anything that worked at `6c72da5`? Look first at the GPU job (`train`), whose wrappers moved into plain functions, and at the upload job, whose logic moved into `upload_checked`.
7. **Nothing else changed.** No file under `src/` changed, so the reward fingerprint is still `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`. The saved pool files and `results/m2/pool/` keep their bytes, and so does `results/m2/grpo/trial.meta.json` (`git diff 2460e46..HEAD --stat`).
8. **Rerun:** run `uv run pytest` and report the count. Leave out the same 25 tests as last time (seven in `tests/test_score.py::test_m1_saved_answers_rescore_to_the_same_numbers`, six in `tests/test_prepublish_check.py`, twelve in `tests/test_run_ollama.py`) and say what you left out.

Write `reviews/M2-05c-codex-review.md` in plain language a non-engineer can follow:
- a verdict line at the top: `ACCEPT` or `CHANGES REQUIRED`. Known limits alone do not make it `CHANGES REQUIRED`.
- then each finding, with the file and line, what is wrong, why it matters, and the fix
- mark each finding `must fix`, `nice to have` or `known limit`
- if you find nothing, say so
