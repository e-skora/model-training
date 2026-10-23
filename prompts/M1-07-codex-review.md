# Prompt M1-07 review: Codex checks the training and scoring code

**Where:** Codex (ChatGPT desktop app), **new session**, working directly in the folder `~/code/model-training` on your Mac.
**Branch:** `m1-07-train` (the folder is already on it; do not switch).

Copy everything below the line into Codex.

---

You are reviewing code in `~/code/model-training`, which is on branch `m1-07-train`. Read `AGENTS.md` first. **This is a read-only review:**
- Do not switch branches.
- Do not edit, commit or delete anything.
- Do not start any GPU or Modal job, and do not use the network.
- The only file you write is `reviews/M1-07-codex-review.md`.

Context: Step 7 fine-tuned `Qwen3-4B-Instruct-2507` with LoRA (Unsloth, on Modal) to judge whether two memories contradict each other, then scored it once on the frozen test set. The results (`results/m1/trained-v1/summary.md`) say it beats both untrained baselines. Before anyone publishes that claim, check that the training and the scoring were honest and correct. The earlier scoring code was already reviewed (`reviews/M1-05-codex-review.md`, fixed in `reviews/M1-05-fixes.md`).

Review these files:
- `src/judge/train_data.py`
- `src/judge/load.py`
- `scripts/train.py`
- `scripts/eval_trained.py`
- `tests/test_train_data.py`
- every other change on this branch since `main` (`git diff main...m1-07-train`)

Check especially:
1. **Learning only from the answer:** is the loss computed only on the assistant's JSON answer and the end token, never on the prompt? Does the pre-training check really prove that for every row?
2. **The "thinking" block:** the model's chat template adds an empty thinking block. How is it handled in training and in scoring? Does the scorer now accept anything the Step 5 and Step 6 baselines would have counted as invalid? If so, that comparison is unfair. Say exactly what changed.
3. **No leakage:**
   - The test set must be read only in the final scoring run, after its SHA-256 check.
   - The epoch must be chosen only from validation results, by the rule written in code before the results existed.
   - No test pair, person or near copy may reach training.
4. **Same conditions as the baseline:** the trained model must be scored with the same prompt (same SHA-256), the same greedy settings, the same answer loop and the same three test versions as `results/m1/baseline-v2/`. Do the base-model files match what the baseline really loaded? Weigh the "wrong base files on record" note in the summary.
5. **Records:** do `train.meta.json` and the scoring `.meta.json` record the data fingerprints, the settings, the seed, the base commit and the adapter commit? Could someone repeat the result from them?
6. **Secrets and files:** is the Hugging Face token never printed or saved? Are no model files in git? Is the upload going to a private repo?
7. **Rerun locally:** run `uv run pytest`, and rescore the saved answer files in `results/m1/trained-v1/` with `src/judge/score.py`. Confirm the numbers in `summary.md`.

Write `reviews/M1-07-codex-review.md` in plain language a non-engineer can follow:
- a verdict line at the top: `ACCEPT` or `CHANGES REQUIRED`
- then each finding, with the file and line, what is wrong, why it matters, and the fix
- mark each finding `must fix` or `nice to have`
- if you find nothing, say so
