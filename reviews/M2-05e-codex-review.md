ACCEPT

Reviewed `m2-05-grpo` at `f8fc51caa341962ee00fd9784ed69fdc2c523f05` against `cfd104d` on 2026-11-09. This review covers the one fresh-repo upload fix under D-058. I found no must-fix issue in that change.

## The previous finding is fixed

- **Another writer's file is preserved.** I reran the early-writer probe with the real upload helper and the fake Hugging Face service. Another writer added `adapter_config.json` just after repo creation, before the first lookup. The helper stopped before sending files, left the other writer's bytes intact, and left the reservation without a first commit. A retry with that reservation also stopped before sending. See `scripts/train_grpo.py:1182-1199`, `:1202-1210` and `tests/test_train_grpo.py:1179`.
- **The checks use one commit.** A commit identifies one saved version of the repo. The fresh path checks the files at the first commit it observes. It records that exact ID only if the file check passes. The pre-send check compares the current commit to that ID, and the upload sends the same ID as its expected parent. I confirmed the file-list lookup, saved reservation and upload parent matched in the fake. See `scripts/train_grpo.py:1205-1210` and `:1216-1223`.
- **A normal new repo still passes.** The file check ignores the service's `.gitattributes`, which is a file containing Git settings. A fake repo containing just that file passed, and the upload preserved it. I did not test a live repo or the GPU stack. See `scripts/train_grpo.py:1120` and `:1151-1155`.

## Known limit. A change only to the ignored Git settings file can pass

I confirmed that a writer who changes only `.gitattributes` before the first lookup does not trip this check. The fake upload left that writer's file unchanged. That edit could be ordinary. The limit is nonblocking because this check protects model files from replacement, and the staged upload does not include that settings file. A later change to the repo's commit is still covered by the pre-send and parent checks. The fixes note is right to record this as a limit of what the file check detects. No fix is needed within this review's scope. See `scripts/train_grpo.py:1120`, `:1154-1155` and `:1216-1223`.

## Test evidence

The guarded offline `uv run` rerun passed **625 tests**, including all 127 GRPO cases, with **25 deselected**. The guard recorded no attempts in the test process to open protected files, write inside the checkout or use the network. The new test uses a fake service and temporary files. I separately disabled just the new first-commit check in memory. The regression test then failed because the expected refusal never happened. The original source passed.

- Seven saved-answer cases in `tests/test_score.py::test_m1_saved_answers_rescore_to_the_same_numbers`.
- Six publishing checks in `tests/test_prepublish_check.py`. They are `test_real_cards_pass`, `test_wrong_score_in_a_table_is_caught`, `test_unknown_row_in_a_score_table_is_caught`, `test_wrong_count_is_caught`, `test_untraced_number_in_prose_is_caught` and `test_every_fingerprint_in_the_cards_is_known`.
- Twelve cases in `tests/test_run_ollama.py` that start a local server.

Against `2460e46`, Git shows no change under `src/`, in the saved pool files or results, or in `trial.meta.json`. The recomputed reward fingerprint is `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`. Both frozen files have the same stored Git IDs and their contents were never opened. The review-range whitespace check passed.

## Done and next action

1. **Codex finished** the narrow review in `/Users/frank/code/model-training`. This report is the only checkout change.
2. **Claude in Cowork** should record the acceptance and confirm D-057, the answer-format rule for choosing a checkpoint, in the **Fine Tuning and RL training** project with `/Users/frank/code/model-training` connected.
3. **Elias** can then approve the estimated **$1.02** full run in that Cowork project before Claude Code starts training.
