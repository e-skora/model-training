CHANGES REQUIRED

Reviewed `m2-05-grpo` at `cfd104da9575510ab132def55d77dfa38fd7a992` against `d447e0b` on 2026-11-09. This review covers only the three M2-05c fixes under D-058. The padding and billing fixes pass. The upload can still overwrite another writer's file on the fresh-repo path.

## Must fix. Check the files at the first observed commit before accepting it

**Where.** `scripts/train_grpo.py:1202-1206`, `:1212-1219`. The current test at `tests/test_train_grpo.py:1156` adds another writer's file during the copy or immediately before the commit. Both happen after the first repo lookup.

**What I reproduced.** I ran the real `upload_checked` with the fake Hub. Right after creating the private repo and before the first `model_info` lookup, the fake added an `adapter_config.json` from another writer. The script saved that writer's commit as the repo's first commit. Its final check saw the same commit, and the upload named it as the parent. The upload replaced the other writer's config and reported success. No live upload ran.

**Why it blocks.** A parent commit identifies the saved version a new commit will build on. The new guard stops a writer who lands after that parent is recorded. It cannot help if the chosen parent already contains the other writer's file. An ordinary writer can land between creation and the first lookup. This is the same no-overwrite requirement as M2-05c finding 3.

**Fix.** After creating the repo, check that the first observed commit holds no files other than the allowed `.gitattributes`, before saving the reservation. Use the same commit for that file check and the existing parent check. Add a fake-Hub test that inserts `adapter_config.json` before the first lookup and confirms the file is left intact.

## Checks that passed

- **Finding 1, padding.** I reran both paths through the real generation wrappers. Unequal prompts passed with left padding. The same prompts with right padding stopped. The new rule allows rows with no padding and rows with a continuous block of real tokens ending at the last position. That fits the pinned TRL 1.15.0 text path, which adds padding on the left. No GPU run exercised the new guard. See `scripts/train_grpo.py:374-391` and the saved TRL source at `grpo_trainer.py:1769-1773`.
- **Finding 2, billing and retry.** Recording the bill between tries let an interrupted upload resume and a completed upload with a lost receipt recover. Both tries passed identical upload text. Only the two top-level billing fields are omitted. I separately probed a different run and a different chosen checkpoint with identical adapter files. Both were refused. See `scripts/train_grpo.py:1133-1148`, `:1178-1191` and `:1397`.
- **Finding 3, changes after the first lookup.** The fake-Hub tests rejected a writer during the copy and a writer immediately before the commit. The new private check runs immediately before sending on both the fresh and resumed paths. Recovery of an already completed upload checks privacy and sends nothing. The unproven-repo message now calls for checking its history before deleting anything. See `scripts/train_grpo.py:1182-1198` and `:1212-1223`.
- **Pinned Hugging Face client.** I read the local 1.33.0 package. Its `hf_api.py:5173-5179` documents that a commit fails if the branch no longer points to the supplied parent. The folder upload passes that parent through at `:6076-6112`, and `_commit_api.py:930-931` puts it in the request. The library documents that, if it splits a large folder into several commits, the supplied parent applies to the first commit. I verified the client code and tested the rule with a fake Hub. I did not verify live server behavior, privacy changes after the pre-send lookup, or whether the real checkpoint upload would use more than one commit. [Hugging Face Hub API reference](https://huggingface.co/docs/huggingface_hub/v1.33.0/en/package_reference/hf_api)
- **Saved evidence.** Against `2460e46`, Git shows no changes under `src/`, to the saved pool files or results, or to `trial.meta.json`. The recomputed reward fingerprint is `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`. Both frozen files have the same stored Git IDs at the two commits. Their contents were never opened. The review-range whitespace check passed.

## Test evidence

The guarded offline `uv run` rerun passed **624 tests**, including all 126 GRPO cases, with **25 deselected**. The test-process guard recorded zero attempts to open protected files, write inside the checkout or use the network. The new cases used fake generation, fake workers, a fake Hub and temporary files. I did not run the unrestricted suite.

- Seven saved-answer cases in `tests/test_score.py::test_m1_saved_answers_rescore_to_the_same_numbers`.
- Six publishing checks in `tests/test_prepublish_check.py`. They are `test_real_cards_pass`, `test_wrong_score_in_a_table_is_caught`, `test_unknown_row_in_a_score_table_is_caught`, `test_wrong_count_is_caught`, `test_untraced_number_in_prose_is_caught` and `test_every_fingerprint_in_the_cards_is_known`.
- Twelve cases in `tests/test_run_ollama.py` that start a local server.

I independently disabled each of the five new guards in memory, one at a time. Each change made a targeted test fail. Across those altered runs, seven cases failed and eight passed. Removing the early commit check still let the fake Hub refuse the write later, which the test correctly distinguished from stopping before sending.

## Done and next action

1. **Codex finished** the narrow offline review in `/Users/frank/code/model-training`. This report is the only checkout change.
2. **Claude Code in the desktop Code panel** should fix the fresh-repo check in `/Users/frank/code/model-training` and return the new commit for a check of that fix. D-057 remains with Claude in the **Fine Tuning and RL training** Cowork project.
3. **Elias has no decision to make now.** The full run still waits for review acceptance and separate cost approval in Cowork.
