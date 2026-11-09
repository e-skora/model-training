# M2-05c: fixes after Codex's third review

Codex's third review `reviews/M2-05c-codex-review.md` (at `7be1643`, 2026-11-09) said CHANGES REQUIRED: three must-fix findings. Every fix from the two earlier reviews held. Claude Code fixed all three on branch `m2-05-grpo` on 2026-11-09. No GPU or Modal job ran. No file under `src/`, no saved pool file and no saved trial result changed (`git diff 2460e46..HEAD --stat` shows none). The reward fingerprint is still `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`.

Elias decided on 2026-11-09 that the next Codex review checks only these three fixes (D-058).

**Tests:** 649 pass (644 before). `tests/test_train_grpo.py` has 126 cases (121 before).

**Proof that the new tests test the real thing.** Each new guard was broken on purpose and its tests were run against the broken copy; the script was then restored byte for byte (SHA-256 checked). All 5 were caught:

| Guard turned off | Test that then fails |
|---|---|
| padding only on the left | `test_the_mask_must_be_one_unbroken_run_per_row`, `test_the_real_wrappers_stop_a_changed_step` |
| the bill stays out of the uploaded record | `test_recording_the_bill_between_two_upload_tries_changes_nothing` (both cases) |
| the repo is still at the checked commit before sending | `test_another_writer_between_the_check_and_the_commit_is_never_overwritten` |
| the commit names the checked commit as its parent | `test_another_writer_between_the_check_and_the_commit_is_never_overwritten` |
| private before sending, also on a new repo | `test_a_repo_that_is_public_after_it_was_made_gets_nothing` |

## Finding 1 (must fix): the generation check accepted right padding

`unpadded_rows` now also needs each row's real tokens to end at the row's last position, so padding can only be on the left, as in the pool (`src/judge/generate.py`). Right padding stops the run at once. TRL 1.15.0's text path pads on the left, so a correct run passes.

**Tests:** a right-padded case in `test_the_mask_must_be_one_unbroken_run_per_row`, and a right-padding case through the real wrappers with prompts of two lengths in `test_the_real_wrappers_stop_a_changed_step`.

## Finding 2 (must fix): recording the bill could strand an upload retry

The uploaded record is now `upload_payload(meta)`: the run and choice record without the two fields `bill` adds (`billed_usd`, `billed_total_usd`). Those fields stay in the local record. So a bill recorded between two tries of one upload does not change the record text the retry must find on Hugging Face, and nothing else changes the record between tries (`upload` refuses a record that already holds an upload). This is simpler than saving a separate copy of the payload, which Codex offered as one way: the text is rebuilt the same way every time.

**Tests:** `test_recording_the_bill_between_two_upload_tries_changes_nothing`, in two cases, through `run_stage` and the real `upload_checked` with the fake Hugging Face: a first try that stops after making the repo (the retry resumes), and a first try whose upload lands but whose local record is lost (the retry recovers). In both, `bill` runs between the tries, and the two tries send the same text.

## Finding 3 (must fix): the upload could overwrite a file another writer added

- Just before sending, the job looks again: the repo must be private and still at the commit it checked (on a new repo, its first commit). Otherwise it stops and sends nothing.
- The upload names that commit as its parent (`parent_commit`). In the pinned `huggingface_hub` 1.33.0 the commit then fails if the repo has moved on (`hf_api.py`, `create_commit` docs at lines 5173 to 5177, passed through by `upload_folder`). So a writer who lands in the last moment is not overwritten either.
- Privacy is now checked before sending on every path, including a repo just made.
- The message for an empty repo the script cannot prove it made now says to check who made it and what its history holds before deleting anything.

**Tests:** `test_another_writer_between_the_check_and_the_commit_is_never_overwritten` (a writer during the copy, and one just before the commit), `test_a_repo_that_is_public_after_it_was_made_gets_nothing`.

**Not tested live.** The fake Hugging Face refuses a commit whose parent is not the newest one, as the pinned library's documentation says the real one does. No live write was made.

## Unchanged

- The new checks have still not run on a GPU. A mismatch stops the run during step 1 (about $0.05).
- Known limits (D-053): code written on purpose to fake records, or to replace the reward or `generate` in memory, can still pass the checks.
