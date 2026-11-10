# M2-05b: fixes after Codex's re-review

Codex's re-review `reviews/M2-05b-codex-review.md` (at `6c72da5`, 2026-11-08) said CHANGES REQUIRED: three must-fix findings and one nice-to-have. Claude Code fixed all four on branch `m2-05-grpo` on 2026-11-09. No GPU or Modal job ran. No file under `src/`, no saved pool file and no saved trial result changed (`git diff 2460e46..HEAD --stat` shows none). The reward fingerprint is still `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`.

**Tests:** 644 pass (613 before). `tests/test_train_grpo.py` has 121 cases (90 before).

**Proof that the new tests test the real thing.** Each new guard was broken on purpose in the script, and its tests were run against the broken copy. The script was then restored byte for byte (its SHA-256 checked). All 15 were caught:

| Guard turned off | Test that then fails |
|---|---|
| `generate` gets exactly TRL's prompt tokens | `test_generate_must_get_exactly_trls_prompts`, `test_the_real_wrappers_stop_a_changed_step` |
| the pool's two end tokens | `test_other_stop_rules_stop`, `test_the_real_wrappers_stop_a_changed_step` |
| the length limit is the padded width plus 256 | `test_other_stop_rules_stop`, `test_the_real_wrappers_stop_a_changed_step` |
| the mask is one unbroken run per row | `test_the_mask_must_be_one_unbroken_run_per_row` |
| one `generate` call per TRL step | `test_the_mask_must_be_one_unbroken_run_per_row`, `test_the_real_wrappers_stop_a_changed_step` |
| no answer over 256 tokens | `test_a_group_must_be_eight_copies_of_one_pool_prompt` |
| the wrapper hands `generate`'s real input to the watch | `test_the_real_wrappers_pass_a_correct_step_and_come_back_out` |
| recovery needs every file and the record | `test_a_repo_that_differs_in_any_file_is_refused` |
| resuming needs this run's note | `test_an_upload_that_failed_after_making_the_repo_resumes_into_it` |
| the copy to upload is checked | `test_the_upload_refuses_a_public_repo_a_changed_copy_and_a_short_commit` |
| the commit the upload returned is checked | `test_the_upload_refuses_a_public_repo_a_changed_copy_and_a_short_commit` |
| saved answer files keep their bytes | `test_saved_answers_that_do_not_check_out_are_never_used_or_overwritten` |
| saved answers still give the saved choice | `test_saved_answers_that_no_longer_give_the_saved_choice_stop` |
| the saved choice is this run's | `test_saved_answers_that_do_not_check_out_are_never_used_or_overwritten` |
| a retry reuses the saved answers | `test_a_choose_stopped_after_its_answers_were_saved_finishes_without_answering_again` |

## Finding 1 (must fix): the sampling watch did not check the tokens that enter the model

- **The wrappers are now plain functions** (`wrap_generation`, `wrap_trainer_turn`), outside the GPU job. The GPU job calls them, and the tests drive the same code with a fake `generate` shaped like transformers 5.17.0 and a fake trainer step shaped like TRL 1.15.0's text path (left-pad, `generate(input_ids=..., attention_mask=...)`, cut the prompt off).
- **Exact prompt tokens.** `generate`'s wrapper hands the watch its real `input_ids` and `attention_mask`. The watch keeps each row's tokens under the mask (which must be one unbroken run). When TRL's step returns, those rows must equal TRL's prompt rows exactly, in order. A shortened, extended, reordered or dropped row stops the run at once. So does a second `generate` call inside one TRL step, or a step with none.
- **The length limit and the end tokens.** The stop-rule wrapper now reads each rule's values, not only its class name. The length limit must be the padded prompt width plus 256, as transformers sets it. The end tokens must be exactly the pool's, 151645 and 151643. A missing end-token rule, another end token or another limit stops the run. Where these numbers come from: `results/m2/grpo/settings.md`, "Tightened after Codex's re-review".
- **Also:** no answer may be longer than 256 tokens.
- **Kept:** the group and coverage checks, the counts after every step, and the allow-lists.
- **Tests:** `test_generate_must_get_exactly_trls_prompts` (3 cases), `test_the_mask_must_be_one_unbroken_run_per_row`, `test_other_stop_rules_stop` (6 cases), `test_the_real_wrappers_pass_a_correct_step_and_come_back_out`, `test_the_real_wrappers_stop_a_changed_step` (8 cases: shortened prompts, reordered rows, two calls, one end token, another end token, no end-token rule, another length limit, other sampling), `test_stop_rule_reads_tensors_lists_and_single_ids`.

**Not tested on a GPU.** The fakes follow the pinned source; they are not the real libraries. If the real call differs, the run stops during step 1, after about 4 minutes and about $0.05.

## Finding 2 (must fix): upload recovery could accept a different model and strand an interrupted upload

The upload job's logic is now a plain function, `upload_checked`, with Hugging Face passed in. The Modal job only connects it to the real client and the volume. The tests run the real logic with a fake Hugging Face.

- **The exact files.** The repo must end up holding exactly the checked adapter folder (every file, by SHA-256) and the run and choice record `train.meta.json` (its SHA-256 from the text sent). The only other file allowed is the `.gitattributes` Hugging Face adds by itself.
- **Recovery is proven at one commit.** If the repo already exists, it must be private. Every file at its newest commit is downloaded at that commit and fingerprinted. Recovery is recorded only when they are exactly the wanted files. Matching weights with another `adapter_config.json`, another record or an extra file are refused.
- **An interrupted upload can resume.** Before making the repo, the job saves a note on the volume next to the run (`<run>/hf-upload.json`): this run, this checkpoint, the wanted files. After making the repo, it adds the repo's first commit to the note. A retry resumes into an existing repo only if it is empty and the note names that same first commit, so this try made it and nothing came after. Any other existing repo is refused.
- **Checked before and after sending.** The copy made for the upload is fingerprinted before it is sent. The commit the upload returns is fingerprinted again and must hold exactly the wanted files.
- **Tests:** `test_a_fresh_upload_reserves_the_repo_and_records_its_own_commit`, `test_an_upload_that_failed_after_making_the_repo_resumes_into_it`, `test_a_finished_upload_whose_local_record_was_lost_is_recovered`, `test_a_repo_that_differs_in_any_file_is_refused` (3 cases), `test_the_upload_refuses_a_public_repo_a_changed_copy_and_a_short_commit`, `test_the_upload_stops_before_hugging_face_when_the_folder_changed`.

**Known limit.** If the job stops in the moment between making the repo and noting its first commit, the retry finds an empty repo it cannot prove it made. It stops and says to delete that empty repo on Hugging Face and retry. That window is one network reply long.

## Finding 3 (must fix): an interrupted choice could force a second paid evaluation

- **The whole choice is saved with the answers.** Choose writes every answer file and a `choice.json` into a scratch folder, then renames it to `valid/` in one move. `choice.json` holds the run, every candidate's file fingerprints, the reward fingerprint, a fingerprint of the item list, each answer file's SHA-256, and the whole choice record (counts, scores, the choice, the validation files' fingerprints, the worker runs and their app id). Only then is `train.meta.json` updated.
- **A retry finishes from the saved answers.** If `valid/` exists and the record has no choice, choose checks the folder instead of answering again: the same run, checkpoint fingerprints, validation files, reward and items; exactly the listed files, each with its saved bytes; every item once per candidate. It then scores the saved answers again, and they must give the saved counts, scores and choice. Then it writes the record, marked `recovered_from: valid/choice.json`. No candidate answers again.
- **Anything else stops.** A `valid/` that fails any check is never used and never overwritten.
- **Tests:** `test_a_choose_stopped_after_its_answers_were_saved_finishes_without_answering_again` (the record write fails after the rename; the retry dispatches nothing and gives the same choice as a clean run), `test_saved_answers_that_do_not_check_out_are_never_used_or_overwritten` (5 cases), `test_saved_answers_that_no_longer_give_the_saved_choice_stop`.

## Nice to have: when each check fires, and what 305 steps means

`results/m2/grpo/full-run-plan.md` now has "When each check stops the run": the Mac checks cost nothing; load checks about $0.02; the answer-writing checks and the step-1 counts about $0.05; a missing step-1 log shows at step 2, about $0.06; the coverage check runs only after all 314 steps, so it costs the full run, and then choose refuses the record. The 305-step note now says it is an estimate, not a cap, and that it would leave some pool pairs with one group.

## Unchanged known limits (D-053)

Code written on purpose to fake a record, or to replace the reward or `generate` in memory, can still pass the checks.
