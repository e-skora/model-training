# M2-05: fixes after Codex's review

Codex's review `reviews/M2-05-codex-review.md` (at `2460e46`, 2026-11-08) said CHANGES REQUIRED: five must-fix findings and one nice-to-have. Claude Code fixed all six on branch `m2-05-grpo`. No GPU or Modal job ran for these fixes. No file under `src/`, no saved pool file and no saved trial result changed. The reward fingerprint is still `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`.

**Tests:** 613 pass (565 before). `tests/test_train_grpo.py` has 90 cases (42 before).

**Proof that the new tests test the real thing.** Most new tests check functions that did not exist at `2460e46`, so failing there would prove nothing (LESSONS 2026-10-30). Instead, each guard was broken on purpose in the script, and its test was run against the broken copy. The script was then restored byte for byte. All 12 broke as they should:

| Guard turned off | Test that then fails |
|---|---|
| Style A gate | `test_answers_not_in_v1s_format_disqualify` |
| one answer per prompt copy | `test_other_sampling_stops_at_once` |
| word filters | `test_any_word_filter_stops` |
| 8 copies of one prompt per group | `test_a_group_must_be_eight_copies_of_one_pool_prompt` |
| a call that skips a check | `test_a_call_that_skips_any_check_stops_at_the_step_end` |
| finished-run check before choose | `test_choose_refuses_an_unfinished_or_unchecked_run` |
| every candidate answered every item | `test_choose_writes_nothing_when_answers_are_missing_or_from_other_files` |
| the choice rechecked before upload | `test_upload_refuses_a_choice_the_counts_no_longer_make` |
| validation fingerprints | `test_choose_refuses_changed_validation_files_and_a_changed_reward` |
| a fresh run name | `test_every_run_gets_a_fresh_name` |
| a missing step metric | `test_a_logged_step_missing_a_watch_metric_stops` |
| each pair exactly twice | `test_coverage_in_a_full_run_is_each_pair_exactly_twice` |

For finding 5, collecting the tests with `--noconftest` and an unreadable home `.modal.toml` fails with `modal.exception.InvalidError`. With the new `tests/conftest.py` it passes.

## Finding 1 (must fix): checkpoint folders could be reused, and the upload could differ from the choice

- **Fresh folders.** Every trial or training run gets a new name: the stage, the UTC time and a random tag (`new_run_id`, for example `train-20261108T171500Z-3f9a1c`). The `--run` option is gone, so `train --run trial` cannot happen. The GPU job makes the run's folder with `exist_ok=False` before it writes anything, and each checkpoint folder the same way. So a retry or a second checkout can never write into another run's folder. A run that crashes leaves its folder. The next try gets a new one.
- **Adapter fingerprints.** Every saved checkpoint records the SHA-256 of each of its files (`saved[].files`). The training record also holds the SHA-256 of v1's `adapter_config.json` and `adapter_model.safetensors` at the pinned commit (`adapter_check.v1_files`), for step 0.
- **Checked before use.** `answer` refuses to load a candidate whose files differ from the recorded fingerprints. The Mac checks the fingerprints each answer came back with before it scores anything (`check_outcomes`). `upload` checks the folder again before it sends it.
- **The remote repo is protected.** `upload` refuses when the repo already exists, unless it already holds this same adapter file. That is the case of a finished upload whose local record was lost, and it is recorded as `recovered`. It never replaces another model. It records the commit the upload itself returned (`commit.oid`), not a separate lookup.
- **Records are written whole.** A new record is written to a scratch file and given its name in one step, which fails if the name exists (`write_new_json`). An update replaces the record in one step (`replace_json`).
- **Also, from "What checked out":** the trainable weights must now match v1's file by exact name set, not only contain `lora_`, so a partial freeze would stop the run.
- **Tests:** `test_every_run_gets_a_fresh_name`, `test_files_sha256_fingerprints_each_plain_file`, `test_records_are_written_whole_and_a_new_one_never_replaces_an_old_one`, `test_choose_sends_each_candidates_fingerprints`, `test_choose_writes_nothing_when_answers_are_missing_or_from_other_files` (other files), `test_upload_sends_the_chosen_fingerprints_and_records_its_own_commit`, `test_lfs_sha256_reads_both_listing_shapes`.

## Finding 2 (must fix): choose and upload accepted an incomplete or stale record

- **A finished, checked full run only** (`check_finished_run`). The record must say:
  - stage `train`, 314 pool rows, 314 planned and done steps;
  - exactly the checkpoints 50, 100, 150, 200, 250, 300 and 314, each with file fingerprints;
  - fingerprints of v1's files for step 0;
  - a step log of 314 steps and one checked generation per step;
  - and that the run's own checks passed (`run_checks.passed`).

  A trial record, a partial run or a run whose checks failed stops before any remote call.
- **Recomputed, not trusted:** the reward fingerprint is recomputed from today's source before choose and upload. Both validation files must match the fingerprints recorded when training started (`valid_kept_apart`), and, at upload, the ones the choice used.
- **Complete answers before anything is written:** every planned candidate must come back exactly once, with exactly one answer for each of the 588 items, from its fingerprinted files. Otherwise nothing is written and a clean retry works. The answer files go into a scratch folder that is renamed to `valid/` only after every check passed.
- **Before upload:** the recorded counts must still pick the recorded winner by D-056. The winner must be one of the recorded checkpoints, not step 0.
- **Tests:** `test_choose_refuses_an_unfinished_or_unchecked_run` (6 cases), `test_choose_refuses_a_trial_record`, `test_choose_refuses_changed_validation_files_and_a_changed_reward`, `test_choose_writes_nothing_when_answers_are_missing_or_from_other_files`, `test_check_outcomes_wants_each_candidate_once`, `test_upload_refuses_a_choice_the_counts_no_longer_make`, `test_upload_refuses_step_zero_and_a_missing_choice`.

## Finding 3 (must fix): the sampling check could pass after the answer generation changed

The checks now live in `GenerationWatch`, a class with no GPU code, so tests feed it fake generations. The GPU job feeds it from five places:
1. **`generate` itself**, wrapped before the model loads, so Unsloth's own saved copy of `generate` is the wrapped one;
2. **the settings `generate` ends with**, which must be the pool's, with exactly one answer per prompt copy (`num_return_sequences` 1);
3. **the word filters `generate` adds**, an allow-list of none: any filter stops, including one that bans words;
4. **the stop rules it adds**, an allow-list of the 256-token limit and the end token: a stop string or a time limit stops;
5. **TRL's generation step** (`_generate_single_turn`), which sees the prompts and answers. Each group must be 8 copies of one pool prompt, token for token, with exactly one answer per copy.

After every step, all five must have seen one call per step, and the reward must have scored exactly as many answers as were written (`check_counts`). So a call that skips any recording point stops the run at the end of that step. A logged step that lacks a watch metric stops the run (`check_step_log`), and so does a step that was never logged. After the run, every one of the 314 pool pairs must have had exactly 2 groups, one per epoch, before the record says its checks passed.

**Tests:** `test_the_watch_passes_the_pools_sampling_and_counts_groups`, `test_other_sampling_stops_at_once` (9 cases), `test_any_word_filter_stops`, `test_other_stop_rules_stop` (3 cases), `test_a_group_must_be_eight_copies_of_one_pool_prompt`, `test_a_call_that_skips_any_check_stops_at_the_step_end` (5 cases, one per recording point), `test_rewarded_answers_must_equal_written_ones`, `test_coverage_in_a_full_run_is_each_pair_exactly_twice`, `test_a_logged_step_missing_a_watch_metric_stops`.

**Not tested on a GPU.** The new wrappers have not run on Modal. The trial's record shows the settings they check (`GOOD_CONFIG` in the tests is what the trial's 20 calls used). If a wrapper does not see a call on the real stack, `check_counts` stops the run at the end of step 1, after about 3 minutes and about $0.05.

## Finding 4 (must fix): a model could win with answers that break v1's format

The choice now also puts out any candidate with an answer that the reward's own reader (`judge.reward.read_answer`) rejects, in either validation file and any version. That covers invalid answers plus a missing or wrong `type` or no `reason`. Each checkpoint's counts gain `style_a_breaks`. The scorer's published counts are unchanged. Recorded as D-057 (Default), **waiting for Claude in Cowork to confirm**, as Codex asked.

**Tests:** `test_answers_not_in_v1s_format_disqualify`, `test_counts_come_from_the_scorer_and_the_rewards_reader`, and in `test_choose_answers_every_candidate_and_applies_the_rule` a fake checkpoint that answers with the label only is put out.

## Finding 5 (must fix): importing the test script could read the real Modal config

New `tests/conftest.py`. pytest loads it before any test module. It points `MODAL_CONFIG_PATH` at an empty file in a temporary folder, removes every `MODAL_*` and `HF_*` setting from the test process, and points Hugging Face's cache at the same temporary folder. Modal reads its config path once, when it is imported (`modal/config.py` line 121), so no test can read `~/.modal.toml` now. This covers the older tests that import Modal scripts too.

**Tests:** `test_modal_reads_an_empty_temporary_config_not_the_real_one`, and `test_collecting_the_tests_never_reads_a_home_modal_config`. The second collects two test files in a fresh Python whose home folder holds an unreadable `.modal.toml`. It passes with the conftest and fails without it (`--noconftest`).

## Finding 6 (nice to have): the cost and record details

- `results/m2/grpo/full-run-plan.md`: training is now about 4,255 s (71 L4 minutes), about **$1.02**, likely $0.95 to $1.10, still over $1. That is Codex's arithmetic: 12.65 s a plain step and 7.75 s more for each of 7 saves. The old $1.08 counted the saves twice. Memory is described as trial headroom.
- New records say the save cadence really used (`save_every_used`) and the GPU seconds of each stage (`gpu_seconds`, and `choice.gpu_seconds`). The upload records its own Modal app id, and `bill` adds up train, choose and upload separately.
- `trial.meta.json` is a saved result and stays as it is (R-0049). Its `save_every: 50` is the full-run setting. The trial saved at 10 and 20, as its `save_points` shows. Its `cost_estimate_inside` counts the training job only; with the load check the trial used 628.9 GPU seconds (10.5 minutes). Its bill, $0.18, covers both.

## Also from "What checked out"

- Ties listed in reverse order still go to the earlier step (`test_a_recall_tie_goes_to_higher_precision_then_the_earlier_step`).
- Exactly 0.03 below step 0 on M1 stays in, for precision and for recall, and anything lower is out (`test_the_m1_limits_at_exactly_0_03`, 4 cases).

## Known limits (D-053)

- The fingerprints and checks catch ordinary edits, retries and model outputs. Code written on purpose to fake a record, or to replace the reward or `generate` in memory, can still pass them.
- If choose crashes after naming the `valid/` folder but before writing the record, the next choose stops because `valid/` exists. Moving that folder aside makes it retryable. It is never overwritten.
