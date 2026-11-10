CHANGES REQUIRED

Reviewed `m2-05-grpo` at `6c72da569625ad46d5427ccaf99d7a9311ab1003`, compared with `2460e46`, on 2026-11-08. A checkpoint is a saved copy of the adapter, the small set of model weights this run changes. The new checks reject the original incomplete-record and label-only answer cases. Three issues still block the full run.

## Findings

### 1. Must fix. The sampling watch no longer checks the tokens that enter the model

**Where.** `scripts/train_grpo.py:413`, `:420`, `:687`, `:692`, `:704`, `:813`. The check in `2460e46` compared `input_ids` and `attention_mask` at the generation boundary. The new wrapper checks the prompt before the trainer prepares that input. A token is a piece of text represented by a number.

**What I reproduced.** I ran the actual new wrappers around a fake trainer that shortened each prompt on the way to `generate`. The watch saw the original pool prompts, counted one call at all five points, and accepted the step. I also passed a different end token to the stopping rule. The watch checks the rule's class name only, so it accepted that change too. A missing end-token rule is accepted as long as the maximum-length rule is present.

**Why it blocks.** An ordinary edit to prompt preparation or to the end token can change what the model sees or when it stops. Every counter can still add up. The previous prompt check would have caught the shortened inputs. The saved trainer source shows the current pinned text path left-padding the original token IDs, so this is a gap in the check. The probe does not show that the trial used wrong tokens.

**Fix.** Keep the group and coverage checks. Compare the actual generation inputs after removing padding with the exact ordered token rows supplied by the trainer. Verify the effective length limit and end-token IDs against the pool policy as well. Add an offline test that drives the real wrappers with altered model inputs and an altered or missing end token.

### 2. Must fix. Upload recovery can accept a different model and can strand an interrupted upload

**Where.** `scripts/train_grpo.py:950` to `:973`. The existing test at `tests/test_train_grpo.py:714` uses a fake for the whole upload worker, so it cannot see either branch inside that worker.

**What I reproduced.** With the real upload function and a fake Hugging Face client, an existing private repo with matching weight bytes, a different `adapter_config.json`, and no training record returned `recovered: True`. The config affects how the model uses the weights. In a second probe, creating the repo succeeded and the upload then failed. Every retry refused the now-existing empty repo. A normal fresh upload did return the commit from its own upload call, even though the separate lookup reported another commit.

**Why it blocks.** A config correction, a pre-existing model or an ordinary connection failure can cause either outcome. Recovery must establish that the saved commit holds the checkpoint that passed validation.

**Fix.** For an existing repo, verify the full expected adapter files and the saved run and choice record at one specific private commit before recording recovery. Verify the staged copy before sending it. Give the script a way to resume a repo it can prove was reserved by the same upload attempt. Keep refusing an unrelated or unverified repo. Test the upload worker itself with a fake client for both failure cases and for successful recovery.

### 3. Must fix. An interrupted choice can force a second paid evaluation

**Where.** `scripts/train_grpo.py:1082`, `:1104` to `:1117`. The answer directory and the record are each written whole, but they are published separately.

**What I reproduced.** I made the record write fail just after the complete answer folder was renamed to `valid/`. All candidates had already answered. The record still had no choice, and a normal retry stopped at the existing folder. The fixes note describes moving that folder aside. Following that procedure makes the script dispatch all eight candidates again.

**Why it blocks.** An interruption during a local write is an ordinary failure. The current recovery path discards the benefit of a complete evaluation and risks paying about $0.71 again. Incomplete worker replies are a different case. Those now stop before the answer files are written, and a clean retry works.

**Fix.** Save a complete choice record with the answers in the same published folder, or make a retry verify and use the saved answers when the final record write is missing. Tie the saved answers to the run, validation files and checkpoint fingerprints. Test an interruption after the folder rename and require the retry to finish without dispatching workers.

### Nice to have. Distinguish early wrapper failures from late checks in the cost plan

**Where.** `results/m2/grpo/full-run-plan.md:35`, `:36`, `:55` and `scripts/train_grpo.py:772`, `:835` to `:847`.

The first step should expose a missed or doubled wrapper call. From the old trial, load plus the first step took about 241 seconds inside the worker, or four minutes. At the recorded rate, that is about $0.05 for the GPU alone. A completely missing first step log can be noticed at the next step, and the full-pool coverage check runs after training. Those failures can cost more. The 305-step example is also a point estimate and gives no assurance of staying below $1. Update the plan to state when each check fires and distinguish the estimate from a spending cap. The full-run estimate already clears the approval threshold, so this wording does not change the decision.

## Checks that passed

- **Run folders and adapter checks.** The command now accepts only a stage, so a caller cannot select the reserved `trial` name. Two simulated checkouts with separate output folders got different train IDs. Repeating the actual folder reservation against a temporary shared folder stopped and left its first file intact. On the normal scoring and upload paths the workers compare the expected file fingerprints with the local checkpoint before loading or uploading. The chooser also compares the fingerprints returned by each worker. The remaining upload issues are in the recovery path above.
- **Run and choice records.** The combined step-50-only record with no sampling evidence and changed validation and source fingerprints stopped before worker dispatch. Separate tests verify that changed validation and source fingerprints stop a finished record too. Missing and repeated answers stop before any final output. The upload stage recomputes the winner from recorded counts and rejects a different recorded winner. The local JSON writes publish whole files, subject to the two-file interruption above.
- **Generation checks and coverage.** Two returned answers per prompt copy, a word filter, a missed recording point, a mixed group and a missing step metric were rejected in offline tests. I traced the regular text path in the saved pinned source from the earlier tool check. TRL 1.15.0's `grpo_trainer.py:1216-1274`, `:1480-1512`, `:1721-1834` and `:2152-2156` show a batch of 16 rows, eight copies of each of two prompts, one generation shared over two accumulation steps, and a two-value return from `_generate_single_turn`. Unsloth 2026.10.7's `models/rl_replacements.py:1725-1767` keeps that shape for text, and `models/rl.py:270-294` and `models/llama.py:2807`, `:4198` show how the saved generation method is called. Transformers 5.17.0's `generation/utils.py:2606` and `:2781-2796` show one config, filter and stop preparation on the regular path. I found no evidence of a double count on that path. At 157 two-pair steps per pass through the pool, 314 steps should give two groups per pair, assuming the sampler visits every pair once per pass. The saved source shows the call to `RepeatSampler` and its settings. It does not include the sampler's implementation. The new wrappers and two-pass coverage were not exercised on a GPU. The package pins and source-check method are in [settings.md](../results/m2/grpo/settings.md).
- **Answer format.** A label-only candidate is now disqualified. The chooser calls the reward's own reader on every answer, so the D-057 gate uses the same definition of validity as training. The shared scorer still reports invalid answers and format errors separately, and D-056's ranking and limits are intact. Cowork's confirmation of D-057 is still open.
- **Cost and saved evidence.** From the original timings I get 12.647 seconds per plain step and 7.753 seconds extra per save. With seven saves, load, the first step and final overhead, the estimate is 4,255 seconds, or 70.9 L4 minutes. Applying the plan's rate and overhead gives **$1.02**, above $1. New records carry the actual save cadence, training and trial-load GPU seconds, the choice's total GPU seconds, and the upload app ID for billing. Rates are from the saved [full-run plan](../results/m2/grpo/full-run-plan.md) and its [Modal price reference](https://modal.com/pricing). No live provider behavior, prices or bill were checked.
- **Unchanged inputs.** Git shows no differences under `src/`, in the saved pool files or pool results, or in `trial.meta.json`. The recomputed reward fingerprint was `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`. The frozen files have the same stored Git IDs on both commits. Their contents were never opened. The review-range whitespace check passed.

**Known limit under D-053.** Deliberate code that falsifies run evidence or replaces the reward in memory can fool checks on files. The three blockers above were reproduced through ordinary input preparation, an existing repo or an interruption.

## Test evidence

The guarded offline `uv run` rerun passed **588 tests** with **25 deselected**, including all 90 GRPO tests. I left the config redirection to `tests/conftest.py` and blocked the real config and credential paths in the test process. The guard recorded zero denied attempts to open those paths, read either frozen file, write inside the checkout or use the network. The new subprocess collection test used a fake home folder. The unrestricted suite was not run.

- Seven saved-answer cases in `tests/test_score.py::test_m1_saved_answers_rescore_to_the_same_numbers`.
- Six publishing checks in `tests/test_prepublish_check.py`. They are `test_real_cards_pass`, `test_wrong_score_in_a_table_is_caught`, `test_unknown_row_in_a_score_table_is_caught`, `test_wrong_count_is_caught`, `test_untraced_number_in_prose_is_caught` and `test_every_fingerprint_in_the_cards_is_known`.
- Twelve cases in `tests/test_run_ollama.py` whose shared setup starts a local server.

I independently disabled each of the 12 listed guards in memory, one at a time. All 12 corresponding tests failed for the intended reason. Many cases exercise new code such as `GenerationWatch`, `check_finished_run` and `check_outcomes`, so running those against `2460e46` would mostly show that the new code is absent. The tests of the watch feed it fake events. They do not prove that the five wrappers observe the same prompts in a real training step.

## Done and next action

1. **Codex finished** the read-only re-review in `/Users/frank/code/model-training`. This report is the only checkout change. No GPU, Modal or network work ran.
2. **Claude Code in the desktop Code panel** should fix the three blockers on `m2-05-grpo` in `/Users/frank/code/model-training` and return the new commit for review. The Cowork project **Fine Tuning and RL training** can confirm D-057 separately.
3. **Elias has no decision to make now.** The next decision is cost approval in Cowork after the fixes pass review, before starting the full run.
