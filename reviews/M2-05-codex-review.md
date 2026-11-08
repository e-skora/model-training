CHANGES REQUIRED

Reviewed branch `m2-05-grpo` at `2460e46f942f2b0ee2f17e9be72bca43e57aa627` on 2026-11-08, against `main` at `79e083af5fecaffbac11b192fc772c5dc887ec23`. The saved 20-step trial gives good evidence for that run. The full run still needs fixes to how the script checks generation, protects saved models, and chooses a model. A checkpoint is a saved copy of the model's adapter, the small set of weights that this run trains.

## Findings

### 1. Must fix. Checkpoint folders can be reused and the uploaded model can differ from the one chosen

**Where.** `scripts/train_grpo.py:508`, `:534`, `:611`, `:639`, `:648`, `:690`, `:700`, `:794`.

Training protects the local result file but opens the remote log in write mode and saves checkpoints to reusable folders. For example, `train --run trial` passes the local checks when there is no `train.meta.json`, even with a saved trial. A retry after an interruption or a second checkout using the default name `v2` also targets the same folders. Choose and upload trust those folder names. They do not check that the weights in the folder are the ones saved by the recorded run or scored during validation. The upload also accepts an existing private destination and reports the repository head from a separate lookup, which can differ from the commit created by that upload.

**Why it blocks.** Ordinary retries and reused names can overwrite evidence or send up a different model from the one that passed validation. The reserved-name case reached the training dispatch in an offline probe with a fake worker.

**Fix.** Reserve a fresh, unique run folder before any checkpoint writes. Record a fingerprint of each saved adapter and its config. A fingerprint is a short value computed from the file bytes. Verify those bytes when scoring and again when preparing the upload. Protect the remote destination against replacement on a retry and record the commit returned by the upload itself. Write local results atomically and make an interrupted run or upload recoverable without reusing a checkpoint folder.

### 2. Must fix. Choose and upload accept an incomplete or stale run record

**Where.** `scripts/train_grpo.py:374`, `:697`, `:743`, `:751`, `:761`, `:789`.

Choose trusts the checkpoint list in the local record. It does not require a finished full run, the exact expected checkpoint list, or one complete answer set per candidate. It also ignores the validation fingerprints checked before training. Upload compares the validation files with the choice, but that choice can already be based on changed files. Both stages compare the recorded reward fingerprint to a constant. Neither recomputes it from the current source, which also contains the code that scores validation answers. An offline probe made an incomplete record with only step 50, no sampling evidence and a changed validation fingerprint, and simulated a changed source fingerprint. Choose still dispatched step 0 and step 50, and the resulting choice reached fake upload.

**Why it blocks.** A partial record or ordinary edit between stages can change who wins under D-056 or spend money on a comparison that should stop.

**Fix.** Before dispatching choose or upload, require a completed full-run record with the expected steps, checkpoints, and run evidence. Recompute the source fingerprint and match both validation files to the recorded fingerprints. Before saving a choice, check that each planned candidate returned exactly one answer for each expected item. Treat missing output as a failed evaluation. Before upload, verify that the recorded winner still follows D-056 and is one of the recorded checkpoints. Validate results before writing any final answer files so a failed choose can be retried safely.

### 3. Must fix. The sampling check can pass after the answer generation changes

**Where.** `scripts/train_grpo.py:118`, `:451`, `:463`, `:488`, `:553`, `:563`, `:580`. The tests at `tests/test_train_grpo.py:206` and `:226` check settings on paper and whether a fake record has a field called `sampling_used`.

The check compares each observed prompt's unpadded token IDs with the pool. A token ID is the model's number for a piece of text. That is a good membership check for the calls it sees. It does not check eight copies of each prompt per group or reconcile generated, checked and rewarded answers. The code records how many answers each prompt copy requests, but it does not require that value to be one. It also blocks only a list of named token filters. An offline probe of the two real recording functions accepted both two returned answers per prompt copy and a filter that bans specified words. If a generate path stops calling the recording functions, the run can finish and save an empty record of checks.

**Why it blocks.** An ordinary generation-setting edit or change in how the trainer calls the model can change the distribution of answers without stopping the job. The trial shows neither failure.

**Fix.** Check the actual generation entry point and fail on unapproved token filters or stop rules. Require one returned answer per prompt copy and verify eight copies of the same pool prompt in each group. Reconcile generation, prompt and reward counts before accepting checkpoints, and verify the planned per-pair coverage for the full run. Test the runtime checks with fake generation, including one call that skips recording. The training record should also fail clearly if any required per-step watch metric goes missing. Today the watch can quietly skip such a step at `:521` or `:263`.

### 4. Must fix. A model can win with answers that break v1's format

**Where.** `scripts/train_grpo.py:290`, `:328`, `:772`. See also `src/judge/score.py:145`, `:174`, `:188` and `src/judge/reward.py:48`.

The chooser only checks the shared scorer's narrow invalid-answer count. The scorer counts a missing type or reason separately as a format error and still scores the label. The training reward treats those same answers as invalid. With a fake worker I made a candidate answer each question with only its correct label. It got 97 format errors on the 97 hard validation pairs in the as-written version, beat a baseline that missed one contradiction, and reached fake upload.

**Why it blocks.** An ordinary model can leave out a required field. D-024's reporting can keep label scores and format errors separate, and Style A still requires v1's answer shape. The current choice can approve a model that breaks that shape.

**Fix.** Keep the shared scorer's published counts. Add an explicit Style A eligibility check for both validation sets and all versions. Use at least the same type and reason checks as the training reward. Have Cowork record how this format gate relates to D-056's “any invalid answer” rule before the full run.

### 5. Must fix. Importing the test script can read the real Modal config

**Where.** `tests/test_train_grpo.py:29`, `:36`, `scripts/train_grpo.py:38`. Installed local Modal client source, `modal/config.py:121`, `:175`, `:182`.

The fake workers prevent live jobs, but they are used after the test imports the real client. The guarded test run caught that import trying to open the real `~/.modal.toml`. Older tests importing Modal scripts did the same. The guard blocked all four attempts before reading the file. The suite passed when rerun with an empty config and Modal and Hugging Face environment settings removed from the test process.

**Why it blocks.** On a normal logged-in Mac, this test can read the user's home config, contrary to AGENTS.md rule 10.

**Fix.** Set an empty temporary Modal config before any test imports a Modal script, or move the offline pieces behind an import that does not load Modal. Test collection with the real config path blocked.

### 6. Nice to have. Correct the cost and record details before asking for approval

**Where.** `results/m2/grpo/full-run-plan.md:28`, `:35`, `:36`, `results/m2/grpo/trial.meta.json:49`, `:77`, `:527`, `:593`, `scripts/train_grpo.py:719`, `:824`.

The 255.8 seconds for steps 2 through 20 already includes both checkpoint saves. The plan multiplies a rate that includes saves across every later step and then adds more saves. Using the observed steps without saves gives 12.65 seconds per step. The two save steps add about 7.75 seconds each over that rate. Counting seven saves, loading and first-step overhead once gives about 4,255 seconds, or 71 L4 minutes. With the rate and 8% allowance used in the plan, that is about **$1.02**. It is still over $1 and is an estimate from a short trial. The choose estimate of about 50 L4 minutes and $0.71 is consistent with its inputs. The measured memory peak was 7.49 GiB for 40 of 314 pairs, so describe it as trial headroom.

The saved trial lists `save_every: 50` even though it saved at 10 and 20. Its inside-cost estimate only counts the training worker. The separate load check brings the measured total to 628.9 seconds, or 10.48 L4 minutes. The billed total of $0.1775 is present. The billing step tracks train and choose app IDs but does not collect a separate upload app ID.

**Why this rating and fix.** The arithmetic and labels can mislead a reader, but the over-$1 decision is unchanged and the trial's source measurements are available. Update the estimate and its uncertainty, store the actual cadence and GPU time for each stage, and label the costs by the stages they cover. Rates and billing here come from the repo's recorded inputs and the plan's [Modal price reference](https://modal.com/pricing). No live prices, credits or bill were checked.

## What checked out

- **Starting model.** Both load calls use the exact D-031 commits, and the existing adapter is loaded as trainable. The code does not add another adapter. The trial recorded matching settings and 504 trainable tensors, the same count as the 504 tensors compared against the pinned adapter file with zero difference. The guard at `scripts/train_grpo.py:437` checks that every trainable name contains `lora_`, but does not match the exact trainable name set to the saved adapter. Matching those sets would also catch an accidental partial freeze before a future run.
- **Training data and validation.** The dataset test compared all 314 prompts and their saved order/date choices to `judge.pool.show`, using the original candidate files. The checked pool hash was `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77`. The host reads both validation files for the overlap check and sends only pool rows to the training worker. Only the trial load check and choose send validation prompts to a model. The two files currently match the trial's recorded fingerprints and have 97 and 99 pairs.
- **Sampling and settings evidence.** For the recorded 20 calls, the trial saw 320 pool prompts, 40 distinct, temperature 1.0, top_p 1.0, top_k off, min_p unset, repetition penalty 1.0, one beam, a 256-token limit and no token filters. It requested one answer per prompt copy. The token match shows that the rendered prompts reaching generation agree with the pool template for those calls. My assessment of TRL, Unsloth and transformers uses the installed-source excerpts cited in `settings.md:11`, `:35`, `:42`, `:51`, `:78` and the saved trial evidence. Those excerpts support the wrapper condition that skips `_caller_sampling_only` here and the explicit TRL values taking precedence over model defaults. I did not independently inspect those GPU package sources or rerun the trial. The trial log identifies Unsloth's automatic `epsilon_high=0.28`; HEAD sets it explicitly and records the resolved values. Keeping the truncation mask off lets a response cut off before it completes valid JSON contribute its negative reward to training. No trial response hit the cap. The small pool and zero pull toward the starting model make the validation forgetting checks important. I found no setting that justifies retuning from the trial.
- **Watch and load check.** I reconciled all 20 logged steps with the reward totals. Mean reward was 0.785, mean length 41.0 tokens, and 47.5% of groups had no spread in reward. The 320 answers split 160 contradiction, 113 non-contradiction, 46 unclear and 1 invalid. The 20-step window correctly gave no reward warning. Waiting for a full window fits the written rule. The raw log also records a separate xformers fallback at line 12. All eight trial load-check answers passed the reward reader's format check.
- **D-056 score math.** The chooser uses the lowest score across all three versions, exact fractions for the strict 0.02 and 0.03 cutoffs, and the written recall, precision and earlier-step order. Its zero precision for a model that never says contradiction is a sensible conservative selection value. The shared score still reports precision as undefined in that case. The greedy worker reuses `load_judge`, `answer_items` and the batch size from M1. Tests cover a step-0 win, disqualifications, ties and the exact 0.02 boundary. Two worthwhile test additions are reversing the input order for a tied pair of checkpoints, so the test fails if the earlier-step rule is removed, and testing exactly 0.03 for both M1 cutoffs.
- **Unchanged files.** `git diff main..HEAD` shows no change under `src/` or in the five named saved pool files. The live local calculation returned reward fingerprint `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`. Git lists the same stored file IDs for the two frozen files on both commits. Their contents were never opened. The whitespace check passed for code, tests and docs after leaving out terminal progress output in the raw trial log.

**Known limit under D-053.** The reward fingerprint covers source bytes on disk. Deliberate code that replaces reward logic in memory or falsifies the training record can still defeat it. This does not block the run. The failures above can happen through ordinary edits, retries or model outputs.

## Test evidence

The offline `uv run` rerun used the existing environment with no dependency sync. **540 passed, 25 deselected.** All 42 M2-05 cases ran. The final Python test guard recorded zero blocked attempts to read either frozen file, read protected credentials, write to the checkout, or open a network connection. Test caches and bytecode writes were off. The unrestricted suite was not run.

- Seven cases of `tests/test_score.py::test_m1_saved_answers_rescore_to_the_same_numbers` were excluded.
- Six tests in `tests/test_prepublish_check.py` were excluded. They are `test_real_cards_pass`, `test_wrong_score_in_a_table_is_caught`, `test_unknown_row_in_a_score_table_is_caught`, `test_wrong_count_is_caught`, `test_untraced_number_in_prose_is_caught` and `test_every_fingerprint_in_the_cards_is_known`.
- Twelve cases in `tests/test_run_ollama.py` were excluded because their shared setup starts a local server.

## Done and next action

1. **Codex finished** the read-only review in `/Users/frank/code/model-training`. The review file is the only checkout change. No GPU, Modal or network work was started.
2. **Claude Code, in the desktop Code panel**, should fix the five must-fix findings on `m2-05-grpo` in `/Users/frank/code/model-training`. Then Codex can review the corrected commit and Cowork can update the cost estimate.
3. **Elias has no decision to make now.** The next decision comes in the Claude Cowork project **Fine Tuning and RL training** after the fixes pass review. Approve the updated estimate there before the full training run starts.
