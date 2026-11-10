CHANGES REQUIRED

Reviewed `m2-05-grpo` at `7be1643273eb4b210478146e4a3d15f6ac4acb86` against `6c72da5` on 2026-11-09. The original shortened-prompt and stop-rule probes now fail as intended. An interrupted choice now reuses its saved answers. Three issues still block the full run.

## Must fix

### 1. The generation check accepts padding on the wrong end

**Where.** `scripts/train_grpo.py:374`, `:385`, `:428`, and `:476`. Padding is extra space used to give all prompts in a batch the same width. The attention mask marks which spaces are real prompt tokens.

**What I reproduced.** I used the real wrappers and changed the fake trainer to pad on the right. A short prompt's mask changed from `[0, 0, 1, 1, 1]` to `[1, 1, 1, 0, 0]`. All five checks passed. The watch removes padding before comparing tokens, and accepts a continuous stretch of prompt tokens anywhere in the row.

**Why it blocks.** The pool pads on the left. On the pinned decoder-only generation path, the first new token uses the prediction at the last input position. With right padding, a shorter row ends on padding and can sample differently. An ordinary change to the padding side would pass the guard. The current pinned TRL text path already pads on the left, so this does not show that the old trial or current trainer uses the wrong side. See `src/judge/generate.py:79` and the local transformers 5.17.0 source at `generation/utils.py:2667-2681` and `:3042`.

**Fix.** Require any padding to be on the left. The last position in every row must be a real prompt token. Add a real-wrapper test with unequal prompt lengths and right padding.

### 2. Recording the bill can strand an upload retry

**Where.** `scripts/train_grpo.py:1130`, `:1163`, `:1171-1175`, `:1372`, and `:1414-1416`.

**What I reproduced.** I ran `upload_checked` against the fake Hub. First I let the remote upload finish and simulated the loss of its local receipt. Then I added the billing fields that `bill` writes and retried. Recovery refused the exact same adapter and choice. I also interrupted an upload after the empty repo was reserved. Adding the same billing fields made that repo impossible to resume. The retry hashes a fresh copy of the entire local record, so the bill changes the identity of the upload.

**Why it blocks.** Running the supported billing stage while recovering from an interruption is an ordinary operation. It changes no model weights or validation decision, but can leave an already uploaded model without a local receipt or leave an empty repo that this run can no longer use.

**Fix.** Give the upload a stable record. For example, save the exact upload payload when the attempt begins and reuse it after checking the same run, choice and adapter fingerprints. Keep later billing fields in the local record. Test both retry cases after a billing update.

### 3. The upload can overwrite a file added after its repo check

**Where.** `scripts/train_grpo.py:1166-1198`. The code checks whether it may write, prepares the upload, and then sends it without tying the write to the repo commit it checked.

**What I reproduced.** With the real upload helper and the fake Hub, another writer added an `adapter_config.json` during the copy step. The upload replaced that file and returned success. The final check saw the intended files in the new commit, so it could not tell that the previous writer's file had been replaced.

**Why it blocks.** Two ordinary writers can overlap at the fixed repo name. Verifying the new commit proves what landed, while the requirement also protects files already there. The fake demonstrates the missing condition in the code. I did not run a live Hugging Face write.

**Fix.** Make the upload conditional on the repo still being at the checked commit. Verify the repo is private before sending, including on the fresh-create path. Test a competing commit between preflight and the write. If a worker stops between creating the repo and saving its first commit in the note, the retry correctly refuses an unproven repo. Change the error at `:1179` to ask the operator to verify ownership and state before any cleanup. An empty repo alone does not prove it belongs to this attempt.

## Checks that passed

- **Exact prompts and stop rules.** The real wrappers now reject shortened or reordered input rows, two generation calls in one trainer turn, a missing or changed end-token rule, and a length limit other than padded width plus 256. The positive wrapper test passed with unequal prompt lengths and left padding. The wrapper accepts IDs passed by name or as the first positional argument. The pinned regular text path passes two-dimensional token and mask tensors by name, with one generation call. I found no evidence in that path of a correct call being rejected for its argument shape, mask shape or call count. This was a source check plus offline probes. The new wrappers have not run with the actual GPU stack.
- **Pinned generation behavior.** The local TRL 1.15.0 source at `grpo_trainer.py:398-405`, `:1116-1131` and `:1769-1828` takes the model's end-token list, adds the tokenizer's end token if absent, and passes the list to generation. The pool call supplies no end-token override. In the local transformers 5.17.0 source, `generation/utils.py:1863` adds the requested 256 tokens to the input width and `:1487-1515` constructs the length and end-token rules. The reviewed notes identify the pinned model's two IDs as 151645 and 151643 and say the tokenizer's end token is already in the list. I could confirm how the code carries those values. The original pool record did not capture the actual end-token rule, and I did not have the pinned model config and tokenizer files to independently verify the two numbers. The local TRL sampler implementation at `trainer/utils.py:932-951` also confirms that the 314 pairs split evenly into 157 two-pair batches per pass. With one new generation per optimizer step, 314 steps cover two groups per pair. [Settings and source references](../results/m2/grpo/settings.md)
- **Upload with a stable record.** The original same-weights, different-config, missing-record probe now stops. The helper checks every intended file and the training record at one commit, with the documented exception for the Hub's `.gitattributes` file. With an unchanged payload, the fake upload resumed into its reserved empty repo after a connection failure and recovered a finished upload without sending again. The staged copy is checked before send and the returned commit is checked afterward. The two upload findings above concern retries and concurrent writes outside those successful cases.
- **Validation recovery.** I interrupted the local record write after `valid/` was named. A normal retry dispatched zero workers and produced the same winner, counts, scores and exclusions as a clean run. I separately changed the run ID, checkpoint fingerprints, reward fingerprint, validation fingerprints and item-list fingerprint. Each retry stopped, left the folder intact and dispatched nothing. The suite also rejects changed answer bytes and a saved winner that the answers do not support.
- **Cost.** The timing table distinguishes checks during generation, counts at step end, a missing log detected on the next step, and coverage after step 314. A first-step failure has a rough GPU-only estimate of about $0.05 based on the old trial. The 305-step note now states that $0.99 is an estimate and that some pairs would get one group instead of two. The full-run estimate remains about 71 L4 minutes and **$1.02**, above the approval threshold. No current prices or bill were checked. [Full-run plan](../results/m2/grpo/full-run-plan.md)
- **Saved evidence.** No source under `src/`, saved pool file, pool result or `trial.meta.json` changed against `2460e46`. The recomputed reward fingerprint was `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`. Git lists the same stored IDs for both frozen files. The review-range whitespace check passed.

**Known limit under D-053.** Code written to forge fingerprints and matching records or to replace the generation or reward code in memory can fool these checks. The blockers above were reproduced through ordinary input preparation, billing or overlapping writes.

## Test evidence

The guarded offline `uv run` rerun passed **619 tests**, including all 121 GRPO cases, with **25 deselected**. The guard reported zero attempted reads of either frozen file or protected home config, zero checkout writes, and zero network attempts in the test process. The Modal collection subprocess used a fake home. I did not run the unrestricted suite.

- Seven cases of `tests/test_score.py::test_m1_saved_answers_rescore_to_the_same_numbers`.
- Six publishing checks in `tests/test_prepublish_check.py`. They are `test_real_cards_pass`, `test_wrong_score_in_a_table_is_caught`, `test_unknown_row_in_a_score_table_is_caught`, `test_wrong_count_is_caught`, `test_untraced_number_in_prose_is_caught` and `test_every_fingerprint_in_the_cards_is_known`.
- Twelve cases in `tests/test_run_ollama.py` that start a local server.

I independently replayed the 15 guard changes described in the fixes note in memory, one at a time. Each change made at least one targeted test fail. Across those altered runs there were 26 expected failures and 51 passing cases. A few failures were later checks catching the bad state after the intended early check was removed. The ordinary suite used the original source. No new test started a live job or accessed a real frozen file.

## Done and next action

1. **Codex finished** the read-only review in `/Users/frank/code/model-training`. This report is the only checkout change.
2. **Claude Code in the desktop Code panel** should fix the three findings on `m2-05-grpo` in `/Users/frank/code/model-training` and return the new commit for review. D-057 remains with Claude in the **Fine Tuning and RL training** Cowork project, outside this review.
3. **Elias has no decision to make now.** The next decision is cost approval in Cowork after the fixes pass review, before starting the full run.
