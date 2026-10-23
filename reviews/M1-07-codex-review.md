CHANGES REQUIRED

Reviewed on 2026-10-22 in `/Users/frank/code/model-training`, branch `m1-07-train`, commit `0de3dfe7f3a8fb1d753ed880926355758de715bf`. Compared every branch change with local `main` at `515dc06f2ae7d907be8c1fd53450f688b8d19987`.

The saved scores are correct. All 72 tests pass, and the saved validation answers select epoch 2 again. I found no evidence that the current training run used the test set. The changes below concern preserving results, keeping records tied to the right model and data, and stating what the baseline records actually prove. They do not show that the current scores were calculated incorrectly.

## Findings

### 1. Must fix: a smoke test can overwrite the final test results

**File and lines:** `scripts/eval_trained.py:63`, `80-81`, `98-123`.

**What is wrong:** A smoke test, meaning a small practice run on validation data, uses the same default output folder and filenames as the final test. Running the entry point with `smoke=True` and no separate output folder writes validation answers over all three final answer files and their settings record. Existing files are not checked before the remote job starts. A second full evaluation also overwrites existing results.

**Evidence:** I ran the local entry-point body with a made-up pair, a fake answer function, and all writes caught in memory. It tried to write the three existing `results/m1/trained-v1/qwen3-4b-trained-v1.<version>.jsonl` paths. No real files were changed and no remote function ran. The documented smoke command supplies a scratch folder, but the program does not require one.

**Why it matters:** A routine loading check can erase the saved final exam. Repeating the final exam can also replace evidence of the first run without notice.

**Fix:** Give smoke runs separate default names or require a separate empty folder. Before starting any remote work, refuse to replace an existing answer or settings file. Add offline tests for smoke defaults and existing outputs.

### 2. Must fix: choosing again can leave the upload pointing to the old choice

**File and lines:** `scripts/train.py:337-340`; `scripts/eval_trained.py:73-78`, `94-95`.

**What is wrong:** The `choose` stage replaces `meta["choice"]` but keeps an existing `meta["upload"]`. If a later selection chooses a different epoch, the record still points to the previously uploaded adapter, the small trained add-on. Final evaluation accepts that upload without checking that its checkpoint matches the current choice.

**Evidence:** In a local fake run, choosing again selected `epoch-1` but retained the existing upload for `epoch-2`. A separate fake evaluation with that mismatch accepted and dispatched the old upload commit. Outputs stayed in memory. The real saved record currently has matching `epoch-2` fields.

**Why it matters:** The script can score a different model from the one the validation rule selected while producing an apparently normal final result.

**Fix:** Remove an old upload record when replacing a choice, or retain it only after verifying it belongs to the identical choice and run. Final evaluation must reject a mismatch between `choice.chosen` and `upload.checkpoint` before loading the test set or calling Modal. Add offline tests for this sequence.

### 3. Must fix: selection does not check the recorded validation fingerprint

**File and lines:** `scripts/train.py:273`, `315-320`, `335-340`.

**What is wrong:** The `choose` stage reads today's validation file but never checks it against `train.meta.json`'s `valid_sha256`. SHA-256 is a fingerprint of the file's exact bytes. Selection can therefore use changed pairs or labels while keeping the old fingerprint in the same record. The prompt fingerprint is not checked at this stage either, although final evaluation checks it later.

**Evidence:** I supplied a fake training record with an intentionally false validation fingerprint and one made-up validation pair. Selection proceeded through all three fake checkpoint calls, wrote new scores, and kept the false fingerprint. The real training and validation files do match their recorded fingerprints today.

**Why it matters:** The record can say that one held-back dataset selected the model when another dataset actually did. Someone following the record could not repeat that selection.

**Fix:** Before selection, check the validation file and prompt against the training record and stop on any mismatch. Check the relevant saved run settings too, rather than silently using changed script constants. Calculate training data fingerprints from the same bytes that are loaded before dispatch, instead of rereading files after training returns. Add offline tests showing that mismatches stop before any remote call.

### 4. Must fix: qualify the claim that the baseline used identical base files

**File and lines:** `results/m1/trained-v1/summary.md:78`; `scripts/train.py:31-35`; `tests/test_train_data.py:234-239`.

**What is wrong:** The summary concludes that the baseline and trained model used identical base files. The saved baseline record names `unsloth/Qwen3-4B-Instruct-2507` at `992063681dc2f7de4ee976110199552935cad284`. Training and final evaluation name the ready-made 4-bit copy at `7744afa8566e264af1a92a806d8d9aae00cc7c78`. The summary explains an automatic redirect and says the baseline logs were not kept. The test checks only a repository-name suffix and a 40-character revision string. It does not establish which weight files the baseline loaded.

**Why it matters:** The explanation may be right, but the available local evidence does not prove the final sentence. Matching prompt fingerprints proves matching prompt text, not matching model weights. A published comparison should distinguish an expected match from a verified match.

**Fix:** Preserve any available evidence from the pinned loader and remote file history that establishes the baseline's actual files. If that evidence is unavailable, describe identical files as an inference and explicitly say the baseline's exact loaded revision was not independently verified. Update the test's comment so it does not claim to prove weight identity. Keep the existing answers and scores. A paid rerun is not required merely to qualify the claim, and none was authorized or performed for this review.

## Checks that passed

- **Answer-only training:** `build_rows` builds the exact scoring prompt followed by the JSON answer and end-of-turn token. `check_masks` checks every prepared training and validation row: prompt labels must be `-100`, meaning excluded from the training error calculation, and every answer token, including the end token, must be kept. It also rejects missing, reordered or shortened rows. The checks run after response masking and before training. The saved run reports 1,522 training rows and 99 validation rows, with longest rows of 539 and 530 tokens, below the 1,024-token limit. The local tests use a fake tokenizer; I did not repeat the GPU-side check with the real training libraries.
- **Thinking block and fair parsing:** The only special handling is in training text construction, `src/judge/train_data.py:59-64`. It permits the full conversation template to contain the exact empty thinking block but leaves that block out of the text used for learning. Neither scoring nor generation strips it from a model answer. A direct local check confirmed that an empty thinking block followed by JSON still fails parsing. The scorer, shared answer loop, prompt, formatter and GPU image file are byte-for-byte unchanged from `main`.
- **No detected data leakage:** The frozen test file passed its recorded fingerprint check. Both dataset checker commands passed, including checks for test-person names, exact pairs, near copies and train/validation overlap. These are the repository's word-overlap checks, not proof that no two examples express similar ideas. `scripts/train.py` reads only training and validation files. Final evaluation reads the test set through `load_frozen`, which checks its bytes before returning pairs. Git history contains the selection rule in `ac513bd`, before the saved training and validation results were committed. Rescoring all nine saved validation answer files exactly reproduced their recorded metrics and selected `epoch-2`. Local records support the reported sequence; they cannot prove the absence of unrecorded outside runs.
- **Same recorded scoring conditions:** Baseline v2 and trained 4B records have identical prompt fingerprints (`83d14d39a2115eb1f4d2d461581673090ec25d0a6f05d444cb2da8b756a2865c`), first rendered prompt fingerprints (`917bb8d558eb14ddc87cbe361b1da5c264940f56ebbe06e9047374e1430db52b`), greedy generation settings, chat-template options and three test versions. Both callers use batches of 16 and the same answer loop. The base-file evidence limit is finding 4.
- **Records:** Current training and validation fingerprints match `train.meta.json`. Training settings, both seeds, LoRA settings, base revision, loss curves, selection rule, validation results and uploaded adapter revision are recorded. The scoring record names the same base revision and adapter commit `1d9ebbf66e1a2abcad9932932bb7d810a38c59b9`, plus the frozen test fingerprint and generation settings. The saved per-epoch loss summaries recompute correctly. Findings 2 and 3 address what happens when stages are repeated.
- **Secrets and model files:** No likely credential strings appeared in the branch changes, and no model weight files are tracked by Git. The new scripts do not print or save the token value. The upload function requests a private repository and checks its privacy before uploading. The saved upload record says `private: true`. Live repository privacy and upstream authentication behavior were not checked.

## Local test and rescore results

Ran `uv run pytest -p no:cacheprovider` with `UV_OFFLINE=1`, `UV_NO_SYNC=1`, `UV_NO_CACHE=1` and `PYTHONDONTWRITEBYTECODE=1`. Result: **72 passed in 0.65 seconds**. These options prevent network dependency updates and repository cache writes.

With the same environment, ran:

```text
uv run python -m judge.score data/m1/test.jsonl results/m1/trained-v1/qwen3-4b-trained-v1.as_is.jsonl results/m1/trained-v1/qwen3-4b-trained-v1.swapped.jsonl results/m1/trained-v1/qwen3-4b-trained-v1.no_dates.jsonl
```

| Version | Precision: correct alarms / all alarms | Recall: caught / real contradictions | Correct labels | Invalid | Format errors |
|---|---|---|---|---|---|
| As is | 95/96 = 0.989583 | 95/100 = 0.95 | 193/200 | 0 | 0 |
| Swapped | 96/101 = 0.950495 | 96/100 = 0.96 | 188/200 | 0 | 0 |
| No dates | 90/92 = 0.978261 | 90/100 = 0.90 | 188/200 | 0 | 0 |

These match `summary.md`. Both saved baseline-v2 models were also rescored; their table values match. The trained model clears the stated precision and recall bar in all three versions and improves on the saved untrained 4B answers. Against baseline-v2 8B, recall is lower both as is (0.95 versus 0.96) and without dates (0.90 versus 0.96). The summary gives those numbers correctly, but its last sentence on line 33 should mention both cases rather than saying only the no-dates case falls short. This wording correction is **nice to have**; it does not change the scores.

No network or GPU/Modal jobs were used. Current outside documentation, remote commit history, live uploaded files and pricing were therefore not independently checked. This review's findings rely on local code, saved records and offline probes. Only this report was added to the repository; code, datasets, project records, branch and commits were left unchanged.

## Done and next action

1. **Done, Codex:** reviewed the requested commit, ran the local checks and wrote `/Users/frank/code/model-training/reviews/M1-07-codex-review.md`.
2. **Next, Elias:** in Claude Code's desktop Code panel, open `/Users/frank/code/model-training` on `m1-07-train` and ask it to address the four must-fix findings, preserve the existing answers, and add offline tests. No new training or final test run is needed for the fixes described here.
3. **After fixes:** Codex reviews the changes; Claude in the Cowork project "Fine Tuning and RL training" checks the result and handles merging. This review did not commit, push or merge, as the review prompt explicitly forbids those actions.
