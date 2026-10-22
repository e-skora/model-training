CHANGES REQUIRED

Reviewed on 2026-10-22 in `/Users/frank/code/model-training`, branch `m1-05-baseline`, commit `c568333d0993f8c2aab9afa8f008ccaeb897c103`.

During the review, another process advanced this branch to `aac5270fdf9a6897618c204697d5d098209baa4b`. I checked the difference: it only added a handoff document outside this review. All reviewed code, tests, results, test data, dependencies, and project rules are unchanged between those commits. The findings therefore also apply to that later commit.

The saved scores reproduce correctly under the current scoring rules. The main problems are the mismatch about invalid answers, a missing frozen-file check in the standalone scorer, and acceptance of some text that is not valid JSON. The review is finished. Fix these before merging Step 5.

## Findings

### 1. Must fix: settle how invalid answers affect the reported scores

**Files and lines:** `src/judge/score.py:76-89`, `tests/test_score.py:64-79`, `results/m1/baseline/summary.md:14-18`.

**What is wrong:** Invalid answers never enter the precision calculation. Precision here means the share of answers labelled "contradiction" that are right. That is the standard definition, and the code calculates it correctly. However, it does not satisfy this review's requirement that invalid answers count as errors in precision too. The existing invalid-answer test explicitly accepts this behavior: precision is undefined when every contradiction answer is invalid.

**Why it matters:** Qwen3-4B has 83 correct contradiction answers, 6 false alarms, and 10 invalid answers. Its reported precision is `83 / 89 = 0.932584`. All 10 invalid answers are excluded from that number. Eight are real contradictions, so they correctly lower recall to `83 / 100`. The other two are non-contradictions and do not affect standard recall. In a small local example, one correct contradiction and two invalid answers still produce precision `1.0`.

**Fix:** Make the rule explicit before using these results to choose or judge a trained model. Keep standard precision and recall clearly named. To apply an invalid-answer penalty, add a separately named measure with a written formula and use it consistently in the summary and acceptance rules. For example, dividing correct contradiction answers by contradiction answers plus all invalid answers gives Qwen3-4B `83 / 99 = 0.838384`, not `0.93`. This is a different measure, not a correction to standard precision. Add tests mixing correct answers, false alarms, invalid positive examples, and invalid negative examples. Do not say every invalid answer lowers both standard measures.

### 2. Must fix: check the frozen file before standalone scoring

**Files and lines:** `src/judge/score.py:100-104`; compare `scripts/run_baseline.py:105-114`.

**What is wrong:** The baseline runner checks the test file's SHA-256, its file fingerprint, before loading pairs or requesting model answers. Its fixed value matches `STATE.md` today. The standalone scoring command does not check any fingerprint. It directly loads whichever test file it receives.

**Why it matters:** Someone can edit or replace the test file and then get apparently normal scores from the documented scoring command. An in-memory check confirmed that this command accepted substituted test data without checking `STATE.md`. The actual frozen file was not changed.

**Fix:** Add a shared frozen-file loader and use it before scoring the M1 test file as well as before running models. Compare with the frozen entry in `STATE.md`, or check that an authoritative fixed value agrees with that entry. Keep the pure `score()` function usable with made-up examples. Add a temporary-file test showing that changed bytes stop scoring before results are produced.

### 3. Must fix: reject invalid JSON constants and ambiguous repeated fields

**File and lines:** `src/judge/score.py:38-46`.

**What is wrong:** The JSON reader accepts `NaN` and `Infinity`, which are not valid JSON values. For example, `{"label":"contradiction","type":null,"reason":NaN}` is accepted with no problem recorded. It also silently keeps the last value when a field is repeated. An answer containing both `"label":"contradiction"` and `"label":"non_contradiction"` is accepted as the second label.

**Why it matters:** Text that should fail the answer-format check can receive credit. A repeated label can even change the scored answer according to field order.

**Fix:** Reject non-JSON constants and repeated field names while reading an answer. Add tests for both. No such constants or repeated fields were found in the three saved answer files, so this issue does not change their current scores.

### 4. Nice to have: report whether the whole answer follows the prompt's format

**Files and lines:** `src/judge/score.py:42-46`, `src/judge/prompt.py:18-25`, `results/m1/baseline/summary.md:18`.

**What is missing:** A recognized label is enough for acceptance. The parser also accepts missing `type` and `reason` fields, invented types, and reasons that are not text. The prompt requires `type` to be JSON `null` for contradictions, but 90 saved Phi answers have a different value, including the text `"null"`. Phi still has zero invalid answers under the summary's narrower definition of invalid.

**Why it matters:** Zero invalid answers does not mean a model followed the full requested format. That distinction will matter when another program uses the answers.

**Fix:** Add a separate count for answers that break the required fields or field types, and test it. Keep this distinct from label accuracy unless the project explicitly changes its scoring rule. Do not silently change the meaning of existing baseline scores.

### 5. Nice to have: reject duplicate or unexpected answer IDs

**File and line:** `src/judge/score.py:59`.

**What is missing:** Two answers with the same pair ID silently replace each other; the last answer wins. Answers for unknown IDs are ignored.

**Why it matters:** An accidentally combined or wrong answer file can produce scores without warning. A local example changed the score simply by adding a second answer for the same ID.

**Fix:** Reject duplicate and unexpected IDs with a clear message. Keep the existing rule that a missing answer counts as wrong. Add made-up test cases. All three saved files currently have exactly 200 unique IDs matching the test set, so this does not change today's results.

### 6. Nice to have: save enough run settings to verify the decoding claims

**Files and lines:** `scripts/run_baseline.py:20-24`, `scripts/run_baseline.py:62-77`, `scripts/run_baseline.py:82-90`, `results/m1/baseline/summary.md:5`.

**What is missing:** The code passes `do_sample=False`, which requests answers without random sampling. It does not explicitly set `num_beams=1`, the single-choice setting needed to establish greedy decoding without relying on a downloaded model's defaults. It passes `enable_thinking=False` into Qwen3-8B's chat template. However, the full-run metadata saves neither the final generation settings nor a rendered prompt: all three `first_prompt` values are null. Downloaded model and tokenizer revisions are also not recorded.

**Why it matters:** The source shows the intended settings, but the saved records cannot independently establish all settings used in the completed GPU runs or guarantee identical answers on a future run.

**Fix:** Set `num_beams=1` explicitly. Record the effective generation settings, template options, model/tokenizer revisions, and a rendered prompt or its fingerprint. Add offline tests with fake model and tokenizer objects to check that settings are forwarded. Describe repeatability more carefully in the summary. No new GPU run is needed to fix these records and tests prospectively.

## Local verification

I ran the tests with the package runner offline, dependency changes disabled, Python bytecode writes disabled, automatic pytest plugins disabled, and pytest's cache disabled:

```text
UV_OFFLINE=1 UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never UV_NO_SYNC=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --offline --no-sync pytest -p no:cacheprovider -q
7 passed in 0.03s
```

The seven tests use made-up pairs and a temporary answer file. They do not call the network, read personal home-folder data, or run Modal. They cover ordinary scoring, wrong labels and types, missing answers, a JSON code box, surrounding prose, file loading, and prompt construction. They lack the cases requested in the findings above, including a mixture of valid contradiction answers and invalid answers in the precision test.

I checked the frozen fingerprint before loading the real pairs, then called `score()` on each saved answer file. I also checked label counts independently. These exact scores reproduce the summary's rounded values:

| Model | Precision | Recall | Label accuracy | Type accuracy | Invalid | Code boxes |
|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 83/89 = 0.932584 | 83/100 = 0.83 | 175/200 = 0.875 | 81/92 = 0.880435 | 10 | 0 |
| Qwen3-8B | 94/98 = 0.959184 | 94/100 = 0.94 | 190/200 = 0.95 | 63/96 = 0.656250 | 0 | 0 |
| Phi-4-mini-instruct | 85/95 = 0.894737 | 85/100 = 0.85 | 175/200 = 0.875 | 80/90 = 0.888889 | 0 | 5 |

Contradiction is consistently the positive label, meaning the outcome precision and recall measure. Type accuracy includes only correctly labelled non-contradictions, exactly as the summary says. All ten invalid Qwen3-4B answers use `update` as their label.

The parser accepts one JSON object, optionally inside one whole-answer markdown code box, with surrounding whitespace allowed. It rejects surrounding prose, arrays, plain label text, unknown labels, and different label capitalization. Accepting the five Phi code boxes agrees with the documented scoring policy.

The runner builds messages once with `build_messages()` and sends those same messages to each model. The function excludes the correct labels, types, and explanations from model input. Each model then applies its own chat template. The shared image pins its listed package versions; the baseline function requests an L4 and has a 1,800-second timeout. These are observations of local source, not fresh verification of outside tools.

The stored durations reproduce 3.0, 3.4, and 2.6 minutes. Applying the summary's stated price to those durations reproduces its rounded cost estimates. I did not check the current price, package compatibility, or Qwen template behavior against online documentation because network use was forbidden. No claim of fresh external verification is made.

An attempted extra operating-system sandbox failed to start with `sandbox_apply: Operation not permitted`. The test rerun then used the offline settings above. No network, GPU, or Modal job was used.

The frozen test file is only read by the reviewed code. Its fingerprint matched `STATE.md` before and after this review:

```text
6e2a08323f445477f9600e371f1074ad3e2b41ef97d7072d6734ef481f6e574b
```

## Done and next steps

1. **Codex, done:** Reviewed and reran local checks in `/Users/frank/code/model-training` using the Codex desktop app. Wrote only this review as a project change. I did not change source, data, results, branches, or commits.
2. **Elias, next:** Give this review to Claude Code in the desktop Code panel, opened at `/Users/frank/code/model-training`, to handle the must-fix findings and rerun the offline checks. Settle the invalid-answer scoring rule explicitly before changing reported measures.
3. **Claude in Cowork, after fixes:** In the project "Fine Tuning and RL training", review the corrected evidence before merging Step 5. No additional input from Elias was needed to finish this review.
