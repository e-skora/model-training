CHANGES REQUIRED

Reviewed by Codex on 2026-10-27. Branch: `m2-04-pool-and-reward`. Commit: `a28db378b2ac742c87f8d8eb067f33e61aae58f3`. Compared with local `main` at `8f198391abb88dd0c46d940411b15feca07bbc85`.

The reward values and the saved 314-pair pool are correct. Three checks need fixing before Step 5. All three problems below were reproduced with temporary, made-up data or changes held only in program memory. None was found in the saved pool itself.

## Findings

### 1. Must fix, P2: the reward fingerprint misses changes to accepted answers

**File:** `src/judge/reward.py:74-78`. Related rules: `src/judge/score.py:26`, `38-48`, `66`, `106`.

A fingerprint is a file's or rule's identifying code, calculated with SHA-256. Here the fingerprint includes four functions' source text, but leaves out things those functions use: `TYPES`, the code-fence pattern `FENCE`, `reject_constant`, and `no_repeated_fields`.

**Reproduced:** with the existing fingerprint unchanged:

- Removing `update` from the scorer's allowed types changed an update answer's reward from +1.25 to -2.0.
- Replacing the repeated-key checker with ordinary dictionary reading let an answer with two `label` fields earn +1.0 instead of -2.0.
- Making the special-number reader return null let an answer with `"type": NaN` earn +1.0 instead of -2.0.

**Why it matters:** two training runs could record the same reward fingerprint while teaching different answer rules. `choose_pool` also relies on this fingerprint to stop when the reward has changed. The current test changes only table values, so it misses this problem.

**Fix:** include the shared parser's full relevant code and constants, as well as the reward rules. Include completion-reading rules if the fingerprint is meant to identify the full training reward. Add tests for changes to allowed types, repeated keys, special numbers and fences. Keep the original saved results and their recorded fingerprint; record any revised fingerprint and offline recheck separately.

### 2. Must fix, P2: pool selection accepts samples made with a different prompt

**File:** `scripts/sample_pool.py:383-390`.

`choose_pool` checks the reward and candidate-file fingerprints, but never compares the saved `prompt_sha256` with today's `prompt_sha256()`. Rebuilding `show(pairs)` checks only whether order and dates match. It does not check the words the model was asked to read.

**Reproduced:** sample the fake 21-pair repository from `tests/test_pool.py`, change `judge.prompt.SYSTEM_MESSAGE` in memory, and run `choose_pool`. It writes a pool successfully even though the saved and current prompt fingerprints differ.

**Why it matters:** a pair chosen because the old prompt caused mixed answers could be used with new instructions that no longer cause that behavior. The saved version checks give no warning. D-027 requires the prompt to be fixed and identified.

**Fix:** compare the recorded prompt fingerprint before writing any outputs. Also check the recorded presentation seed against the rules being applied. Add a fake-data test proving that a changed prompt stops selection before any output is written. The current saved prompt matches today's prompt, so this finding does not require a new GPU run.

### 3. Must fix, P2: duplicate sample IDs can hide an unsampled candidate

**File:** `scripts/sample_pool.py:393-403`. Later dictionary construction at `406-407` hides repeated IDs further.

For a full run, the script checks the number of rows, but not that each candidate appears exactly once. Two rows for one candidate can replace another candidate's row while leaving the total unchanged.

**Reproduced:** in the fake 21-pair run, replace the second sample row with a copy of the first. Selection succeeds and writes 21 pool rows containing only 20 different IDs; `tr2-0002` is missing. Every remaining version and answer-count check passes.

**Why it matters:** a damaged or incorrectly combined sample file can silently omit a candidate and give another candidate extra training weight. The report can still show the expected row count.

**Fix:** reject repeated and unknown IDs, and require the full run's ID set to equal the candidate ID set before choosing or writing. For a smoke run, require exactly the intended smoke IDs. Add tests with a repeated ID replacing a missing one, and check that no outputs are written.

## Reward table, format and training wrapper

- An independent check of all 25 combinations of valid right and given label/type answers matched D-040 and D-046. The highest possible reward is +1.25. Existing tests cover every table rule and each correct non-contradiction type, but not every combination; the review checked the full combination table separately.
- Text before or after JSON, repeated keys, `NaN`, `Infinity`, wrong types, a type on `contradiction` or `unclear`, and empty reasons all receive -2.0.
- One surrounding code fence is accepted by both the scorer and reward. Extra fields and a 200,000-character reason also pass both readers. These earn no extra points. Those are inherited format allowances, not a difference between the two readers. The reward is not a length or reason-quality check.
- The scorer records broken required fields as `format_errors` separately from unreadable JSON, and still counts their labels. The reward deliberately gives both kinds -2.0 under the exact-format rule. No tested answer rejected by either shared reader earned points.
- `GRPOReward` fits the TRL calling contract quoted in the review request: it accepts extra dataset arguments, returns one float per completion, counts answer kinds, and optionally calls `log_metric`. A `gold_type` list containing `None` works for contradiction and unclear rows. Passing `gold_type=None` for the whole batch raises a length error; the contract requires a list, one entry per completion.
- For conversational input it reads only `completion[0]["content"]`, as the supplied contract describes. Any later messages are ignored, even if invalid. Empty message lists or missing content raise errors. Step 5 should use the current single-assistant-message answer style. A multi-message training flow would need an explicit reading rule and tests first.
- Comments, whitespace or docstrings inside the four fingerprinted functions change the fingerprint even if behavior stays the same. This is conservative and acceptable, but may require an unnecessary offline recheck. It does not solve finding 1.

The supplied TRL reference is [TRL 1.15.0 GRPO documentation](https://huggingface.co/docs/trl/v1.15.0/en/grpo_trainer). I checked the wrapper against the contract supplied in the prompt. I did not fetch current TRL or Unsloth documentation because this review forbids network access. TRL is not installed in the local project environment, so an actual `GRPOTrainer` integration was not tested. This is not a claim that current outside-tool behavior was independently verified.

## Constant-answer shortcuts

The pool contains 175 contradictions, 28 updates, 42 compatible details, 43 merely related pairs, and 26 unclear pairs. On those 314 pairs:

| Answer given every time | Mean points per answer |
|---|---:|
| `contradiction` | -0.106688 |
| `unclear` | -0.375796 |
| `non_contradiction`, type `update` | -0.175159 |
| `non_contradiction`, type `compatible_detail` | -0.164013 |
| `non_contradiction`, type `merely_related` | -0.163217 |
| v1's saved sampled answers on this same pool | +0.534833 |
| A judge always giving the right label and type | +1.089968 |

None of the constant answers is close to the sampled judge's average. On a decidable pair, `unclear` earns -0.5: less than a right answer, but more than a wrong answer. That intentionally makes uncertainty preferable to a false claim, as D-043 requires. On a truly unclear pair it correctly earns +1.0. A group containing only unclear answers has equal rewards and supplies no learning signal. Keep watching the per-kind counts during Step 5; these averages alone cannot prove how training will behave.

## Saved pool and summary checks

I rescored all 8,144 raw answers and also checked their values against a separately written copy of the reward table. The separate copy used the same shared answer reader. I verified all candidate IDs, all presented versions, all 323 stored reward lists, answer counts, outcomes, original pair fields, and the always-wrong answers.

- Candidates are exactly the 971 M2 training pairs plus M1's 47 unclear training pairs.
- Outcomes: **262 mixed, 747 always right, 9 always wrong**. The fixed-seed draw chooses **52** always-right pairs, giving **314** pool pairs. All IDs are unique in these saved files.
- **7,343 of 8,144** answers have the right label, which rounds to 90%.
- Answer counts: 3,546 contradiction, 3,837 non-contradiction, 752 unclear, and 9 invalid by the reward's combined format check.
- **452 of 7,760** answers on decidable pairs say unclear, which rounds to 6%. **22** candidates always produce unclear.
- All per-trap counts, percentages, selected examples and their rewards agree with the summary. Regenerating its text matches except the billed-minutes figure: using the displayed rounded L4 charge of $0.74 gives 55.6 minutes, while the saved text says 55.2. The latter is consistent with a more precise charge that rounds to $0.74. The exact bill was not available or independently checked in this offline review.
- `tr-1527` appears only in the always-wrong list, not the pool, as D-047 requires.
- Pool SHA-256: `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77`.
- Current reward SHA-256: `a369e730de69a4305f079697c647c0bc25332d6bf5510eae10207bef65b0c70c`.
- Saved prompt SHA-256 matches the current prompt: `83d14d39a2115eb1f4d2d461581673090ec25d0a6f05d444cb2da8b756a2865c`.

`load_candidates` uses only the two named training files and reads the frozen-file list from `STATE.md`, without opening a frozen dataset. There is no validation-file input in the sampling or choosing paths. During the saved-results recheck, an in-memory access guard blocked both frozen test files and both validation files; none was requested. The branch diff contains no changes to either frozen test file. I did not read or hash their bytes, so I do not claim a fresh byte-level verification of those files.

## Settings Step 5 must preserve

The saved sampling settings are eight answers per pair, temperature 1.0, `top_p=1.0`, `top_k=0` (off), sampling enabled, one beam, and a 256-token answer limit. Keep the same pinned base and epoch-2 adapter, prompt, chat-template options, and each selected row's recorded order/date presentation when relying on this pool's evidence. Reapplying `format_pairs` to the smaller pool would draw different presentations.

Do not silently add a probability filter such as `min_p=0.1`. The existing STATE note warns that Unsloth's notebook uses it; that notebook was not fetched here. Step 5 should explicitly choose equivalent sampling settings, check any inherited model defaults, and record the effective settings. Any material change needs a fresh training-only pool check, not a look at either test set. The raw generation settings do not explicitly set `min_p`, so the stored record alone cannot prove every inherited setting was disabled.

## Tests and limits

The new files pass:

```text
PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 UV_CACHE_DIR=/tmp/m2-04-uv-cache \
uv run --offline --no-sync pytest -p no:cacheprovider tests/test_reward.py tests/test_pool.py
77 passed in 0.27s
```

For the broader run, I invoked pytest through `uv run --offline --no-sync python` with an in-memory guard against protected-file reads and network connections, disabled bytecode/cache writes, and excluded tests that read the real frozen M1 test file. Result: **215 passed, 13 deselected, 12 setup errors**, out of 240 collected tests.

- Seven excluded cases are `test_m1_saved_answers_rescore_to_the_same_numbers` in `tests/test_score.py`.
- Six excluded cases in `tests/test_prepublish_check.py` are `test_real_cards_pass`, `test_wrong_score_in_a_table_is_caught`, `test_unknown_row_in_a_score_table_is_caught`, `test_wrong_count_is_caught`, `test_untraced_number_in_prose_is_caught`, and `test_every_fingerprint_in_the_cards_is_known`.
- The 12 setup errors are existing `tests/test_run_ollama.py` cases whose fake local HTTP server cannot bind a socket in this sandbox: `PermissionError: [Errno 1] Operation not permitted`. They did not reach their test assertions. They are an environment limitation, not evidence of a branch regression. No network connection was made.

I therefore cannot confirm the earlier report of all 240 tests passing in this environment. `git diff --check main...HEAD` passes. No GPU or Modal job ran, no branch switched, and no source, state, glossary, lesson, dataset or saved result file was edited. The only checkout file written is this report; test fixtures used temporary folders. The prompt's read-only rule takes precedence over the normal commit/push close-out rule.

## Done and next action

1. **Codex: finished.** Reviewed the requested commit and every changed file, checked saved results, reproduced three must-fix gaps, and wrote `/Users/frank/code/model-training/reviews/M2-04-codex-review.md`.
2. **Elias: next.** In Claude Desktop, continue the Cowork project **Fine Tuning and RL training** and provide this report's absolute path. Ask it to route the three fixes to Claude Code's desktop Code panel in `/Users/frank/code/model-training`, branch `m2-04-pool-and-reward`.
3. **Claude Code, then Codex:** fix the checks and add temporary-data regression tests, then obtain a new review before Step 5. Preserve the saved samples and pool. These fixes need no GPU run, test-set access, or reward-value change. Cowork retains merge responsibility. No message was sent to another app or chat from this review.
