# M2-04 review fixes

Fixes for `reviews/M2-04-codex-review.md` (Codex, 2026-10-27, CHANGES REQUIRED, three must-fix findings). Done by Claude Code on branch `m2-04-pool-and-reward`, 2026-10-27. No GPU or Modal runs; no test set or validation file read. Tests: 253 pass with `uv run pytest` (240 before). Every new test was also run against the old code and failed there, so each one really catches its problem.

I agree with all three findings. No reward value, no saved sample, no pool row and no summary changed.

| # | Finding | What changed | Tests that cover it |
|---|---|---|---|
| 1 | The reward fingerprint missed the reader's helpers and constants | `src/judge/reward.py`: `reward_sha256()` now starts from `reward` and `completion_text` and follows every module-level name their code uses, at any depth (new `rule_parts` and `describe`). So it now covers the table, labels, types, the code-fence pattern, `reject_constant`, `no_repeated_fields`, `RepeatedField`, `parse_answer`, `format_problem`, `read_answer` and how a training completion is read. A helper added later is covered without anyone listing it. A value it cannot describe stops with an error rather than being skipped. The rules did not change, but the fingerprint did: `a369e730...b0c70c` became `92a043f1...c3a53`. The offline recheck is in `results/m2/pool/reward-recheck.md`: the old method on today's code still gives `a369e730...`, and all 8,144 saved answers rescore to the 323 stored reward lists. | `tests/test_reward.py`: `test_fingerprint_changes_with_the_answer_reader` (4 cases: `update` removed from the types, repeated fields allowed, `NaN` allowed, a different fence; each also proves the reward itself changes), `test_fingerprint_changes_with_how_a_completion_is_read`, `test_fingerprint_covers_the_whole_reader` |
| 2 | Pool selection accepted samples made with another prompt | `scripts/sample_pool.py` `choose_pool`: before reading the samples or writing anything, it stops if the recorded prompt fingerprint differs from today's, or if the recorded show seed, keep seed or answers per pair differ from `judge.pool`. It now passes `SHOW_SEED` to `show` itself rather than relying on the default. | `tests/test_pool.py`: `test_choose_stops_if_the_prompt_changed`, `test_choose_stops_if_a_pool_setting_changed` (3 cases); each also checks no output was written |
| 3 | A repeated sample ID could hide a missing candidate | `scripts/sample_pool.py`: new `check_sampled_ids`, called by `choose_pool` before judging or writing. A full run must hold every candidate exactly once; a smoke run exactly the first 16 shown candidates. It stops and names any repeated, missing or unexpected IDs. This replaces the row-count check. | `test_choose_stops_on_a_repeated_id_that_hides_a_missing_one` (Codex's case: row 2 replaced by a copy of row 1), `test_choose_stops_on_a_missing_or_unknown_id`, `test_a_smoke_run_must_hold_exactly_the_smoke_ids`; each also checks no output was written |

## Other notes in the review, and what happens to them

- **Comments change the fingerprint.** Codex called this acceptable; it still holds. A comment edit inside the reward code means a quick offline recheck like `reward-recheck.md`, never a GPU run.
- **`gold_type=None` for a whole batch** raises an error. That is intended: TRL passes one value per completion, so a missing column should stop, not guess.
- **Only the first message of a conversational completion is read.** Step 5 uses one assistant message per answer, as Codex advises. The fingerprint now covers `completion_text`, so any change to this rule shows.
- **Settings Step 5 must keep:** temperature 1.0, top_p 1.0, top_k off, no `min_p`, the same base and epoch-2 adapter, prompt, chat-template options, and each pool row's recorded version (swapped, dates). Step 5 must use each row's saved `swapped` and `dates`, not run `format_pairs` again. Recorded in `STATE.md` for Step 5.
- **Codex's 12 setup errors** in `tests/test_run_ollama.py` came from its sandbox, which blocks local sockets. On the Mac all 253 tests pass.
- **The billed-minutes rounding** (55.2 against 55.6) comes from the summary using the exact L4 charge, then showing it rounded to $0.74. Not a mistake; left as is.

## Second review: fixes for `reviews/M2-04b-codex-review.md`

Codex's second review (2026-10-29, CHANGES REQUIRED, two must-fix findings) was about the reward fingerprint only. Done by Claude Code on branch `m2-04-pool-and-reward`, 2026-10-30, following D-048 and `prompts/M2-04c-fingerprint-fixes.md`. No GPU or Modal runs; no test set or validation file read. Tests: 307 pass with `uv run pytest` (253 before). No reward value, saved sample, pool row or summary changed.

I agree with both findings. The name walk (`rule_parts`, `describe`) is gone. `reward_sha256()` is now the SHA-256 of the full text of the source files, each with its path, in sorted order. Those files are found by reading the imports in the code with Python's `ast` module, starting from `judge/reward.py` and `judge/__init__.py`. Final fingerprint: `4d64ca4c54a353e3cb60522cfd49ed5dd38a14ade5ec55f8e87a98ee15f8a42a`.

| # | Finding | What changed | Tests that cover it (`tests/test_reward.py`) |
|---|---|---|---|
| 1 | Default values, captured values and helpers reached through a module could change points without moving the fingerprint | All of them are text in a covered file, so any change to them moves the fingerprint. `judge` modules are followed through their imports, including imports inside a function and relative imports, with no list to keep. | `test_fingerprint_changes_with_a_default_value` (Codex's `STRICT` case), `test_fingerprint_changes_with_a_captured_value`, `test_fingerprint_changes_with_a_helper_reached_through_a_module` (6 import forms), `test_fingerprint_follows_a_module_reached_without_importing_it`, `test_fingerprint_follows_a_relative_import_two_levels_up`, `test_a_module_added_later_is_covered_without_a_list`, `test_a_module_imported_by_a_later_module_is_covered`, `test_any_edit_to_a_covered_file_moves_the_fingerprint` |
| 2 | `GRPOReward.__call__` was outside the fingerprint | It is in `judge/reward.py`, so a change to it now moves the fingerprint. | `test_fingerprint_changes_with_the_training_wrapper`: doubling the points moves +1.0 to +2.0 and moves the fingerprint, while `reward` stays the same |
| | Silent gaps | The fingerprint now stops with an error instead of skipping: `importlib`, `builtins`, `__import__`, `exec`, `eval`; `import judge` alone; a star import from `judge`; an import with no source file (a `judge` subfolder needs an `__init__.py`); a relative import that leaves `judge`; project code beside `judge`. | `test_code_that_cannot_be_followed_stops_with_an_error` (14 forms in 2 files), `test_a_star_import_from_the_package_stops_with_an_error`, `test_project_code_beside_the_package_stops_with_an_error` |

**Proof the tests work.** 46 of the tests fail on the old `reward.py`, for the right reason: the points change but the fingerprint stays the same, or no error is raised. Most use copies of the real code in a temporary folder, loaded in a fresh Python, so the same test runs on the old and new code. Two kinds of test behave differently, on purpose:
- `test_fingerprint_is_stable_and_does_not_depend_on_the_folder` passes on the old code too, because the old fingerprint was also stable. So it cannot fail there. To prove it can catch a problem, I ran it on a scratch copy that hashes full folder paths, and it failed.
- The 14 rewritten older tests (each reward value, the reader, how a completion is read) pass on old and new code. They used to change the code in memory, which no longer moves the fingerprint, so they now edit file copies.

**What it found beyond the prompt's list.** `judge/__init__.py` is a fifth covered file, because Python runs it on any `judge` import. The other four are as expected: `judge/reward.py`, `judge/score.py`, `judge/prompt.py`, `judge/frozen.py`.

**What is covered, exactly.** The full text of those five files, plus any `judge` module they import later.

**What is not covered:**
- Python itself and outside packages such as `json` and `re`. Their versions stay in `uv.lock` and the pinned image.
- Data files or environment settings that the code might read. Today the reward code reads none; `load_frozen` reads `STATE.md`, but no reward code calls it.
- Changes made in memory while a program runs. This is new: the old fingerprint saw some of these, and the new one reads files on disk when called. `test_fingerprint_reads_files_not_memory` records this limit. Step 5 must not replace reward code in memory, and must not edit the files during a run.
- Code that wraps the reward from outside these files, such as Step 5's training script.
- Code hidden on purpose, such as building the word `eval` from pieces. The stops catch plain uses; this is a check against mistakes, not against sabotage.

**Correction to the table at the top of this file.** Row 1 says "A helper added later is covered without anyone listing it." That was too broad: Codex showed it was false for default values, captured values, helpers reached through a module and `GRPOReward.__call__`. The row is kept as written, as the record of that fix. This section replaces it. The same goes for `A value it cannot describe stops with an error`: that code is gone, and the stops are listed above.
