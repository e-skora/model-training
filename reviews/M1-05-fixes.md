# M1-05 review fixes

Fixes for `reviews/M1-05-codex-review.md` (Codex, 2026-10-22, CHANGES REQUIRED). Done by Claude Code on branch `m1-05-baseline`, 2026-10-22. No GPU or Modal runs. Tests: 33 pass with `uv run pytest` (7 before).

| # | Finding | What changed | Test that covers it |
|---|---|---|---|
| 1 | How invalid answers affect the scores | Kept standard precision and recall (D-024). An invalid answer is neither "yes" nor "no": it lowers recall when the right label is contradiction and lowers accuracy, but never enters precision. The summary and glossary now say exactly that instead of "counts as wrong", and the invalid count stays next to precision and recall. No extra penalised measure was added: D-024 rules it out. | `tests/test_score.py::test_invalid_answers_lower_recall_and_accuracy_not_precision` (correct contradictions, a false alarm, a miss, an invalid answer whose right label is contradiction, and one whose right label is non-contradiction) |
| 2 | Scorer did not check the frozen file | New `src/judge/frozen.py`: `load_frozen` reads the file's bytes once, checks their SHA-256 against the "Frozen files" entry in `STATE.md`, and stops if they differ or the file is not listed. The scorer command and `scripts/run_baseline.py` both use it; the fixed value in the runner is gone. `score()` still takes made-up pairs. | `tests/test_frozen.py` (5 tests, temporary folder), including `test_one_changed_byte_stops_loading` and `test_scorer_command_stops_before_scoring_on_a_changed_byte` |
| 3 | `NaN`, `Infinity` and repeated fields accepted | The answer reader rejects `NaN`, `Infinity` and `-Infinity` ("not valid JSON") and any repeated field name ("repeated field", its own count `invalid_repeated_field`). | `test_non_json_constants_are_invalid` (3 cases), `test_repeated_field_names_are_invalid` |
| 4 | No check of the full answer format | New `format_errors` count: answers with a readable label whose `type` is not JSON `null` for a contradiction or not one of the three types otherwise, or whose `reason` is not non-empty text. Its own column in the summary; label scores unchanged. | `test_format_problem` (10 cases), `test_format_errors_are_counted_but_do_not_change_label_scores` |
| 5 | Repeated and unknown answer IDs | `score()` stops with a clear message on a repeated ID or an ID not in the test set. A missing answer still counts as wrong. | `test_repeated_answer_ids_stop_scoring`, `test_unknown_answer_ids_stop_scoring`, `test_missing_answers_still_count_as_wrong` |
| 6 | Run settings not saved | New `src/judge/generate.py` holds the answering loop and the generation settings, with `num_beams=1` set explicitly. Each run's `.meta.json` now saves the generation settings, chat-template options, the model and tokenizer revision (the Hugging Face commit looked up before loading and passed to Unsloth as `revision`), and the SHA-256 of the first prompt. The GPU job was not rerun, so this path has run only with fakes. The summary says the saved runs predate these records. | `tests/test_generate.py` (fake model and tokenizer: settings passed to every batch, template options, prompt fingerprint) |

## Rescore (no GPU)

The three saved answer files were rescored with the new scorer. Precision, recall and accuracy are exactly as before:

| Model | Precision | Recall | Accuracy | Invalid | Format errors |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 83/89 | 83/100 | 175/200 | 10 | 0 |
| Qwen3-8B | 94/98 | 94/100 | 190/200 | 0 | 0 |
| Phi-4-mini-instruct | 85/95 | 85/100 | 175/200 | 0 | 90 |

No answer had `NaN`, `Infinity` or a repeated field, and every file had 200 unique IDs matching the test set.

## Checked against outside docs
- Current transformers no longer keeps the downloaded commit on the loaded config or tokenizer (`_commit_hash` is dropped): https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py and `tokenization_utils_base.py`. So the commit is looked up first instead.
- `HfApi.model_info(repo).sha` is the "Repo SHA at this particular revision": https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py
- `FastLanguageModel.from_pretrained` takes `revision` and passes it to the model and the tokenizer: https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py. Checked on Unsloth's main branch, not on the pinned 2026.10.7, and not on a GPU.
