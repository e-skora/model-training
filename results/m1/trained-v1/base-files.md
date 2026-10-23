# Which base files did the untrained baseline load?

Checked 2026-10-22 for Codex review finding 4 (`reviews/M1-07-codex-review.md`). No GPU was used. Two sources were read: the public Hugging Face history of the two repos, and the source code of the pinned Unsloth version.

**Answer.** The baseline almost certainly ran on the same base files as training: the 4-bit copy `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at commit `7744afa8566e264af1a92a806d8d9aae00cc7c78`. This follows from the pinned Unsloth code and the repo's history. It was not seen in the baseline run itself: that run recorded neither the repo it really loaded nor its commit, and its log was not kept. So it is a well-supported inference, not a verified fact.

## 1. Hugging Face history of the 4-bit copy

Read from the public API (no token sent):
- https://huggingface.co/api/models/unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit/commits/main
- https://huggingface.co/api/models/unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit/refs

| Commit | Date (UTC) | Title |
|---|---|---|
| `7744afa8566e264af1a92a806d8d9aae00cc7c78` | 2025-08-06 21:36:32 | Upload folder using huggingface_hub |
| `f62e267f33116d9131515bcc0814736f50552ee0` | 2025-08-06 21:36:13 | initial commit (empty) |

The repo has one branch, `main`, pointing at `7744afa8...`, and no tags. Its weights file `model.safetensors` is 3,545,956,414 bytes, SHA-256 `bcd75cba3491304d5aad317661dbd9201a25d40f135a2adefcd5d040da8078dc`.

**So yes:** `7744afa8...` was the latest commit on `main` on 2026-10-22, when the baselines ran (baseline v2 was committed at 2026-10-22 15:11 -0700). Nothing was added after 2025-08-06. This is the history as the API shows it today.

The original repo `unsloth/Qwen3-4B-Instruct-2507`, whose commit the baseline record names, has three commits, all on 2025-08-06. The newest, `992063681dc2f7de4ee976110199552935cad284` (21:35:26 UTC), is the one in `results/m1/baseline-v2/qwen3-4b-instruct-2507.meta.json`.

## 2. How Unsloth 2026.10.7 maps the name

Source: the wheel `unsloth-2026.10.7-py3-none-any.whl` from PyPI (https://pypi.org/project/unsloth/2026.10.7/), SHA-256 `3ffd1dc001d799eb4ec5a955852f41a3505febb9075504bb2f64a1621c9c7c4e`, read without running it. The baseline record says this version ran (`"unsloth": "2026.10.7"`, `"transformers": "5.17.0"`).

The baseline called `FastLanguageModel.from_pretrained(model_name="unsloth/Qwen3-4B-Instruct-2507", load_in_4bit=True, revision="99206368...")` (`scripts/run_baseline.py`, lines 57 to 58). In that version:

1. `unsloth/models/mapper.py`, line 1060: the entry for `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` lists `unsloth/Qwen3-4B-Instruct-2507` among its 16-bit names. `build_mappers` (line 1400) turns every listed name into a key of `FLOAT_TO_INT_MAPPER` that points back to the 4-bit repo.
2. `unsloth/models/loader.py`, line 1140: `use_exact_model_name` defaults to `False`. Lines 1297 to 1300: when it is `False`, the name goes through `get_model_name`.
3. `unsloth/models/loader_utils.py`, lines 1047 to 1053: with `load_in_4bit` on (and transformers 4.37 or newer, line 49), a name found in `FLOAT_TO_INT_MAPPER` that does not already end in `-bnb-4bit` is replaced by the 4-bit repo. The name is not in `BAD_MAPPINGS` (line 56), so nothing changes it again.
4. `unsloth/models/loader.py`, lines 218 to 237 (`_revision_for_resolved_repo`): when the repo name changed, the `revision` is dropped, with the warning "Ignoring revision ... since ... resolved to ...". Lines 1392 to 1393 apply it. With no revision, the files come from the default branch, `main`.

## 3. What this proves, and what it does not

Proved: with the code and settings the baseline recorded, Unsloth 2026.10.7 loads `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` from `main`, and on 2026-10-22 `main` was `7744afa8...`, the commit training and the test run pinned by name.

Not proved: that the baseline run really took this path. It did not record `model.config._name_or_path`, the name of what was really loaded (Step 7's scripts now do), and its log was not kept. Something the record does not show, such as an environment setting, could in principle have changed the path. Matching prompt fingerprints prove the prompts matched, not the weights. Only a rerun that records the loaded name would turn the inference into an observation; none was done, and none is needed for the Step 7 result.
