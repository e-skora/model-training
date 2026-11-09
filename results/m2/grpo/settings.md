# M2 Step 5: GRPO settings, and what the tools really do

Written 2026-11-04 by Claude Code (prompt M2-05). Part A checks the tools against the exact versions our Modal image installs. Part B lists each training setting and where it comes from. All settings live in one place: `GRPO` in `scripts/train_grpo.py`.

**How the tools were checked.** A small CPU-only Modal job (no GPU, under one cent) copied the installed source files out of our pinned image (`src/judge/modal_image.py`). It reported these versions: TRL 1.15.0, Unsloth 2026.10.7, unsloth_zoo 2026.10.7, transformers 5.17.0, peft 0.21.2, torch 2.14.1. The line numbers below are in those installed files. The copies were kept in a scratch folder, not in git.

## Part A: what the tools do

### 1. TRL 1.15.0 `GRPOConfig` defaults: confirmed

All 15 values Claude in Cowork read on 2026-11-02 match `trl/trainer/grpo_config.py`:

| Setting | Default | Line |
|---|---|---|
| `learning_rate` | 1e-6 | 422 |
| `num_generations` | 8 | 479 |
| `max_completion_length` | 512 | 493 |
| `temperature` | 1.0 | 527 |
| `top_p` | 1.0 | 531 |
| `top_k` | 0 | 538 |
| `min_p` | None | 545 |
| `repetition_penalty` | 1.0 | 568 |
| `use_vllm` | False | 582 |
| `beta` | 0.0 | 676 |
| `num_iterations` | 1 | 684 |
| `epsilon` | 0.2 | 688 |
| `scale_rewards` | "group" | 784 |
| `loss_type` | "dapo" | 796 |
| `log_completions` | False | 980 |

Two more facts from the same file. `max_prompt_length` no longer exists in TRL 1.15.0, so prompts are never cut. `mask_truncated_completions` defaults to False (line 831).

### 2. Unsloth's patch, and whether the sampling stays ours

**The claim checked:** `_caller_sampling_only` in `unsloth/models/rl.py` stops the model's own settings (for Qwen3: top_p 0.8, top_k 20, temperature 0.7) from replacing TRL's.

**What the code shows:**
- `_caller_sampling_only` (line 203) runs only when the model doing the generating is a wrapper, not the model itself (line 282: `if generating_model is not unwrapped_model`). That is PPO's case, where TRL generates through `.policy` (line 276).
- **On our path (GRPO, no vLLM) it does not run.** This corrects the note it came from. Our sampling is still kept, for a different reason, below.

**What really keeps TRL's sampling on our path:**
1. Without vLLM, TRL builds its own `GenerationConfig` from the config: `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` and `max_new_tokens` (`trl/trainer/grpo_trainer.py` lines 1116 to 1131). It passes that to `generate`.
2. In transformers 5.17.0, `generate` fills in a field from the model's own settings only when that field is empty (`None`) in the config it was given (`transformers/generation/utils.py` lines 1929 to 1937, `defaults_only=True`). TRL sets temperature 1.0, top_p 1.0 and top_k 0 itself, so the model's 0.7, 0.8 and 20 never get in.
3. `min_p` is empty (None). The model's file could fill it. The base model's `generation_config.json` (`unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at `7744afa8...`) has no `min_p`, so it stays off. It holds temperature 0.7, top_k 20 and top_p 0.8, which do not get in (point 2).
4. TRL's own override of the model's settings (`trl/models/utils.py` lines 174 to 181) is skipped on transformers 5 or newer, because transformers fixed the problem itself.
5. Unsloth's own `generate` (`unsloth/models/llama.py` line 2721 onward) changes the cache, the padding token and the logits it keeps, never the sampling.
6. Which sampling steps run: transformers adds a temperature step only if temperature is not 1.0, top_k only if not 0, top_p only if below 1.0, and min_p only if set (`utils.py` lines 1440 to 1455). With our settings none is added, so the model samples from its plain probabilities. The pool did the same, with `generate(**SAMPLING)`.

**So the sampling matches the pool, on paper.** The trial checks it for real. The GPU job records the settings `generate` ended up with and the sampling steps it added, and stops at the first answer if either differs from the pool's. It also checks that every prompt is one of the pool's 314 prompts, token for token, built the way the pool built them.

**Tightened after Codex's review (2026-11-08, `reviews/M2-05-fixes.md`):**
- The checks now also watch `generate` itself and TRL's generation step.
- They allow no word filter at all, and only the length limit and the end token as stop rules.
- They need exactly one answer per prompt copy and 8 copies of one pool prompt per group.
- After every step, the counts at all five places must agree, and the reward must have scored every answer.
- A full run must give every pool pair exactly 2 groups.

**Tightened after Codex's re-review (2026-11-09, `reviews/M2-05b-fixes.md`):**
- `generate` must get exactly TRL's prompt tokens, in order: each row's tokens under its attention mask must equal TRL's row. TRL 1.15.0 left-pads the prompts and passes `input_ids` and `attention_mask` by name (`trl/trainer/grpo_trainer.py` lines 1769 to 1812); Unsloth's own `generate` passes them on unchanged (`unsloth/models/llama.py` lines 2732 to 2775).
- The length limit must be the padded prompt width plus 256 (transformers 5.17.0 sets it so, `generation/utils.py` line 1863).
- The end tokens must be the pool's: 151645 and 151643 (`<|im_end|>`, `<|endoftext|>`). The pool's `generate` passed none of its own, so it used the pinned base's `generation_config.json` (revision `7744afa8...`). TRL builds its list from the same file and adds the tokenizer's end token, which is already in it (lines 398 to 405).
- One `generate` call per TRL generation step, and no answer longer than 256 tokens.
- All of this is read from the pinned source and the base's files, not yet seen on a GPU. If the real call differs, the run stops during step 1 (about $0.05).

**Unsloth's other changes that matter here** (`unsloth/models/rl.py` lines 3238 to 3296, 3434 to 3491): Unsloth replaces some of TRL's defaults with its own, for example learning rate 5e-5, weight decay 0.001, warmup 0.1, `per_device_train_batch_size` 4 and `gradient_accumulation_steps` 2. With `loss_type` "dapo" it also turns `mask_truncated_completions` on by itself (lines 3474 to 3476), and sets `epsilon_high` to 0.28 when it is empty (lines 3477 to 3479). So the script sets every one of these values itself, and the record saves the values the trainer really ended with (`resolved_config`).

**Found in the trial (2026-11-04):** the first script left `epsilon_high` empty, and the trial's log says "Unsloth: The DAPO paper recommends `epsilon_high = 0.28` - we will set it." So the trial ran with 0.28, which its `trial.meta.json` does not show under `grpo` (the full log is `trial-output.log`, line 36). The script now sets 0.28 itself, so the full run uses the trial's value and its record shows it. A test checks that every value Unsloth would fill in is set in the script.

**Also found in the trial:** TRL writes a pair's 8 answers by putting the prompt in the batch 8 times, each with one answer (`num_return_sequences` 1). The pool asked for 8 answers to one prompt (`num_return_sequences` 8). Each answer is drawn the same way in both, from the same settings, so they come from the same spread of answers. The order of the random draws differs, so single answers do not repeat the pool's.

### 3. Unsloth's GRPO notebook for Qwen3

Notebook: `nb/Qwen3_(4B)-GRPO.ipynb` in `unslothai/notebooks`, last changed at commit `fd03092cdf9e95a159cba0ad0b93579c8ef2d388` (2026-10-03), https://github.com/unslothai/notebooks/blob/fd03092cdf9e95a159cba0ad0b93579c8ef2d388/nb/Qwen3_(4B)-GRPO.ipynb

| Setting | Notebook | Ours |
|---|---|---|
| Model | `unsloth/Qwen3-4B-Base`, 16-bit, vLLM on | v1 (4-bit base + v1's adapter), no vLLM |
| LoRA | r 32, alpha 64, all 7 projection layers, a new adapter | v1's adapter: r 32, alpha 32, the same 7 layers, kept |
| `learning_rate` | 5e-6 | 5e-6 |
| `optim` | adamw_8bit | adamw_8bit |
| `weight_decay` | 0.001 | 0.001 |
| warmup | `warmup_ratio` 0.1 | 10% of the steps, as a whole number |
| `lr_scheduler_type` | linear | linear |
| `max_grad_norm` | not set, so transformers' 1.0 (`training_args.py` line 864) | 1.0, set |
| batch | `per_device_train_batch_size` 1, `gradient_accumulation_steps` 1 | 8 and 2 (below) |
| `num_generations` | 4 | 8 |
| steps | `max_steps` 100 | 314 (below) |
| sampling | `min_p` 0.1, top_p 1.0, top_k off, temperature 1.0 (vLLM) | `min_p` off, the rest the same |

Unsloth's guide says: "Wait for at least **300 steps** for the reward to actually increase", and that it can take 1,000 or more (https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide). It also says RL works without vLLM: set `fast_inference=False`. Its advanced page (https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation) gives no learning rate, LoRA, `optim` or `max_grad_norm` advice. It lists `loss_type` "grpo" as not recommended because of a length bias. It says `min_p` is usually 0.01 to 0.2, with no default.

### 4. Keeping v1's adapter training, rather than adding a new one

- **Unsloth's own way** is `FastLanguageModel.from_pretrained(model_name=<adapter repo>)`. Its loader then loads the base and attaches the adapter with `PeftModel.from_pretrained(..., is_trainable=True)` (`unsloth/models/loader.py` lines 1919 to 1925), then runs `patch_peft_model` (line 1939).
- **Its catch for us:** on that path the `revision` names the adapter, and the base is loaded with no revision (lines 1766 to 1772). So the base commit would not be pinned.
- **What the script does instead** is the same three steps, with the base pinned:
  1. `FastLanguageModel.from_pretrained(BASE_REPO, revision=BASE_REVISION, load_in_4bit=True, fast_inference=False)`;
  2. `PeftModel.from_pretrained(model, V1_ADAPTER_REPO, revision=V1_ADAPTER_REVISION, is_trainable=True)`;
  3. `FastLanguageModel.patch_peft_model(model, use_gradient_checkpointing="unsloth")` (`unsloth/models/llama.py` line 4914).
- `get_peft_model` is not called. On a model that already has the same LoRA, Unsloth only says "Already have LoRA adapters! We shall skip this step." (`llama.py` line 4446). Calling it would add nothing, and with other settings it could add a second adapter.
- **Checks before training:** the adapter's settings must be r 32, alpha 32, dropout 0 and the 7 layers. Only LoRA weights may be trainable. Every weight must equal v1's `adapter_model.safetensors` at `1d9ebbf6...`, with a largest difference of exactly 0.

## Part B: the settings chosen, and why

| Setting | Value | Source |
|---|---|---|
| Start point | base `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at `7744afa8566e264af1a92a806d8d9aae00cc7c78`; adapter `eskora/m1-contradiction-judge-qwen3-4b-lora` at `1d9ebbf66e1a2abcad9932932bb7d810a38c59b9` (epoch 2), trainable | D-031, D-037; Part A item 4 |
| Answer format | Style A: v1's exactly; same prompt (`83d14d39...2865c`) | D-040 |
| Data | all 314 pool rows, each in its saved version | prompt M2-05; pool SHA-256 `2c5c3bf2...dac4e77` |
| Reward | `GRPOReward()`, unchanged, fingerprint `769f58fe...025e7` | D-049 to D-054 |
| `num_generations` | 8 | the pool (D-046) |
| `max_completion_length` | 256 | the pool (`judge.generate.SAMPLING`) |
| `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` | 1.0, 1.0, 0 (off), None (off), 1.0 | the pool; TRL defaults; the notebook's `min_p` not copied (prompt M2-05) |
| `use_vllm` | False | the pool did not use vLLM; Unsloth's guide says RL works without it |
| `per_device_train_batch_size` | 8 | one pair's 8 answers per micro-batch. The pool's runs showed 4 prompts x 8 answers reached 19.4 GiB of the L4's 22 GiB on the longest prompts (LESSONS 2026-10-27), so each generate call here makes 16 answers |
| `gradient_accumulation_steps` | 2 | 2 pairs (16 answers) per step, a group average over 2 pairs. TRL makes the generation batch 8 x 2 = 16 (`grpo_config.py` lines 1085 to 1087) |
| Steps | 314 = 2 epochs of 157 steps | Unsloth's guide: at least 300 steps. 2 epochs means each pair is seen twice, not more, on a small pool |
| `learning_rate` | 5e-6 | the notebook. TRL's 1e-6 is meant for full training; LoRA usually needs more. Our alpha/r is 1, half the notebook's 2, so the same rate already moves the weights less |
| `lr_scheduler_type`, warmup | linear, 31 steps (10%) | the notebook |
| `optim` | adamw_8bit | the notebook; M1 used it too |
| `weight_decay` | 0.001 | the notebook; Unsloth's default |
| `max_grad_norm` | 1.0 | transformers' default, which the notebook uses; set explicitly so no other default can change it |
| `beta` | 0.0 | TRL's and Unsloth's default for "dapo". No reference model is loaded (less memory). The checks against forgetting are the 52 always-right pairs in the pool (D-046) and D-056's M1 validation limit |
| `loss_type` | "dapo" | TRL's default; Unsloth's docs: "grpo" has a length bias, not recommended |
| `epsilon`, `num_iterations`, `scale_rewards` | 0.2, 1, "group" | TRL defaults |
| `epsilon_high` | 0.28 | Unsloth sets it for "dapo" when empty (`rl.py` lines 3477 to 3479), and the trial ran with it; Unsloth's advanced docs: "DAPO recommends **0.28**". It lets a good but unlikely answer gain a bit more per step than `epsilon` alone would. Set here so the record shows it |
| `mask_truncated_completions` | False | TRL's default. Set on purpose, because Unsloth would turn it on for "dapo". A cut-off answer is invalid JSON and must still cost -2.0, or the model would not learn to stop |
| Gradient checkpointing | "unsloth" | as M1 |
| Seed | 3407 | as M1 and the notebook |
| `logging_steps` | 1 | so the reward-hacking watch reads each step |
| Checkpoints | adapter every 50 steps and at the end: 50, 100, 150, 200, 250, 300, 314 | prompt M2-05; D-056 chooses among them and step 0 |
| Reward-hacking watch | over the last 20 steps: invalid > 2%, any one label > 85%, `unclear` > 15% | prompt M2-05 |

Nothing here was tuned on any result. The validation files have not been answered by any trained checkpoint. The trial only proves a checkpoint loads and answers in v1's format.
