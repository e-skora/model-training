---
license: apache-2.0
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
language:
- en
datasets:
- eskora/memory-contradiction-pairs-hard
- eskora/memory-contradiction-pairs
tags:
- lora
- qlora
- unsloth
- qwen3
- grpo
- reinforcement-learning
- contradiction-detection
---

# Memory contradiction judge v2 (Qwen3-4B, GRPO): an experiment that missed its bar

**This is an experiment that missed the bar set before scoring, by one pair.** On the hard test set it needed to catch 139 of 150 real contradictions in its weakest version, and it caught 138. So it is not called better than v1.

**v1 is still the recommended model:** [eskora/m1-contradiction-judge-qwen3-4b-gguf](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-gguf) (one file for Ollama on a laptop) and [eskora/m1-contradiction-judge-qwen3-4b-lora](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-lora) (its adapter).

What the experiment taught, and how the second attempt uses it: [What I learned from training a small judge model with reinforcement learning](https://github.com/e-skora/model-training/blob/main/docs/m2-what-we-learned.md).

Made by [Elias Skora](https://www.linkedin.com/in/eliassskora/), who builds systems with language models and experiments with fine-tuning different types of models.

## What it does

The same job as v1, with the same prompt and the same answer format: you give it a person's name and two memories, each with the date it was written if you know it, and it answers with one line of JSON that says whether they contradict each other.

Input:

```
Person: Baraka Otieno
Memory A (written 2025-05-03): Baraka's flat is in Kisumu.
Memory B (written 2025-12-14): Baraka moved to Mombasa for a new job and lives there now.
```

Its answer (this exact text, from the test run on a GPU):

```json
{"label": "non_contradiction", "type": "update", "reason": "The newer memory says Baraka moved to Mombasa and lives there now."}
```

The labels are `contradiction`, `non_contradiction` (with a type: `update`, `compatible_detail` or `merely_related`) and `unclear`. The exact prompt, its SHA-256 `83d14d39a2115eb1f4d2d461581673090ec25d0a6f05d444cb2da8b756a2865c` and what each label means are on [v1's adapter card](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-lora). They did not change.

## How it was trained

- **Start:** v1's adapter, [eskora/m1-contradiction-judge-qwen3-4b-lora](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-lora) at commit `1d9ebbf66e1a2abcad9932932bb7d810a38c59b9`, on Unsloth's 4-bit copy of Qwen3-4B-Instruct-2507. Training kept changing that same adapter; no new one was added.
- **Method:** reinforcement learning with **GRPO** (group relative policy optimization), using TRL 1.15.0 and Unsloth 2026.10.7 on one NVIDIA L4 GPU rented from [Modal](https://modal.com). For each pair, the model wrote 8 answers. A small program, the reward function, scored each one, and training pushed the model toward the answers that scored above the group's average.
- **The reward:** right label +1.0, and +0.25 more for the right type; a false alarm (saying "contradiction" when it is not) -1.5; a missed contradiction -1.0; `unclear` on a pair that can be decided -0.5; an answer not in the exact format -2.0. A false alarm costs the most among wrong labels, because it interrupts a person. The code: [src/judge/reward.py](https://github.com/e-skora/model-training/blob/main/src/judge/reward.py).
- **Practice pairs:** 314 pairs from [eskora/memory-contradiction-pairs-hard](https://huggingface.co/datasets/eskora/memory-contradiction-pairs-hard) (`rl-pool.jsonl`): 262 where v1's 8 answers disagreed, and 52 it always got right, so it would not forget them. GRPO learns nothing from a pair whose 8 answers all score the same.
- **Settings:** 314 steps of 2 pairs with 8 answers each (every pair twice), learning rate 5e-6, temperature 1.0, no KL penalty (`beta` 0.0), loss `dapo`, clipping 0.2 and 0.28. Every setting and where it comes from: [results/m2/grpo/settings.md](https://github.com/e-skora/model-training/blob/main/results/m2/grpo/settings.md).
- **Time and cost:** 71.3 minutes, $0.97 billed for the training run. All of M2, every run included, cost about $3.21 to $3.72.

## How this checkpoint was chosen

A copy was saved every 50 steps and at the end. A rule written before training chose among them, using only the validation pairs: any candidate with a broken answer, or with precision that fell too far below v1's, was out; then the highest worst-case recall won, then precision, then the earlier step. **Step 250 won.** Steps 300 and 314 tied with it exactly, a sign the last steps taught little. This repo holds step 250, at commit `0964d031073408707397ce7438fc7df7b8a772c0`.

## Results

Scored once on each frozen test set, by the same script as v1, adapter against adapter. Each test has three versions: **as is**, **swapped** (newer memory first) and **no dates**. **Precision** is how often "contradiction" was right when the model said it. **Recall** is how many real contradictions it caught.

| Model | Test set | Version | Caught / said | Precision | Recall | Unclear answers | Invalid answers |
|---|---|---|---|---|---|---|---|
| v1 | hard | as is | 138 / 144 | 0.958 | 0.920 | 12 | 0 |
| v1 | hard | swapped | 136 / 142 | 0.958 | 0.907 | 15 | 0 |
| v1 | hard | no dates | 134 / 142 | 0.944 | 0.893 | 15 | 0 |
| v2 | hard | as is | 141 / 145 | 0.972 | 0.940 | 6 | 0 |
| v2 | hard | swapped | 138 / 143 | 0.965 | 0.920 | 7 | 0 |
| v2 | hard | no dates | 143 / 148 | 0.966 | 0.953 | 5 | 0 |
| v1 | M1 | as is | 95 / 96 | 0.990 | 0.950 | 3 | 0 |
| v1 | M1 | swapped | 96 / 101 | 0.950 | 0.960 | 6 | 0 |
| v1 | M1 | no dates | 90 / 92 | 0.978 | 0.900 | 3 | 0 |
| v2 | M1 | as is | 96 / 97 | 0.990 | 0.960 | 2 | 0 |
| v2 | M1 | swapped | 98 / 103 | 0.951 | 0.980 | 1 | 0 |
| v2 | M1 | no dates | 92 / 94 | 0.979 | 0.920 | 1 | 0 |

The hard test set ([eskora/memory-contradiction-pairs-hard](https://huggingface.co/datasets/eskora/memory-contradiction-pairs-hard), 300 pairs, 150 contradictions) is built around 11 traps. The M1 test set ([eskora/memory-contradiction-pairs](https://huggingface.co/datasets/eskora/memory-contradiction-pairs), 200 pairs, 100 contradictions) checks that nothing got worse.

**The bar, fixed before any score:** on the hard set, worst-case recall at least v1's plus 0.03 (139 of 150), and worst-case precision no more than 0.01 below v1's; on the M1 set, every version within 0.02 of v1; 0 invalid answers. v2 passed every part except the first: 138 of 150, in the swapped version.

What changed, plainly:
- **Better:** more contradictions caught in every version of both sets, fewer false alarms on the hard set, and `unclear` less than half as often (18 against 42 on the hard set).
- **Not better:** the `looks_like_clash` trap did not move, and far-apart pairs got slightly worse when swapped (21 to 20 of 25). That one version decided the miss.

Every count and the per-trap table: [results/m2/v2/summary.md](https://github.com/e-skora/model-training/blob/main/results/m2/v2/summary.md).

## How to load it

The same way as v1, with this repo's name and commit. This is a shortened copy of the loader the scores came from (`src/judge/load.py` in the GitHub repo), with Unsloth 2026.10.7 and PEFT on one NVIDIA L4:

```python
import unsloth  # import first, as Unsloth asks
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    revision="7744afa8566e264af1a92a806d8d9aae00cc7c78",
    max_seq_length=2048, load_in_4bit=True)
model = PeftModel.from_pretrained(model, "eskora/m2-contradiction-judge-qwen3-4b-grpo-lora",
                                  revision="0964d031073408707397ce7438fc7df7b8a772c0", is_trainable=False)
FastLanguageModel.for_inference(model)
```

Then build the messages with the exact prompt from v1's card and generate with `do_sample=False`, `num_beams=1` and `max_new_tokens=256`, as the code on v1's card shows. There is no laptop (GGUF) version of v2.

## Limits

- **It missed its bar.** Use v1 unless you want to study what RL changed.
- **Made-up data, one writer.** All training and test pairs were written by Claude, a language model made by Anthropic, about made-up people, in one style. Scores on memories written by real people will likely be lower.
- **English only, everyday topics only.** Two short memories about one named person at a time.
- **It still dodges sometimes** (`unclear` on a real contradiction), though less than v1.
- **Not for high-stakes decisions.** Use it to flag pairs for a person to check.

## Files

`adapter_model.safetensors` and `adapter_config.json` (the adapter), the tokenizer and chat template, and `train.meta.json` (every setting, the run's checks and the checkpoint choice). Code, raw answers and full results: [github.com/e-skora/model-training](https://github.com/e-skora/model-training).

License: Apache 2.0, the same as the base model.
