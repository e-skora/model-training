---
pretty_name: Memory contradiction pairs, hard
license: cc-by-4.0
language:
- en
task_categories:
- text-classification
tags:
- synthetic
- contradiction-detection
- memory
- benchmark
annotations_creators:
- machine-generated
source_datasets:
- original
size_categories:
- 1K<n<10K
configs:
- config_name: benchmark
  default: true
  data_files:
  - split: test
    path: test-hard.jsonl
- config_name: training
  data_files:
  - split: train
    path: train-hard.jsonl
  - split: validation
    path: valid-hard.jsonl
- config_name: rl_pool
  data_files:
  - split: train
    path: rl-pool.jsonl
---

# Memory contradiction pairs, hard

Pairs of short memories about made-up people, each built around a **trap**: a way a judge could get it wrong by reading the surface instead of the meaning. The 300-pair test file works as a **small benchmark** for contradiction checking, with 11 trap groups so you can see where a model fails, not only how often.

It was made to train and test a small judge model for a personal memory app, which compares stored memories and flags clashes for a person to review. Every person and every memory in it is made up. There is no real personal data.

The easier first dataset, in the same format: [eskora/memory-contradiction-pairs](https://huggingface.co/datasets/eskora/memory-contradiction-pairs). Models: [eskora/m1-contradiction-judge-qwen3-4b-gguf](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-gguf) (v1, recommended) and [eskora/m2-contradiction-judge-qwen3-4b-grpo-lora](https://huggingface.co/eskora/m2-contradiction-judge-qwen3-4b-grpo-lora) (v2, missed its bar). What the experiment taught: [What I learned from training a small judge model with reinforcement learning](https://github.com/e-skora/model-training/blob/main/docs/m2-what-we-learned.md).

## The files

| File | Subset | Pairs | What it is for | SHA-256 |
|---|---|---|---|---|
| `test-hard.jsonl` | `benchmark` | 300 | The benchmark. Scoring only, never trained on | `d9e8e5f3a53369981d11c835a0a4bf68b9b105ae0c8dd78b2d02df967645ff82` |
| `train-hard.jsonl` | `training` | 971 | Training pairs on the same traps | `8c2abddb93b21a0850c000c2919ee0bb11277ecf3b9e272d55cf9703110ca856` |
| `valid-hard.jsonl` | `training` | 97 | Validation pairs, used only to choose a checkpoint | `dd29c90d5343c5e7b59208de90f0334a7b63b2849c7941f990bf38d18320b384` |
| `rl-pool.jsonl` | `rl_pool` | 314 | The pairs used for reinforcement learning, with how the starting model did on each | `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77` |

A SHA-256 is a fingerprint of a file's exact contents: change one letter and it changes. The test file here is byte for byte the frozen file the scores below were measured on.

```python
from datasets import load_dataset
benchmark = load_dataset("eskora/memory-contradiction-pairs-hard", "benchmark", split="test")
```

## What a pair is

The format of [eskora/memory-contradiction-pairs](https://huggingface.co/datasets/eskora/memory-contradiction-pairs), plus a `hard` field naming the trap:

```json
{"id": "h-001", "person": "Mulugeta Girma", "a": {"text": "Mulugeta takes coffee with three spoons of sugar.", "written_at": "2025-09-06"}, "b": {"text": "Mulugeta drinks coffee without any sugar at all.", "written_at": "2025-09-06"}, "label": "contradiction", "type": null, "hard": "explain_away", "reason": "One says lots of sugar and the other says none, for the same coffee habit."}
```

| Field | What it holds |
|---|---|
| `id` | `h-` for the benchmark, `tr2-` for train, `va2-` for validation; pool rows keep the id of the pair they came from |
| `person`, `a`, `b` | The made-up person and the two memories, each with `text` and `written_at` |
| `label`, `type`, `reason` | The right answer: `contradiction`, `non_contradiction` (with type `update`, `compatible_detail` or `merely_related`) or `unclear`, and one sentence why |
| `hard` | The trap the pair was written for |
| `source`, `label_by` | Train, validation and pool only: `synthetic` and `claude-code` on every row |
| `version`, `swapped`, `dates` | Pool only: how the pair was shown to the model (newer memory first or not, with dates or not) |
| `rewards`, `answer_counts`, `outcome` | Pool only: the reward of each of the starting model's 8 answers, how many of each label it gave, and `mixed` or `always_right` |

A model should see only the person and the two memories, never `label`, `type`, `hard` or `reason`. Labels follow five reading rules, written out on the [first dataset's card](https://huggingface.co/datasets/eskora/memory-contradiction-pairs).

## The benchmark: 11 traps

| Trap | What makes it hard | Right answer | Pairs |
|---|---|---|---|
| `explain_away` | You could invent a way both are true ("maybe two dogs"), but the rules say no | contradiction | 40 |
| `paraphrase` | The clash is said in different words | contradiction | 30 |
| `small_difference` | One number, day, colour or place differs | contradiction | 30 |
| `change_word_contradiction` | A change word appears, but both describe the same moment | contradiction | 25 |
| `far_apart_no_change` | Written months apart, with no change described | contradiction | 25 |
| `indirect_update` | A change told without the usual change words | update | 25 |
| `update_with_overlap` | Wording close to a contradiction, plus a clear change | update | 25 |
| `looks_like_clash` | Units, counts, part and whole, nicknames, general and specific | compatible_detail | 30 |
| `near_identical` | Almost the same sentence, and both are true | compatible_detail | 20 |
| `near_identical` | Almost the same sentence, but a different fact | merely_related | 25 |
| `rule5_bait` | Different days, times or places with no exclusive word such as "only" | merely_related | 25 |

That is 150 contradictions, 50 updates, 50 compatible details and 50 merely related pairs, for 300 different people. There are no `unclear` right answers in the benchmark. Every pair can be decided from its text alone, so the version without dates has one right answer too.

**Score it in three versions:** as is, **swapped** (newer memory first) and **no dates** (both dates removed). The worst of the three is the honest score, because a real app cannot promise order or dates.

## Baselines

v1 is a Qwen3-4B judge fine-tuned with LoRA; v2 is v1 trained further with reinforcement learning (GRPO). Both are adapters on the same 4-bit base, scored once by the same script, always picking the most likely word. **Precision** is how often "contradiction" was right when the model said it; **recall** is how many of the 150 real contradictions it caught.

| Model | Test set | Version | Caught / said | Precision | Recall | Unclear answers | Invalid answers |
|---|---|---|---|---|---|---|---|
| v1 | hard | as is | 138 / 144 | 0.958 | 0.920 | 12 | 0 |
| v1 | hard | swapped | 136 / 142 | 0.958 | 0.907 | 15 | 0 |
| v1 | hard | no dates | 134 / 142 | 0.944 | 0.893 | 15 | 0 |
| v2 | hard | as is | 141 / 145 | 0.972 | 0.940 | 6 | 0 |
| v2 | hard | swapped | 138 / 143 | 0.965 | 0.920 | 7 | 0 |
| v2 | hard | no dates | 143 / 148 | 0.966 | 0.953 | 5 | 0 |

Where both still miss the most: `paraphrase`, `far_apart_no_change` and `looks_like_clash`. Per-trap scores: [results/m2/v2/summary.md](https://github.com/e-skora/model-training/blob/main/results/m2/v2/summary.md).

## How it was made

- **Who wrote it.** Claude, a language model made by Anthropic, wrote every person and every memory, working in Claude Code. Topics are everyday ones: homes, jobs, food, pets, family, hobbies, travel. Names come from many cultures, and none appears in the first dataset.
- **Hard by design, not by filtering.** Each pair was written to fit its trap. No benchmark pair was picked or dropped by how a model answered it, because a test built from one model's failures makes that model look worse and the next one look better by luck.
- **Blind review of the benchmark.** Two separate Claude agents judged all 300 pairs without seeing the labels, and again with the dates hidden. They agreed with the file on 296 of 300. The 4 disagreements were all `near_identical` pairs with a swapped word read as a clash (an oud and a piano); they stay as written, because both memories can be true. Then the file was frozen.
- **Training and validation pairs** were written by 11 helper agents from one brief, reread and compared across all files for repeated ideas. A blind review of 330 of them agreed on 199 of 200 sampled training pairs and 96 of 97 decided validation pairs. It disputed 32 of the 33 new `unclear` pairs, so those were dropped, not relabelled. Only 1 `unclear` pair is left in train.
- **The RL pool** is 314 pairs on which v1 answered 8 times at temperature 1.0: 262 where its answers disagreed and 52 it always got right. Its 26 `unclear` pairs come from the first dataset's training split, which two blind reviews had agreed on.

## Limits

- **Synthetic, one writer.** Everything was written by the same language model in one style. Scores on memories written by real people will likely be lower.
- **The benchmark is now public,** so a model trained after this date may have seen it. Our own next attempt uses a fresh, larger test set for that reason.
- **Small.** One pair moves recall by 1 in 150. Report counts, not only rates, and treat gaps of a few pairs with care.
- **English only, everyday topics only.** Two short memories about one named person at a time.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): use it for anything, including commercial work, and credit "Memory contradiction pairs, hard, by Elias Skora".
