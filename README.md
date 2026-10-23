# model-training

I am learning to fine-tune and train open language models hands-on, and publishing every model I make, with its data, code and honest scores.

## Models

| Model | What it does | Status |
|---|---|---|
| Memory contradiction judge v1 (Qwen3-4B) | Reads two memories about a person and says whether they contradict each other, with a one-line reason | Done 2026-10-23: trained, tested, runs on a Mac with Ollama |

On Hugging Face:

- [eskora/m1-contradiction-judge-qwen3-4b-gguf](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-gguf): the ready-to-run file for Ollama (start here)
- [eskora/m1-contradiction-judge-qwen3-4b-lora](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-lora): the training add-on (LoRA adapter), for GPUs
- [eskora/memory-contradiction-pairs](https://huggingface.co/datasets/eskora/memory-contradiction-pairs): the made-up training and test data

How to set it up on a Mac and call it from a program: [docs/run-judge-on-mac.md](docs/run-judge-on-mac.md).

## Results

Scored on 200 test pairs the model never trained on, in three test versions: the pairs **as is**, **swapped** (newer memory first) and **no dates**. Precision is how often "contradiction" was right when the model said it; recall is how many of the 100 real contradictions it caught.

| Model | Test version | Precision | Recall | Invalid answers |
|---|---|---|---|---|
| Trained judge (GGUF, Ollama on a Mac) | as is | 0.98 | 0.96 | 0 |
| Trained judge (GGUF, Ollama on a Mac) | swapped | 0.94 | 0.96 | 0 |
| Trained judge (GGUF, Ollama on a Mac) | no dates | 0.98 | 0.94 | 0 |
| Qwen3-4B-Instruct-2507, untrained | swapped | 0.79 | 0.88 | 20 |
| Qwen3-8B, untrained | swapped | 0.73 | 0.93 | 1 |

The untrained rows show the version where they did worst: with the newer memory first, they called most updates contradictions. The trained judge answers in 0.91 seconds per pair on an Apple M4 Pro. Every row, every test version, and the limits: [results/m1/gguf-v1/summary.md](results/m1/gguf-v1/summary.md) and the model cards.

The main limit: all the data, the test set included, was written by Claude, a language model, in one style. Scores on memories written by real people will likely be lower.

## How it is built

- Fine-tuning with [Unsloth](https://github.com/unslothai/unsloth) (QLoRA), on rented GPUs from [Modal](https://modal.com).
- Every model and dataset is published on [Hugging Face](https://huggingface.co/eskora).
- All training and test data is made up. No real personal data is used.
- The test set is written first and frozen (its SHA-256 fingerprint is recorded) before any training.

## How the repo is laid out

| Folder or file | What is in it |
|---|---|
| `data/m1/` | The train, validation and test pairs (JSONL), and how they were made |
| `src/judge/` | Shared code: the one prompt, scoring, data formatting, model loading |
| `scripts/` | Things you run: baseline, training, export to GGUF, Ollama runs, checks, upload |
| `tests/` | Tests (`uv run pytest`). They use fake data and temporary folders only |
| `results/m1/` | Every model's raw answers, settings and a plain-language summary per run |
| `ollama/Modelfile` | The recipe Ollama uses to run the GGUF file |
| `hf/` | The Hugging Face cards, and a record of each upload |
| `docs/` | How to run jobs, how to run the judge on a Mac, post drafts |
| `milestones/` | What each milestone must do (the spec) |
| `prompts/`, `reviews/` | The task prompts each step was built from, and the code reviews |
| `runs/RUNLOG.md` | One line per GPU run: what ran, how long, what it cost |
| `STATE.md`, `DECISIONS.md`, `LESSONS.md`, `GLOSSARY.md` | Where things stand, what was decided and why, mistakes not to repeat, plain meanings of the terms |

## Roadmap

| Milestone | What | Tools | Status |
|---|---|---|---|
| M1 | Memory contradiction judge, fine-tuned (supervised) | Unsloth on Modal, Hugging Face | v1 done |
| M2 | Same judge, improved with reinforcement learning (GRPO) | Unsloth on Modal | Next |
| Later | Managed fine-tuning, the way companies buy it | Together AI | Order not set |
| Later | Reinforcement learning practice | Tinker (Thinking Machines) | Order not set |

## License

Code: Apache 2.0 (`LICENSE`). Models: Apache 2.0, from the base model. Data: CC BY 4.0.
