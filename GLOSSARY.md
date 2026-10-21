# GLOSSARY.md

Plain meanings. Add a term the first time it shows up.

- **AI model:** a program that has learned patterns from lots of examples. You give it text; it gives text back.
- **Parameters:** the model's internal numbers, set during training. "4B" means about 4 billion of them. More parameters usually means smarter but slower and more expensive.
- **Open model:** a model whose parameters anyone can download and change.
- **Base model:** the open model we start from, before our training.
- **Fine-tuning:** training an existing model a bit more on your own examples so it gets good at one job.
- **Supervised fine-tuning (SFT):** fine-tuning where every example shows the exact right answer, and the model learns to copy that kind of answer.
- **LoRA:** a cheap way to fine-tune. Instead of changing all the parameters, you train a small add-on (an "adapter") that sits on top of the model.
- **QLoRA:** LoRA on a squeezed, smaller-memory copy of the model, so it fits on a cheaper GPU.
- **Adapter:** the small add-on file that LoRA training produces. Base model plus adapter equals your fine-tuned model.
- **GPU:** a computer chip that does the math for training fast. We rent them by the second on Modal.
- **VRAM:** the GPU's own memory. The model has to fit in it.
- **Modal:** a service that runs your Python code on rented GPUs and bills by the second.
- **Unsloth:** free, open-source software that makes fine-tuning faster and use less memory.
- **Hugging Face:** the main public website for sharing models and datasets.
- **Model card:** a model's public info page on Hugging Face: what it does, how it was trained, how well it scores, its limits.
- **Dataset:** a collection of examples.
- **JSONL:** a file format with one example per line, each written as JSON (a tidy, standard way to write data that programs can read).
- **Training set:** the examples the model learns from.
- **Test set:** separate examples the model never trains on, used only to score it. Like a final exam the student has never seen.
- **Frozen:** a file that is locked and never changed again, so scores stay fair.
- **Baseline:** the base model's score before our training. Our goal is to beat it.
- **Synthetic data:** made-up examples, here written by an AI model and spot-checked by a person.
- **Contradiction:** two memories that cannot both be true at the same time.
- **Precision:** of all the times the model said "contradiction", how often it was right.
- **Recall:** of all the real contradictions, how many the model caught.
- **Reinforcement learning (RL):** training where the model tries answers, a program scores them, and the model learns to get higher scores.
- **Reward:** the score a program gives each answer during RL.
- **GRPO:** a popular RL method. The model writes several answers to the same question, and it learns from the ones that scored better than the group's average.
- **Ollama / MLX:** two programs that run AI models on your own Mac.
- **Branch:** a separate line of work in git, so changes can be tried without breaking `main`.
