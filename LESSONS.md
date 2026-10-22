# LESSONS.md

Lessons and mistakes, dated, so nothing gets relearned. Add to the bottom. Never edit an old entry.

## 2026-10-20: Mistake: the setup interview asked too much
The first setup interview asked 8 questions plus a list of defaults. Elias did not want to answer them.
**Lesson:** ask only what blocks the next step. Choose sensible defaults for everything else and write them in `DECISIONS.md`, where he can override them.

## 2026-10-21: Lesson: pick a first model whose answers a program can score
Reinforcement learning (RL) only works when a program can score the model's answers by itself. Picking a fine-tuning task with one checkable answer (a label) means the same task carries straight into RL.

## 2026-10-21: Lesson: what M1 Step 3 taught about Modal
- The first run of a new container image is slower because Modal builds it (here about 36 seconds of building). Later runs reuse the image.
- Modal's Unsloth example builds its image with Python 3.11. That is fine next to our local Python 3.12: the image is a separate computer setup.
- Modal's Unsloth example has no Hugging Face secret. We add our own (`huggingface`, key `HF_TOKEN`) so later jobs can upload models.
- Our free Modal plan does not show cost per run. Estimate it from the run's minutes times the GPU price, and log it in `runs/RUNLOG.md`.
- Newer Hugging Face tokens can be hundreds of characters long, not the short ~37 character ones. Check a token with `hf auth whoami`, not by its length.

## 2026-10-21: Mistake: started a run before its file existed
Claude Code started `modal run scripts/hello_gpu.py` while the file write was still blocked, so the run failed in 1 second before reaching Modal (no cost).
**Lesson:** confirm the file is on disk before starting a run.

## 2026-10-21: Lesson: where compatible_detail ends and merely_related begins
Rereading batch 1 of the M1 test set caught a mislabel. "Works from home on Fridays" and "works from the office on Thursdays" had been marked `compatible_detail`, but they are two separate facts, so they are `merely_related`.
**Lesson:** `compatible_detail` means the second memory adds detail to the same fact ("likes tea", then "drinks green tea in the morning"). `merely_related` means a different fact on the same topic ("hikes", then "owns hiking boots"). The rereads between batches are worth doing.

## 2026-10-21: Mistake: a shuffle that could not finish
The first shuffle of the test set kept reshuffling all 200 pairs until no label came more than 3 times in a row. Pure chance almost never gives that, so it ran until Claude Code stopped it (nothing was written, no harm done).
**Lesson:** when the order has to follow a rule, build it one step at a time with the rule built in, rather than reshuffling and hoping. Also: macOS has no `timeout` command.

## 2026-10-21: Mistake: the first test set followed obvious templates and keyword patterns; an independent check caught it before freezing.
**Lesson:** ask for hard cases by name, check for shortcut patterns, and have a second agent review before freezing.

## 2026-10-22: Lesson: the Unsloth pins from mid-2025 were too old
Modal's Unsloth example pins Unsloth 2025.7.8 and transformers 4.54.0. With them, `unsloth/Qwen3-4B-Instruct-2507` and `unsloth/Phi-4-mini-instruct` would not load: Unsloth stopped with "not supported in your current Unsloth version" (Qwen3-8B loaded). Unsloth 2026.10.7, the newest on PyPI, also refuses transformers 4.54.0. The fix: pin Unsloth 2026.10.7 and let `uv pip compile` (for Linux, Python 3.11) pick the matching versions of torch, transformers, trl and the rest, then write them all in one place, `src/judge/modal_image.py`. torch 2.14.1 brings CUDA 13.0, which Modal's driver supports (up to 13.3). `scripts/hello_gpu.py` still works on the new image.
**Lesson:** an example's pins age fast. Do a smoke test on a few pairs first: it found the problem for about $0.03.

## 2026-10-22: Lesson: the untrained models already pass Aeverum's bar on our test set
Untrained, Qwen3-8B scored precision 0.96 and recall 0.94 on the frozen test set, above the 0.85 and 0.80 bar. The other two candidates passed it too. So on this test set there is little room left for training to improve the contradiction label. The clearest room is the type (Qwen3-8B got it right only 66% of the time), and Aeverum's own benchmark may be harder than ours.
**Lesson:** score the base model before planning training, so the goal is measured against a real starting point.
