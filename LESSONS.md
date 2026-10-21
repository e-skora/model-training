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
