# LESSONS.md

Lessons and mistakes, dated, so nothing gets relearned. Add to the bottom. Never edit an old entry.

## 2026-10-20: Mistake: the setup interview asked too much
The first setup interview asked 8 questions plus a list of defaults. Elias did not want to answer them.
**Lesson:** ask only what blocks the next step. Choose sensible defaults for everything else and write them in `DECISIONS.md`, where he can override them.

## 2026-10-21: Lesson: pick a first model whose answers a program can score
Reinforcement learning (RL) only works when a program can score the model's answers by itself. Picking a fine-tuning task with one checkable answer (a label) means the same task carries straight into RL.
