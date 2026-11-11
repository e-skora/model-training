# LinkedIn post draft (M1 and M2). Elias edits before posting.

His LinkedIn has no earlier posts about this work, so this is a first post that introduces the whole project (rewritten 2026-11-11 in his voice). No em dashes.

---

For the past three weeks I have been experimenting with fine-tuning small open AI models, and before I start the second attempt I would like advice from people who have done this, so this post covers both attempts so far.

The project is a small judge for a personal memory app. It reads two memories about a person and decides whether they contradict each other, which matters because an app that remembers things for you has to notice when it is told two things that cannot both be true.

The first version, v1, is Qwen3-4B fine-tuned with LoRA. On its test set it caught 94 to 96 percent of the real contradictions, it was right 94 to 98 percent of the times it called one, and it runs on a laptop in under a second per pair.

The second version tried reinforcement learning (GRPO) on harder pairs, with the pass mark written in code before any score. It needed to catch 139 of 150 hard contradictions in its weakest version, and it caught 138. It improved on most numbers, but a rule set in advance is the rule, so v1 is still the model I recommend.

The miss was still worth publishing, because each thing it taught me became a fix in the second attempt. The test was too small to show the gain I wanted, the practice pairs went stale as the model improved, and the AI-written "can't tell" examples drifted toward dodging. I now check those by hand.

The code, the data and both models are public, and the whole project has cost about $5 of rented GPU time.

If you have trained with GRPO or something like it, how did you decide when to rebuild the practice pool? My plan is two rounds with a fresh pool between them, and I would rather hear what worked for you before I spend the GPU time.

Write-up, with every number linked to the file that shows it: https://github.com/e-skora/model-training/blob/main/docs/m2-what-we-learned.md
Code: https://github.com/e-skora/model-training
Recommended model (v1): https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-gguf
Benchmark data: https://huggingface.co/datasets/eskora/memory-contradiction-pairs-hard
