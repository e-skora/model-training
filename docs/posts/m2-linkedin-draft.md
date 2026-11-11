# LinkedIn post draft (M2). Elias edits before posting.

My second model missed its pass mark by one test pair, and I learned more from it than from the first one.

The model is a small judge that reads two memories about a person and says whether they contradict each other. I trained it further with reinforcement learning (GRPO) and set the pass mark in code before scoring. It needed to catch 139 of 150 hard contradictions in its weakest version. It caught 138.

It still got better: more contradictions caught on every test, fewer false alarms on the hard one, and less than half as many "can't tell" dodges. But a rule set in advance is the rule.

What I learned:
- Size the test for the gain you want to see. Mine could not tell a 4-pair gain from noise.
- Rebuild the practice pool as the model improves. Mine went stale part way through.
- Language models writing "can't tell" examples drift toward dodging, so I now check those by hand.

Every lesson is a fix in attempt 2.

Write-up: https://github.com/e-skora/model-training/blob/main/docs/m2-what-we-learned.md
Model: https://huggingface.co/eskora/m2-contradiction-judge-qwen3-4b-grpo-lora
Benchmark data: https://huggingface.co/datasets/eskora/memory-contradiction-pairs-hard
