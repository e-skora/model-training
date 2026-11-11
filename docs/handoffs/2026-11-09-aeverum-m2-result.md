# Update from the "Fine Tuning and RL training" project: the M2 result (2026-11-09). Information only.

1. v2, the judge trained further with reinforcement learning, missed its bar by one pair (hard test set worst-case recall 138 of 150, bar 139), so it is not called better than v1; details in `docs/m2-what-we-learned.md` in `~/code/model-training`.
2. v1 (`m1-judge:v1`, GGUF SHA-256 `b6d930ce83c06512d12da2549b219edda8506e57404007acd877de29308172a9`) stays the candidate for your step 12; nothing about it changed.
3. A second attempt (M3) is starting, built on the M2 lessons; it is judged against v1 the same way, and nothing here read or will read Aeverum's code, corpus or benchmark.
