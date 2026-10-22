# Prompt M1-05 review: Codex checks the scoring code

**Where:** Codex desktop app, **new session**, working directly in the folder `~/code/model-training` on your Mac (not a cloud copy).
**Branch:** `m1-05-baseline` (the folder is already on it; do not switch).

Copy everything below the line into Codex.

---

You are reviewing code in `~/code/model-training`, which is on branch `m1-05-baseline`. Read `AGENTS.md` first. **This is a read-only review:** do not switch branches, do not edit, commit or delete anything, do not start any GPU or Modal job, and do not use the network. The only file you write is `reviews/M1-05-codex-review.md`.

Context: this repo fine-tunes a small model to judge whether two memories about a person contradict each other. Step 5 scored three untrained models on a frozen 200-pair test set. The results in `results/m1/baseline/summary.md` will decide which model gets trained, so the scoring must be right.

Review these files:
- `src/judge/score.py` (precision, recall, accuracy, type accuracy, invalid answers)
- `src/judge/prompt.py` (the prompt shared by scoring and later training)
- `tests/test_score.py`
- `scripts/run_baseline.py` and `src/judge/modal_image.py`

Check especially:
1. The math: is "contradiction" treated as the positive label everywhere? Do invalid answers count as wrong in both precision and recall? Is type accuracy computed only on correctly labelled non-contradictions, as the summary says?
2. Parsing: what answers get accepted (for example JSON inside a markdown code box)? Could the parser accept something it should reject, or reject a valid answer?
3. Run settings: is decoding really greedy? Is Qwen3-8B's thinking really off? Is the same prompt used for every model?
4. The test set: is `data/m1/test.jsonl` only read, never written, and is its SHA-256 checked against `STATE.md` before use?
5. Tests: do they cover the cases above, and do they stay off the network and out of real folders?
6. Rerun the scorer locally (`uv run pytest`, and the score function on the three answer files in `results/m1/baseline/`) and confirm the numbers in `summary.md`.

Write `reviews/M1-05-codex-review.md` in plain language a non-engineer can follow: a verdict line at the top (`ACCEPT` or `CHANGES REQUIRED`), then each finding with the file, line, what is wrong, why it matters, and the fix. Mark each finding `must fix` or `nice to have`. If you find nothing, say so.
