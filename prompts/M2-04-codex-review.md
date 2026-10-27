# Prompt M2-04 review: Codex checks the reward function and the RL pool

**Where:** Codex (ChatGPT desktop app), **new session**, working directly in the folder `~/code/model-training` on your Mac.
**Branch:** `m2-04-pool-and-reward` (the folder is already on it; do not switch).

Copy everything below the line into Codex.

---

You are reviewing code in `~/code/model-training`, which is on branch `m2-04-pool-and-reward`. Read `AGENTS.md` first. **This is a read-only review:**
- Do not switch branches.
- Do not edit, commit or delete anything.
- Do not start any GPU or Modal job, and do not use the network.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`.
- The only file you write is `reviews/M2-04-codex-review.md`.

Context: M2 trains the contradiction judge further with reinforcement learning (GRPO). For each pair of memories the model writes 8 answers, a reward function gives each answer points, and training pushes the model toward the answers that scored above the group's average. So the reward function decides what the model learns. A mistake there, or a way to score well without judging well ("reward hacking"), would be trained in. Step 5 trains on it next, so check it before any training.

The intended rules are the reward table in `milestones/M2-rl-judge/SPEC.md` and decisions D-040 and D-046 in `DECISIONS.md`. The RL pool rule is D-046; the check of its always-wrong pairs is `reviews/M2-04-always-wrong-review.md` (D-047).

Review these files:
- `src/judge/reward.py` and `tests/test_reward.py`
- `src/judge/pool.py` and `tests/test_pool.py`
- `src/judge/generate.py` (the new `sample_items`)
- `scripts/sample_pool.py`
- every other change on this branch since `main` (`git diff main...m2-04-pool-and-reward`)

Check especially:
1. **The table:** does every row of the SPEC table and D-046 give exactly its points, for every right answer (contradiction, each non_contradiction type, unclear) and every given answer? Is each case covered by a test?
2. **"Invalid" means the same as in scoring:** the reward reads answers with the scorer's strict reader (`judge.score.parse_answer`, `format_problem`). Find any answer the scorer counts as invalid that earns points, or a valid one that gets -2.0. Try edge cases: text before or after the JSON, code fences, repeated keys, `NaN`, a type on `contradiction` or `unclear`, a wrong type name, an empty `reason`, extra fields, very long output.
3. **Reward hacking:** can any answer earn more than +1.25? Work out the average reward of always giving the same answer (always `contradiction`, always `unclear`, always `non_contradiction` with each type) on the label mix of `data/m2/rl-pool.jsonl`. Is any of them close to a good judge's average? Say whether the `unclear` costs make dodging pay anywhere.
4. **Fit with TRL 1.15.0:** Claude in Cowork checked the docs on 2026-10-27 (https://huggingface.co/docs/trl/v1.15.0/en/grpo_trainer): a reward function gets `completions`, every dataset column except `prompt` as keyword arguments, and may get `log_metric(name, value)`; it returns one float per completion; conversational completions are lists of message dicts read as `completion[0]["content"]`. Does `GRPOReward` fit that? What happens if `gold_type` is `None` for some rows, or a completion holds more than one message?
5. **The fingerprint:** does `reward_sha256()` change whenever a value, a rule or what counts as valid changes? Does it also change for things that do not change behaviour (comments, formatting)? Say whether that is a problem.
6. **The pool:** does `scripts/sample_pool.py` follow D-046 (8 answers at temperature 1.0, mixed pairs in, 20% always-right draw with a fixed seed, always-wrong listed)? Rescore `results/m2/pool/v1-samples.jsonl` with `reward.py` and confirm the counts and rewards in `results/m2/pool/summary.md` and `data/m2/rl-pool.jsonl`. Confirm the frozen test sets and validation files are never read.
7. **Sampling for Step 5:** the pool was sampled with temperature 1.0, top_p 1.0, top_k off. Note any setting that GRPO training would need to match for the pool to stay meaningful.
8. **Rerun:** run `uv run pytest` and report the count.

Write `reviews/M2-04-codex-review.md` in plain language a non-engineer can follow:
- a verdict line at the top: `ACCEPT` or `CHANGES REQUIRED`
- then each finding, with the file and line, what is wrong, why it matters, and the fix
- mark each finding `must fix` or `nice to have`
- if you find nothing, say so
