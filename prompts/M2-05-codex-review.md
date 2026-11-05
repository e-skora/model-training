# Prompt M2-05 review: Codex checks the GRPO training code and its trial run

**Where:** Codex (ChatGPT desktop app), **new session**, working directly in the folder `~/code/model-training` on your Mac.
**Branch:** `m2-05-grpo` (the folder is already on it; do not switch).

Copy everything below the line into Codex.

---

You are reviewing code in `~/code/model-training`, which is on branch `m2-05-grpo`. Read `AGENTS.md` first. **This is a read-only review:**
- Do not switch branches.
- Do not edit, commit or delete anything.
- Do not start any GPU or Modal job, and do not use the network.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`, and never run a model on them.
- The only file you write is `reviews/M2-05-codex-review.md`.

Context: M2 Step 5 trains judge v1 further with GRPO (reinforcement learning). For each pair the model writes 8 answers, the reward function scores them, and training pushes the model toward the answers that scored above the group's average. Claude Code wrote the training code, its tests and a 20-step trial run (prompt `prompts/M2-05-grpo-trial.md`). The full run has not started; it waits for this review and for Elias's yes to its cost. Read these first:
- `scripts/train_grpo.py` (the code) and `tests/test_train_grpo.py` (its tests);
- `results/m2/grpo/settings.md` (what the installed TRL and Unsloth code does, and each setting with its source);
- `results/m2/grpo/trial.meta.json` and `results/m2/grpo/trial-summary.md` (the trial run);
- `results/m2/grpo/full-run-plan.md` (the full run's steps and cost);
- `DECISIONS.md` D-031, D-037, D-040, D-046, D-049 to D-056; `milestones/M2-rl-judge/SPEC.md` Steps 5 and 6.

The installed TRL, Unsloth and transformers files are not in the repo. Where your check needs them, judge from what `settings.md` quotes, with its file paths and line numbers, and say so.

**How to rate what you find (D-053, Elias's ruling, used here for the training code too).**
- A problem that ordinary, honest code or an ordinary run could cause by accident is `must fix`.
- A problem that needs code written on purpose to fool a check is a `known limit`. List it, but it does not block the full run.
- Anything else worth doing is `nice to have`.
- Say for each finding why you rated it that way.

Review `git diff main..HEAD` and check:
1. **Start point.** Does training start from v1's epoch-2 adapter on the pinned base, at the commits recorded for M1 (D-031: base `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at `7744afa8566e264af1a92a806d8d9aae00cc7c78`, adapter `eskora/m1-contradiction-judge-qwen3-4b-lora` at `1d9ebbf66e1a2abcad9932932bb7d810a38c59b9`)? Is that adapter's own weights what trains, with no new adapter added? Do the checks before training (LoRA settings, only LoRA weights trainable, every weight equal to v1's file) prove it? Does `trial.meta.json` show they passed?
2. **Data.** Is the training data exactly the 314 rows of `data/m2/rl-pool.jsonl` (SHA-256 `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77`), each shown in the version saved in that row (`swapped`, `dates`), without running `format_pairs` again? Does the test that compares them with `judge.pool.show` really prove it? Is the GPU job's check that every prompt it generates from is one of the pool's prompts, token for token, sound?
3. **Sampling.** Does the sampling match the pool: 8 answers a pair, temperature 1.0, top_p 1.0, top_k off, no `min_p`, 256 new tokens, the same chat template options? Is `settings.md` right that Unsloth's `_caller_sampling_only` does not run on this path, and that transformers 5.17.0 keeps TRL's values for another reason? What did the trial really use (`sampling_used` in `trial.meta.json`)? Could the recording miss a generate call, or pass when the sampling differs?
4. **Validation stays out of training.** Do `data/m2/valid-hard.jsonl` and `data/m1/valid.jsonl` never reach training? Are they used only by the choose stage and the trial's 8-answer load check?
5. **The D-056 rule.** Is `choose_checkpoint` exactly D-056: the candidates, each "out" rule and its limit (more than 0.02, more than 0.03, any invalid answer), worst case as the lowest of three versions, the pick order and the ties? Is reading "never says contradiction" as precision 0 right, or should it be handled another way? Is it right that the choose stage scores with the same answer code as `scripts/train.py`'s choose stage? Do the tests cover the edges?
6. **The reward-hacking watch.** Do the logged answer shares, mean reward, `frac_reward_zero_std` and mean answer length really come from `GRPOReward` and TRL each step? Do the warnings fire as the prompt asks (over the last 20 steps: invalid over 2%, any one label over 85%, `unclear` over 15%)? Is it right that they stay silent until 20 steps exist?
7. **Settings.** Is each choice in `settings.md` backed by its source? Is anything set by a default that Unsloth or TRL could change without notice? Is `mask_truncated_completions` off for the reason given? Is any setting risky for a small pool (314 pairs, 2 epochs)?
8. **Records.** Does `trial.meta.json` hold everything the prompt asks: the pool, prompt and reward fingerprints, all settings, seeds, base and adapter commits, the sampling really used, GPU minutes, peak memory, and the billed cost? Do the choose and upload stages check the record before any remote work, and refuse to replace saved results?
9. **Nothing else changed.** No file under `src/` changed, so the reward fingerprint is still `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7` (run `uv run python -c "from judge.reward import reward_sha256; print(reward_sha256())"`). `data/m2/rl-pool.jsonl`, `data/m2/rl-pool-always-wrong.jsonl`, `results/m2/pool/v1-samples.jsonl`, its `.meta.json` and `results/m2/pool/summary.md` keep their bytes (`git diff main..HEAD --stat`).
10. **The full-run plan.** Is the cost in `full-run-plan.md` worked out correctly from the trial's real seconds per step? Is it clear whether it is over $1?
11. **The tests.** Do the tests in `tests/test_train_grpo.py` test the real thing? Would each fail if the code it guards were broken? Do any touch the live environment, the network, or a frozen test file?
12. **Rerun:** run `uv run pytest` and report the count. Some older test functions read `data/m1/test.jsonl` (listed in `reviews/M2-04-fixes.md`). Leave those out, as you did in the M2-04 reviews, and say what you left out.

Write `reviews/M2-05-codex-review.md` in plain language a non-engineer can follow:
- a verdict line at the top: `ACCEPT` or `CHANGES REQUIRED`. Known limits alone do not make it `CHANGES REQUIRED`.
- then each finding, with the file and line, what is wrong, why it matters, and the fix
- mark each finding `must fix`, `nice to have` or `known limit`
- if you find nothing, say so
