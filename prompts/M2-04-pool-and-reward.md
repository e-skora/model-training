# Prompt M2-04: The reward function and the RL pool

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `main`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`. Read `AGENTS.md` first and follow it. Then read:
- `STATE.md`;
- `milestones/M2-rl-judge/SPEC.md` (sections "Reward" and "Training data" are the contract);
- `DECISIONS.md` D-036 to D-046, above all D-040, D-043, D-045 and D-046;
- `LESSONS.md`;
- `src/judge/score.py`, `src/judge/prompt.py`, `src/judge/format.py`, `src/judge/generate.py`, `src/judge/load.py`;
- `scripts/eval_trained.py` (how v1's adapter is loaded and pinned).

This task is M2 Step 4 (the reward function) and Step 3 part B (the RL pool). Do not start any training. Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`, and never run a model on them.

Create and switch to a branch named `m2-04-pool-and-reward`.

## Part A: the reward function (no GPU)

1. **Write `src/judge/reward.py`.**
   - `reward(raw_answer, gold_label, gold_type)` returns one number, following the D-040 table and D-046.
   - The answer is read with the same strict parser as scoring (`judge.score.parse_answer`), so "valid JSON in the exact format" means exactly what the scorer means. Any answer the scorer counts as invalid gets -2.0.
   - When the right answer is decided (`contradiction` or `non_contradiction`):

     | Answer | Points |
     |---|---|
     | Right label | +1.0 |
     | Right type as well (non-contradictions only) | +0.25 more |
     | `contradiction` when it is not | -1.5 |
     | `non_contradiction` on a real contradiction | -1.0 |
     | `unclear` | -0.5 |

   - When the right answer is `unclear`: `unclear` +1.0, `non_contradiction` 0, `contradiction` -1.5.
   - The `reason` text earns nothing.
   - Keep every number in one named table at the top of the file, so a later change touches one place and shows in its fingerprint. Add `reward_sha256()`, a fingerprint of that table and the rules, the same way `prompt_sha256()` works.
2. **A wrapper for GRPO training.** Write a function with the reward-function signature that TRL's `GRPOTrainer` expects (completions plus dataset columns, returning one float per completion).
   - Check the signature against the current TRL docs and Unsloth's GRPO notebook first, and write the doc URLs in your report.
   - It must also count, per call, how many answers were `contradiction`, `non_contradiction`, `unclear` and invalid, so Step 5 can watch for reward hacking (the model finding a cheap way to score, such as always giving one label).
3. **Tests** in `tests/test_reward.py`, with no GPU:
   - every row of both tables;
   - invalid answers: not JSON, repeated fields, unknown label, `NaN`, extra text around the JSON, a type on a contradiction;
   - the type bonus only when the label is right;
   - the wrapper on a small batch;
   - the fingerprint changes when a value changes.

   Run `uv run pytest`.

## Part B: the RL pool (GPU on Modal)

4. **Candidates:**
   - `data/m2/train-hard.jsonl` (971 pairs);
   - the 47 `unclear` pairs of `data/m1/train.jsonl`, given `"source": "synthetic"`, `"label_by": "claude-code"` and `"hard": "m1_unclear"` in the pool file only (do not edit the M1 file).

   Show each candidate the way training will: choose its version (as is, swapped, or no dates) with `judge.format`'s rules and a fixed seed, and record the version.
5. **Sampling.** v1's adapter (epoch-2, D-031, the same pinned base and adapter commits as `scripts/eval_trained.py`) answers each candidate **8 times** on a Modal L4:
   - temperature 1.0, the same prompt (SHA-256 `83d14d39...2865c`), and the same chat template;
   - generate the 8 answers per prompt in one call if the library allows it.

   First run a smoke test on 16 candidates into a scratch folder, and project the full cost from its timing. If the projection is over $1, stop and ask Elias (D-019). Otherwise run the full set once. Add one line per GPU run to `runs/RUNLOG.md`.
6. **Score and choose** with `judge.reward`, by the D-046 rule. Fix the rule in code before looking at the answers:
   - **mixed** (the 8 rewards are not all equal): in the pool;
   - **always right:** add a random 20% of the mixed count (seed written down);
   - **always wrong the same way:** list them, keep them out.

   Write:
   - `data/m2/rl-pool.jsonl`: each pool pair with its version, the 8 rewards, and the label counts;
   - `data/m2/rl-pool-always-wrong.jsonl`;
   - `results/m2/pool/summary.md`, in plain language:
     - how many pairs are mixed, always right, always wrong and always unclear, per trap;
     - v1's sampled accuracy per trap;
     - how often it said `unclear` on decidable pairs;
     - the 8 answers for 5 example mixed pairs;
     - the pool size;
     - GPU minutes and cost;
     - the SHA-256 of `rl-pool.jsonl` and the reward fingerprint.
7. **Close.**
   - Update `STATE.md`: Step 4 and Step 3 part B written, waiting for the Codex review of the reward function and the check by Claude in Cowork.
   - Add lessons and mistakes to `LESSONS.md`, and new terms to `GLOSSARY.md`.
   - Commit in small steps and push the branch.
   - Do not merge.

**Report back** in plain language:
- the reward table as coded, and its fingerprint;
- the TRL and Unsloth doc URLs you checked;
- the pool counts per trap;
- the always-wrong count;
- GPU minutes and cost;
- the number of tests that pass;
- anything that is Elias's to do (expected: nothing).
