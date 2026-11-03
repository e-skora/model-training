# Prompt M2-05: GRPO training code, tests and a short trial run

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `main`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`. Read `AGENTS.md` first and follow it. Then read:
- `STATE.md`;
- `milestones/M2-rl-judge/SPEC.md` (Steps 5 and 6, "Done means", "Cost");
- `DECISIONS.md` D-019, D-031, D-037, D-040, D-041, D-043, D-046, D-049 to D-056 (D-055 and D-056 are new and set this task);
- `LESSONS.md`, above all the 2026-10-22 entries on M1 training and the 2026-10-27 entries on GPU runs;
- `scripts/train.py` and `prompts/M1-07-train.md` (how M1 trained, chose and recorded; reuse its patterns);
- `src/judge/reward.py` (`reward`, `GRPOReward`, `reward_sha256`), `src/judge/pool.py`, `src/judge/generate.py` (`SAMPLING`), `src/judge/load.py`, `src/judge/prompt.py`, `src/judge/modal_image.py`;
- `results/m2/pool/summary.md`, `results/m2/pool/v1-samples.meta.json` and `results/m2/pool/reward-recheck.md`.

This task is M2 Step 5, part 1: the GRPO training code, its tests, and a short trial run on Modal. GRPO means: for each pair the model writes 8 answers, the reward function scores them, and training pushes the model toward the answers that scored above the group's average.

**Rules for this task:**
- Make and switch to a branch named `m2-05-grpo`.
- Do **not** start the full training run. Elias has approved this prompt and its trial run only (D-055).
- Every single Modal run must stay under $1 (D-019). Estimate each run before you start it; if the estimate is over $1, stop and report.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`, and never run a model on them.
- `data/m2/valid-hard.jsonl` and `data/m1/valid.jsonl` are used only to choose a checkpoint (D-056), never for training.
- Do not change any file under `src/` unless this prompt says so. Any change there moves the reward fingerprint (D-049); if you must change one, say why, and rerun `results/m2/pool/reward-recheck.md`'s check.
- Do not change any saved result in `data/m2/` or `results/m2/pool/`.

## Part A: check the tools first (no GPU)

Check each of these against the docs or the installed source of the exact pinned versions in `src/judge/modal_image.py` (TRL 1.15.0, Unsloth 2026.10.7, transformers 5.17.0), and write what you found, with links or file paths, into `results/m2/grpo/settings.md`:
1. `GRPOConfig` defaults. Claude in Cowork read TRL 1.15.0's `trl/trainer/grpo_config.py` on 2026-11-02: `temperature` 1.0, `top_p` 1.0, `top_k` 0, `min_p` None, `repetition_penalty` 1.0, `num_generations` 8, `max_completion_length` 512, `learning_rate` 1e-6, `beta` 0.0, `loss_type` "dapo", `scale_rewards` "group", `epsilon` 0.2, `num_iterations` 1, `use_vllm` False, `log_completions` False. Confirm these.
2. How Unsloth 2026.10.7 patches `GRPOTrainer`, and whether it changes any sampling setting. Claude in Cowork saw `_caller_sampling_only` in `unsloth/models/rl.py`: it stops the model's own generation defaults (for Qwen3, top_p 0.8 and top_k 20) from replacing TRL's. Confirm it applies on our path (no vLLM).
3. Unsloth's current GRPO notebook for a Qwen3 model: its learning rate, LoRA settings, `optim`, `max_grad_norm`, steps, and sampling (the notebook is said to use `min_p` 0.1). Its guide says to wait at least 300 steps for the reward to rise (https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).
4. How to keep training an existing LoRA adapter (v1's epoch-2 adapter, D-031, D-037) with Unsloth, rather than adding a new one.

## Part B: the training code (no GPU)

Write `scripts/train_grpo.py`, in the same shape as `scripts/train.py` (stages run one at a time, results on a Modal volume named `m2-grpo`, never in git). Stages: `trial` now; `train`, `choose` and `upload` written now but run only in the next prompt.

1. **Start point:** the base files and v1's epoch-2 adapter at the exact commits recorded for M1 (D-031), with that adapter's weights trainable.
2. **Data:** every row of `data/m2/rl-pool.jsonl` (314 pairs), shown in the version saved in that row (`swapped`, `dates`). Do not run `format_pairs` again. Build each prompt with `judge.prompt.build_messages`. Dataset columns: the prompt, `gold_label`, `gold_type`, and `id`. Check the pool's SHA-256 is `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77` before training.
3. **Reward:** `judge.reward.GRPOReward()`, unchanged. Take `reward_sha256()` on the Mac before launch (D-051), check it is `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7`, and record it. Stop if it differs.
4. **Sampling must match the pool:** 8 answers per pair, temperature 1.0, top_p 1.0, top_k off, no min_p, 256 new tokens at most, the same chat template options. Do not copy the notebook's `min_p`. In the trial, record the sampling settings the generation really used, not only what the config asked for.
5. **Settings:** Style A only (D-040): the answer format stays v1's exactly. Choose the learning rate, steps, batch, gradient steps, `optim`, `max_grad_norm`, `beta` and `loss_type` from Part A's findings, write each choice and its source in `results/m2/grpo/settings.md`, and keep them in one place in the script. Save a checkpoint (the adapter) every 50 steps and at the end.
6. **Reward-hacking watch:** log, every logging step, the share of each answer kind from `GRPOReward` (`answers/contradiction`, `answers/non_contradiction`, `answers/unclear`, `answers/invalid`), the mean reward, `frac_reward_zero_std`, and the mean answer length. Write a plain warning line into the run log if, over the last 20 steps, invalid answers pass 2%, any one label passes 85% of answers, or `unclear` passes 15%.
7. **Choose stage, by D-056 exactly:** step 0 and every checkpoint answer `valid-hard.jsonl` and `data/m1/valid.jsonl`, greedy, three versions each, with the same answer code as `scripts/train.py`'s choose stage. Code the rule from D-056 before any result exists, with a test.
8. **Records:** write a `.meta.json` for each run, as `scripts/train.py` does: the pool, prompt and reward fingerprints, all settings, seeds, base and adapter commits, the sampling really used, GPU minutes, peak memory, and the billed cost from `modal billing report` when it is available (LESSONS 2026-10-27).

## Part C: tests (no GPU)

Add `tests/test_train_grpo.py`. Test, with no GPU and no network: the dataset rows match the pool rows and their saved versions; a wrong pool or reward fingerprint stops before anything runs; the sampling settings match `judge.generate.SAMPLING`; the reward-hacking warnings fire on made-up logs; the D-056 rule picks right on made-up scores, including ties, a disqualified candidate and step 0 winning; the frozen test files are never read. Run `uv run pytest` on the Mac.

## Part D: the trial run (GPU, under $1)

1. Estimate the trial's cost first from the M2-04 pool sampling run (`runs/RUNLOG.md`). Stop if it is over $1.
2. Run `--stage trial`: about 20 training steps from v1, saving a checkpoint at step 10 and at the end. Then load the last checkpoint and answer 8 `valid-hard` pairs to prove it loads and answers in v1's format.
3. Report from the trial: seconds per step, peak memory, the logged answer shares, mean reward and `frac_reward_zero_std` per step, any warnings, the sampling settings really used, and the cost.
4. Add one line to `runs/RUNLOG.md` per GPU run.

## Part E: the full-run plan and the Codex review prompt (no GPU)

1. Write `results/m2/grpo/full-run-plan.md`: the steps, the expected GPU minutes, and the cost of the full run and of the choose stage, worked out from the trial's real seconds per step. Say plainly whether the full run is over $1. Elias decides on that number, so make it clear.
2. Write `prompts/M2-05-codex-review.md`, in the shape of `prompts/M2-04h-codex-review.md`: Codex (ChatGPT desktop app), new session, folder `~/code/model-training`, branch `m2-05-grpo`, read-only, writes only `reviews/M2-05-codex-review.md`, no GPU or network, never opens either frozen test file. Ask it to check: that training starts from v1's adapter at the recorded commits; that the data is exactly the pool, in its saved versions; that sampling matches the pool, including what the trial really used; that `valid-hard` and M1 valid never reach training; that the D-056 rule is coded exactly; that the reward-hacking logs and warnings work; that the records are complete; and that nothing under `src/` or in the saved pool changed. Use D-053's rating rules.

## Close

- Update `STATE.md`: Step 5 part 1 written; next, Elias pastes `prompts/M2-05-codex-review.md` into Codex; the full run waits for Codex and for Elias's yes to the cost in `results/m2/grpo/full-run-plan.md`.
- Add lessons and mistakes to `LESSONS.md` and new terms to `GLOSSARY.md`.
- Commit in small steps and push the branch. Do not merge.

**Report back** in plain language:
- the settings you chose, each with its source;
- the sampling the trial really used;
- the trial's speed, memory, answer shares, reward and cost;
- the full run's estimated steps, minutes and cost;
- the number of tests that pass;
- confirmation that no file under `src/` and no saved pool file changed;
- anything that is Elias's to do (expected: paste the Codex prompt).
