# M2 Step 5: the GRPO trial run (20 steps)

Run 2026-11-04 by Claude Code (prompt M2-05), `uv run modal run scripts/train_grpo.py --stage trial`, Modal app `ap-Y7grQw6cJWdeIsmcaYZmDi`, one L4. Full record: `trial.meta.json`. Full output: `trial-output.log`. The two trial adapters (step 10 and step 20) stay on the Modal volume `m2-grpo`, in `trial/`. They are for checking only and are never candidates.

## What it proves

- **It starts from v1.** The pinned base and v1's epoch-2 adapter loaded. All 504 adapter tensors were exactly equal to v1's file (largest difference 0). Only the adapter's 66,060,288 numbers were trainable. The adapter's settings were r 32, alpha 32, dropout 0, on the 7 projection layers.
- **It trains on the pool.** All 320 prompts the model answered were pool prompts, token for token (20 steps x 2 pairs x 8 answers). They covered 40 different pool pairs.
- **The sampling matched the pool.** All 20 `generate` calls ran with do_sample on, temperature 1.0, top_p 1.0, top_k 0 (off), min_p none, repetition_penalty 1.0 and at most 256 new tokens, with no sampling steps added (an empty list of logits processors). TRL put each prompt in the batch 8 times with one answer each, where the pool asked for 8 answers to one prompt. Each answer is drawn the same way in both (`settings.md`, Part A item 2).
- **Answers are read as the pool read them.** TRL decoded them with `batch_decode(skip_special_tokens=True)`, the same way `judge.generate` does.
- **A saved checkpoint loads and answers in v1's format.** The step-20 adapter, loaded fresh with `judge.load.load_judge`, answered 8 `valid-hard` pairs greedily. All 8 were valid JSON in v1's exact format: 0 invalid, 0 format errors. They were not scored, because validation only chooses a checkpoint (D-056).

## Speed, memory, cost

| | |
|---|---|
| Load (base, adapter, checks) | 67 s |
| Training, 20 steps | 431 s |
| First step | 174 s (one-off start-up work inside the first step) |
| Steps 2 to 20 | 13.5 s a step, checkpoint saves included |
| Peak GPU memory, training | 7.49 GiB of the L4's 22 GiB |
| Load check (load 124 s, answer 7 s) | 131 s, peak 5.57 GiB |
| Wall time on the Mac | 17.3 min (it waited for GPUs and started two containers) |
| Estimate from GPU seconds | about $0.14 (629 s x $0.000222) |
| Billed (`modal billing report`) | see `trial.meta.json` `billed_total_usd`, added once Modal reports the hour |

## What it logged

These numbers come from 20 steps of 2 pairs each. They are too few to show learning, and the trial was not meant to. The reward goes up and down with which 2 pairs a step drew.

| Step | Mean reward | frac_reward_zero_std | Mean length (tokens) | contradiction | non_contradiction | unclear | invalid |
|---|---|---|---|---|---|---|---|
| 1 | 0.875 | 0.50 | 42.4 | 0.94 | 0.06 | 0.00 | 0.00 |
| 2 | 0.719 | 0.50 | 39.6 | 0.81 | 0.00 | 0.19 | 0.00 |
| 3 | 0.203 | 0.00 | 41.4 | 0.31 | 0.31 | 0.38 | 0.00 |
| 4 | 1.000 | 1.00 | 38.4 | 1.00 | 0.00 | 0.00 | 0.00 |
| 5 | 1.125 | 1.00 | 40.2 | 0.50 | 0.50 | 0.00 | 0.00 |
| 6 | 1.125 | 1.00 | 43.8 | 0.50 | 0.50 | 0.00 | 0.00 |
| 7 | 1.250 | 1.00 | 41.7 | 0.00 | 1.00 | 0.00 | 0.00 |
| 8 | 0.844 | 0.50 | 42.7 | 0.31 | 0.50 | 0.19 | 0.00 |
| 9 | 0.438 | 0.00 | 39.9 | 0.62 | 0.00 | 0.38 | 0.00 |
| 10 | 1.016 | 0.50 | 38.8 | 0.50 | 0.44 | 0.06 | 0.00 |
| 11 | 0.531 | 0.50 | 42.7 | 0.69 | 0.00 | 0.31 | 0.00 |
| 12 | 0.500 | 0.00 | 38.9 | 0.19 | 0.69 | 0.12 | 0.00 |
| 13 | 1.031 | 0.50 | 41.9 | 0.44 | 0.50 | 0.06 | 0.00 |
| 14 | 0.500 | 0.00 | 38.8 | 0.44 | 0.50 | 0.00 | 0.06 |
| 15 | 1.125 | 1.00 | 44.9 | 0.00 | 0.50 | 0.50 | 0.00 |
| 16 | 0.906 | 0.50 | 38.9 | 0.94 | 0.00 | 0.06 | 0.00 |
| 17 | 0.609 | 0.50 | 39.0 | 0.69 | 0.31 | 0.00 | 0.00 |
| 18 | 0.375 | 0.50 | 40.8 | 0.69 | 0.31 | 0.00 | 0.00 |
| 19 | 0.734 | 0.00 | 42.5 | 0.31 | 0.44 | 0.25 | 0.00 |
| 20 | 0.797 | 0.00 | 42.4 | 0.12 | 0.50 | 0.38 | 0.00 |
| **mean** | 0.785 | 0.475 | 41.0 | 0.500 | 0.353 | 0.144 | 0.0031 |

- **frac_reward_zero_std** is the share of a step's pairs whose 8 answers all scored the same, so GRPO learned nothing from them. 0.50 means one of the two pairs. The mean, 0.475, is high for a pool chosen because v1's answers differed. With only 8 new random answers per pair, a pair that was mixed in the pool can come out all the same this time. The full run's log will show whether it stays this high.
- **Answers in all:** 160 contradiction, 113 non_contradiction, 46 unclear, 1 invalid, of 320.
- **No answer hit the 256-token limit** (`completions/clipped_ratio` 0 every step).
- **Warnings:** none. The watch needs 20 logged steps before it reads, so it read once, at step 20. The `unclear` share over those 20 steps was 14.4%, just under the 15% limit. Part of it is right: 26 of the 314 pool pairs (8%) have `unclear` as the right answer. So the full run may well raise the `unclear` warning without any reward hacking, and a warning means "look at the answers", not "stop".

## Found by the trial and fixed

- **Unsloth changed a setting on its own.** The log says "Unsloth: The DAPO paper recommends `epsilon_high = 0.28` - we will set it." The script had left it empty, so the trial ran with 0.28, which `trial.meta.json` does not show under `grpo`. The script now sets 0.28 itself, and a test checks every setting Unsloth would fill in (`settings.md`, Part A item 2).
- Nothing else changed after the trial. Its record keeps the script's settings as they were when it ran.
