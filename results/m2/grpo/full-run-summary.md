# M2 Step 5: the full GRPO run, what happened

2026-11-09, Claude Code, prompt `prompts/M2-05f-record-and-run.md`, branch `m2-05-grpo`. Approved by Elias (D-059). Record: `train.meta.json`; logs: `train-output.log`, `choose-output.log`, `upload-output.log`.

## The answer first

- **Training finished and passed all its own checks.** 314 steps in 71.3 minutes on one L4, run `train-20261109T103829Z-771d0b`.
- **Step 250 was chosen** by the rule written before training (D-056 with D-057's format gate).
- **On validation it beats v1** (worst case over the three versions):

| | valid-hard precision | valid-hard recall | M1 valid precision | M1 valid recall |
|---|---|---|---|---|
| v1 (step 0) | 0.894 | 0.872 | 0.977 | 0.913 |
| step 250 | 0.957 | 0.957 | 0.957 | 0.957 |

- These are validation numbers. The frozen hard test set and the M1 test set have not been opened; scoring v2 on them is M2 Step 6.
- **Uploaded** to the private repo `eskora/m2-contradiction-judge-qwen3-4b-grpo-lora`, commit `0964d031073408707397ce7438fc7df7b8a772c0`.
- **Billed $1.55 in all:** training $0.97 (estimate $1.02), choosing $0.57 (estimate $0.71), upload $0.01.

## Training

- Every check on the real GPU passed: the exact prompt tokens, left padding, the 256-token limit and the end tokens 151643 and 151645, the sampling settings, one answer per prompt copy, 314 calls at all five recording points, and every one of the 314 pool pairs exactly twice.
- Load 56 s; peak memory 7.55 GiB of the L4's 22 GiB.
- Answers written: 5,024 (3,006 contradiction, 1,640 non_contradiction, 371 unclear, 7 invalid).
- Mean reward by 40-step stretch: 0.65, 0.63, 0.83, 0.67, 0.96, 0.70, 0.80, 0.91. It rises, with noise.
- `frac_reward_zero_std` rose from 0.45 in the first 40 steps to about 0.8. By the end the model mostly gave the same answer to all 8 tries of a pair, so later steps had less to learn from. That fits steps 250, 300 and 314 scoring the same.
- 34 warning lines, all "unclear above 15%" over 20-step windows; the highest was 19.4%. The plan said this can come without reward hacking (8% of the pool's right answers are unclear). No invalid or one-label warning.

## The choice

- Candidates: step 0 (v1) and the checkpoints at 50, 100, 150, 200, 250, 300 and 314. Each answered 588 items (valid-hard 97 pairs and M1 valid 99 pairs, three versions each), greedy.
- No candidate gave an invalid answer or one that breaks v1's format.
- Out: step 100, whose worst-case precision on M1 valid fell more than 0.03 below step 0's (0.936 against 0.977).
- Steps 250, 300 and 314 tie on every count; the rule takes the earliest, step 250.
- The answers and the whole choice are saved in `valid/` (`choice.json`).
- GPU time 2,254 s over 8 machines (37.6 minutes in all). The wait was 35 minutes, not the 7 planned; why was not checked.
