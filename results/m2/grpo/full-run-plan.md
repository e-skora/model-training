# M2 Step 5: the full GRPO run, planned from the trial

Written 2026-11-04 by Claude Code (prompt M2-05). Nothing here has run. The full run waits for Codex's review (`prompts/M2-05-codex-review.md`) and for Elias's yes to its cost (D-055).

## The answer first

- **Training: about 75 L4 minutes, about $1.00 to $1.10. That is just over the $1 a run that D-019 allows without asking, so Elias decides.**
- Choosing the checkpoint: about 50 L4 minutes spread over 8 machines at once, about $0.65 to $0.75. Under $1.
- Upload: CPU only, under $0.01.
- **All of Step 5 from here: about $1.70 to $1.85**, inside Modal's free $30 a month (D-014) and the SPEC's $2 to $8 for all of M2.

## What runs

| Stage | Command | What it does |
|---|---|---|
| train | `uv run modal run scripts/train_grpo.py --stage train` | 314 steps (2 epochs of the 314-pair pool, 2 pairs a step, 8 answers a pair), saving the adapter at steps 50, 100, 150, 200, 250, 300 and 314 |
| choose | `uv run modal run scripts/train_grpo.py --stage choose` | step 0 (v1) and the 7 checkpoints each answer `valid-hard` (97 pairs) and M1 valid (99 pairs), greedy, in 3 versions: 588 answers each. D-056 picks one |
| upload | `uv run modal run scripts/train_grpo.py --stage upload` | the chosen adapter to the private repo `eskora/m2-contradiction-judge-qwen3-4b-grpo-lora`. Nothing to upload if step 0 is chosen |
| bill | `uv run python scripts/train_grpo.py bill train` | adds the billed cost from `modal billing report` to `train.meta.json`, after the hour has passed |

## How the numbers are worked out

All seconds below come from `trial.meta.json`, except the answer rate, which comes from M2-02.

**Training.**
- Load: 67 s.
- The first step: 174 s. It holds one-off start-up work, so it is counted once.
- Every later step: 13.5 s. The trial's steps 2 to 20 took 255.8 s in all, including one checkpoint save.
- Steps 2 to 314: 313 x 13.5 s = 4,213 s.
- Six more checkpoint saves than the trial measured, at about 9 s each (the step-10 save added about 9 s): about 54 s.
- Total: 67 + 174 + 4,213 + 54 = about 4,510 s, which is **75 L4 minutes**.
- At Modal's L4 price ($0.000222 a second, https://modal.com/pricing): 4,510 x $0.000222 = **$1.00**.
- Modal also bills CPU and memory. On the M2-04 pool run, the bill was about 8% above the L4 seconds alone ($0.80 against $0.74, `runs/RUNLOG.md`). With that: **about $1.08**.
- The trial's 40 pairs were a random draw from the pool, so the real run should be close. Longer prompts or answers would make it slower, and a run that stops early would make it cheaper.
- The function's timeout is 3 hours, well above 75 minutes. A checkpoint is saved to the volume every 50 steps, so a crash keeps what was saved.
- Peak memory in the trial was 7.49 GiB of the L4's 22 GiB, so memory is not a risk.

**Choosing.**
- Each of the 8 candidates loads the model (about 124 s, the trial's load check) and writes 588 greedy answers.
- The answer rate is about 0.42 s an answer: v1's adapter wrote 900 greedy answers in 6.3 minutes on the hard test set in M2-02 (`runs/RUNLOG.md`). So 588 answers take about 247 s.
- Per candidate: 124 + 247 = about 371 s. For 8: about 2,970 s, which is **50 L4 minutes**, about $0.66 before CPU and memory, **about $0.71** with them.
- The 8 run on 8 machines at once, so the wait is about 7 minutes once GPUs are free.

## What to look at in the full run's log

- **The reward-hacking watch** writes a WARNING line when, over the last 20 steps, invalid answers pass 2%, one label passes 85%, or `unclear` passes 15%. In the trial, `unclear` was 14.4% over 20 steps, and 8% of the pool's right answers are `unclear`. So an `unclear` warning can come without any hacking. It means "look at the answers", not "stop".
- **frac_reward_zero_std** averaged 0.475 in the trial: almost half the pairs gave GRPO nothing to learn from in a given step. If it climbs toward 1 as training goes on, the model has stopped disagreeing with itself, and later checkpoints add little.
- **Mean reward** should drift up over hundreds of steps, not in 20. Unsloth's guide says to wait at least 300 steps.

## Recommendation

Run it as planned: 314 steps, about $1.08. The only way to land under $1 is fewer steps (about 290), which goes below Unsloth's 300-step guidance for a saving of about 8 cents.
