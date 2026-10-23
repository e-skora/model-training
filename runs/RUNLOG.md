# RUNLOG.md

One line per GPU run.

| Date | Milestone/step | What ran | GPU | Minutes | Est. cost |
|---|---|---|---|---|---|
| 2026-10-21 | M1-03 | M1-03 hello GPU (scripts/hello_gpu.py) | L4 | 2.2 (whole run, incl. image build) | about $0.03 (2.2 min x $0.000222/s L4, upper bound) |
| 2026-10-22 | M1-05 | Smoke test, 8 pairs x 3 models (scripts/run_baseline.py --limit 8). Old pins: 2 models failed to load | L4 x3 | 2.3 (GPU minutes, all 3) | about $0.03 |
| 2026-10-22 | M1-05 | hello GPU rerun on new pins, Unsloth 2026.10.7 (scripts/hello_gpu.py) | L4 | 1.3 (whole run, incl. image build) | about $0.02 |
| 2026-10-22 | M1-05 | Smoke test, 8 pairs x 3 models, new pins; all 3 loaded | L4 x3 | 4.7 (GPU minutes, all 3) | about $0.06 |
| 2026-10-22 | M1-05 | Baseline, 200 pairs x 3 models (scripts/run_baseline.py) | L4 x3 | 9.0 (3.0 + 3.4 + 2.6) | about $0.12 |
| 2026-10-22 | M1-06 | Smoke test, 8 pairs x 3 test versions x 2 models (scripts/run_baseline.py --limit 8). A first try stopped on the Mac before reaching Modal (no GPU) | L4 x2 | 3.3 (1.3 + 2.0) | about $0.04 |
| 2026-10-22 | M1-06 | Baseline v2, new prompt, 200 pairs x 3 test versions x 2 models (scripts/run_baseline.py) | L4 x2 | 14.5 (6.6 + 7.9) | about $0.19 |
| 2026-10-22 | M1-07 | Trial 1, 20 training steps (scripts/train.py --stage trial). Stopped before training: the example check found the template's empty think block | L4 | 8 or less (app time 17:34-17:42 PDT, upper bound) | about $0.11 or less |
| 2026-10-22 | M1-07 | Template probe, one tokenizer render (scratch script, CPU only, no GPU) | CPU | under 1 | under $0.01 |
| 2026-10-22 | M1-07 | Trial 2, 20 training steps. Trained, but results could not be sent back (PyTorch values) | L4 | 6 or less (app time 17:44-17:50 PDT) | about $0.08 or less |
| 2026-10-22 | M1-07 | Trial 3, 20 training steps plus 8 validation answers: all worked. 7.4 s a step, peak VRAM 4.5 GiB | L4 | 9 or less (app time 17:51-18:00 PDT; training 2.5 min, model loads 3.3 min) | about $0.12 or less |
| 2026-10-22 | M1-07 | Full training, 3 epochs, 288 steps (scripts/train.py --stage train). Adapter saved after each epoch | L4 | 33.6 (wall; load 1.6 + training 29.6) | about $0.45 |
| 2026-10-22 | M1-07 | Validation answers, 99 pairs x 3 versions x 3 epoch adapters, to pick one (scripts/train.py --stage choose) | L4 x3 | 10.2 (GPU minutes inside the 3 functions; 6.8 min wall) | about $0.14 |
| 2026-10-22 | M1-07 | Smoke test of scripts/eval_trained.py: adapter from Hugging Face, 8 validation pairs x 3 versions | L4 | 2.1 (wall) | about $0.03 |
| 2026-10-22 | M1-07 | Trained judge on the frozen test set, 200 pairs x 3 versions (scripts/eval_trained.py) | L4 | 6.8 (wall; load 1.3 + answer 4.4) | about $0.09 |
| 2026-10-22 | M1-07 | Upload of the chosen adapter to the private Hugging Face repo (scripts/train.py --stage upload, CPU only, no GPU) | CPU | about 1 | under $0.01 |
