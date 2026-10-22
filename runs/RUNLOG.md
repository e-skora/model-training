# RUNLOG.md

One line per GPU run.

| Date | Milestone/step | What ran | GPU | Minutes | Est. cost |
|---|---|---|---|---|---|
| 2026-10-21 | M1-03 | M1-03 hello GPU (scripts/hello_gpu.py) | L4 | 2.2 (whole run, incl. image build) | about $0.03 (2.2 min x $0.000222/s L4, upper bound) |
| 2026-10-22 | M1-05 | Smoke test, 8 pairs x 3 models (scripts/run_baseline.py --limit 8). Old pins: 2 models failed to load | L4 x3 | 2.3 (GPU minutes, all 3) | about $0.03 |
| 2026-10-22 | M1-05 | hello GPU rerun on new pins, Unsloth 2026.10.7 (scripts/hello_gpu.py) | L4 | 1.3 (whole run, incl. image build) | about $0.02 |
| 2026-10-22 | M1-05 | Smoke test, 8 pairs x 3 models, new pins; all 3 loaded | L4 x3 | 4.7 (GPU minutes, all 3) | about $0.06 |
| 2026-10-22 | M1-05 | Baseline, 200 pairs x 3 models (scripts/run_baseline.py) | L4 x3 | 9.0 (3.0 + 3.4 + 2.6) | about $0.12 |
