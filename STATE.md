# STATE.md

Read this first in every session.

- **Phase:** 1, fine-tuning (supervised).
- **Active milestone:** M1, Memory contradiction judge. Spec: `milestones/M1-contradiction-judge/SPEC.md`.
- **Updated:** 2026-10-21 (Step 4 revised)

## Done
- 2026-10-21: Step 1. Repo created with control files (Claude, from Cowork).
- 2026-10-21: Step 3 merged to `main` at cc7cdf4 (checked on disk by Claude in Cowork).
- 2026-10-21: Step 3. Tools installed and first GPU job ran on Modal (Claude Code, branch `m1-03-tool-setup`). uv Python 3.12 project with `modal`; Modal login (profile `e-skora`) and Hugging Face login (`eskora`) work; Modal secret `huggingface` (key `HF_TOKEN`) created; `scripts/hello_gpu.py` ran on an L4 (22.1 GiB VRAM, Unsloth 2025.7.8), about 2 minutes, about $0.03. How to run jobs: `docs/how-to-run.md`.
- 2026-10-21: Step 4 written, waiting for Elias's spot-check (Claude Code, branch `m1-04-test-set`, not merged). `data/m1/test.jsonl`: 200 pairs, 200 different people; 100 contradiction, 34 update, 33 compatible_detail, 33 merely_related. `scripts/check_dataset.py` passes on it. SHA-256 as written: `b465340504a18f3a32667b9608ce43181010abd5c5bb47f1bc92a9335be9c3dd` (not frozen yet). Spot-check sheet: `data/m1/spotcheck-test.md` (20 pairs, seed 42). How it was made: `data/m1/README.md`.
- 2026-10-21: Step 4 revised after an independent check found shortcut patterns (no wrong labels). Claude Code rewrote 86 pairs in place (ids, labels and types kept) and fixed t-068's dates; same branch, not merged. Counts unchanged: 100 contradiction, 34 update, 33 compatible_detail, 33 merely_related. Change words now appear in every label; 10 updates are said indirectly; dates spread about half same-day per label. `scripts/check_dataset.py` now also fails when memory b is dated before memory a, and prints change-word counts; it passes. A second agent reread the rewritten pairs: no wrong labels, 10 wording fixes made. New SHA-256: `6e2a08323f445477f9600e371f1074ad3e2b41ef97d7072d6734ef481f6e574b` (not frozen yet). New spot-check sheet: `data/m1/spotcheck-test.md` (20 pairs, seed 7). Details: `data/m1/README.md`.

## Next single action
Elias: open `~/code/model-training/data/m1/spotcheck-test.md` (the checkout is on branch `m1-04-test-set`) in any text editor. This is the new sheet (seed 7) from the revised set; it replaces the old seed-42 sheet. For each of the 20 pairs, leave `Elias: OK` or change it to `Elias: WRONG, <why>`. Save, then tell Claude in the Cowork project "spot-check done". Claude in Cowork then checks, fixes any pairs he marked wrong, freezes the test set (fingerprint goes under "Frozen files") and merges.

## Frozen files
(none yet; the M1 test set fingerprint goes here)

## Open items
- The Modal image pins Unsloth 2025.7.8 and transformers 4.54.0 (from Modal's example, mid-2025). Step 5 must confirm these load the chosen base models, and upgrade the pins if not.
