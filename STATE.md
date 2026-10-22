# STATE.md

Read this first in every session.

- **Phase:** 1, fine-tuning (supervised).
- **Active milestone:** M1, Memory contradiction judge. Spec: `milestones/M1-contradiction-judge/SPEC.md`.
- **Updated:** 2026-10-22 (Step 5 done)

## Done
- 2026-10-21: Step 1. Repo created with control files (Claude, from Cowork).
- 2026-10-21: Step 3 merged to `main` at cc7cdf4 (checked on disk by Claude in Cowork).
- 2026-10-21: Step 3. Tools installed and first GPU job ran on Modal (Claude Code, branch `m1-03-tool-setup`). uv Python 3.12 project with `modal`; Modal login (profile `e-skora`) and Hugging Face login (`eskora`) work; Modal secret `huggingface` (key `HF_TOKEN`) created; `scripts/hello_gpu.py` ran on an L4 (22.1 GiB VRAM, Unsloth 2025.7.8), about 2 minutes, about $0.03. How to run jobs: `docs/how-to-run.md`.
- 2026-10-21: Step 4 written, waiting for Elias's spot-check (Claude Code, branch `m1-04-test-set`, not merged). `data/m1/test.jsonl`: 200 pairs, 200 different people; 100 contradiction, 34 update, 33 compatible_detail, 33 merely_related. `scripts/check_dataset.py` passes on it. SHA-256 as written: `b465340504a18f3a32667b9608ce43181010abd5c5bb47f1bc92a9335be9c3dd` (not frozen yet). Spot-check sheet: `data/m1/spotcheck-test.md` (20 pairs, seed 42). How it was made: `data/m1/README.md`.
- 2026-10-21: Step 4 revised after an independent check found shortcut patterns (no wrong labels). Claude Code rewrote 86 pairs in place (ids, labels and types kept) and fixed t-068's dates; same branch, not merged. Counts unchanged: 100 contradiction, 34 update, 33 compatible_detail, 33 merely_related. Change words now appear in every label; 10 updates are said indirectly; dates spread about half same-day per label. `scripts/check_dataset.py` now also fails when memory b is dated before memory a, and prints change-word counts; it passes. A second agent reread the rewritten pairs: no wrong labels, 10 wording fixes made. New SHA-256: `6e2a08323f445477f9600e371f1074ad3e2b41ef97d7072d6734ef481f6e574b` (not frozen yet). New spot-check sheet: `data/m1/spotcheck-test.md` (20 pairs, seed 7). Details: `data/m1/README.md`.
- 2026-10-21: Step 4 independent review (a separate Claude agent, judging blind): round 1 found 0 wrong labels but shortcut patterns and 4 broken pairs, fixed in M1-04b; round 2 found 0 wrong labels and 0 wrong types. Kept as correct under reading rule 1: contradictions months apart with no change words (for example t-008, t-123).
- 2026-10-22: Step 4 done. Elias spot-checked 20 pairs (seed 7), all OK ("spot check done"). Test set frozen and merged to `main` by Claude in Cowork.
- 2026-10-22: Step 5 done (Claude Code, branch `m1-05-baseline`, pushed, not merged). Shared prompt `src/judge/prompt.py`; scoring `src/judge/score.py` with 7 tests in `tests/test_score.py` (all pass, `uv run pytest`); baseline run `scripts/run_baseline.py`; shared Modal image `src/judge/modal_image.py`. Untrained scores on the frozen test set (precision / recall): Qwen3-8B 0.96 / 0.94, Qwen3-4B-Instruct-2507 0.93 / 0.83, Phi-4-mini-instruct 0.89 / 0.85. Base model: Qwen3-8B, thinking off (D-022). GPU use for the whole step: about 17 GPU minutes, about $0.23. Results: `results/m1/baseline/summary.md`.

## Next single action
Elias: paste `prompts/M1-05-codex-review.md` into Codex (new session). Claude in Cowork verified the baseline numbers independently on 2026-10-22 (same precision, recall, accuracy and invalid counts). After the review, Claude in Cowork merges Step 5.

## Frozen files
- `data/m1/test.jsonl`, 200 pairs, SHA-256 `6e2a08323f445477f9600e371f1074ad3e2b41ef97d7072d6734ef481f6e574b`, frozen 2026-10-22. Never edit it and never train on it.

## Open items
- Closed 2026-10-22: Unsloth pins. Unsloth 2025.7.8 could not load Qwen3-4B-Instruct-2507 or Phi-4-mini-instruct ("not supported in your current Unsloth version"). Pins moved to Unsloth 2026.10.7, transformers 5.17.0, trl 1.15.0, torch 2.14.1, now in one place, `src/judge/modal_image.py`. `scripts/hello_gpu.py` reran on them and works. Details: `LESSONS.md`.
- Settled 2026-10-22 by D-023 (train Qwen3-4B to beat the untrained Qwen3-8B). Background: the untrained Qwen3-8B already passes Aeverum's bar on our test set (precision 0.96, recall 0.94 against 0.85 and 0.80). "Done means" item 1 also needs the fine-tuned model to beat the base model, so training must beat 0.96 and 0.94. The clearest room to improve is the type (66% right). Worth deciding, before Step 6, whether this goal still fits or the training set should aim at the hard cases and types.
