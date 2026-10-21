# STATE.md

Read this first in every session.

- **Phase:** 1, fine-tuning (supervised).
- **Active milestone:** M1, Memory contradiction judge. Spec: `milestones/M1-contradiction-judge/SPEC.md`.
- **Updated:** 2026-10-21

## Done
- 2026-10-21: Step 1. Repo created with control files (Claude, from Cowork).
- 2026-10-21: Step 3. Tools installed and first GPU job ran on Modal (Claude Code, branch `m1-03-tool-setup`). uv Python 3.12 project with `modal`; Modal login (profile `e-skora`) and Hugging Face login (`eskora`) work; Modal secret `huggingface` (key `HF_TOKEN`) created; `scripts/hello_gpu.py` ran on an L4 (22.1 GiB VRAM, Unsloth 2025.7.8), about 2 minutes, about $0.03. How to run jobs: `docs/how-to-run.md`.

## Next single action
Claude in Cowork: check and merge branch `m1-03-tool-setup`, then write the prompt for Step 4 (write the 200-pair test set; Elias spot-checks 20 pairs; freeze it).

## Frozen files
(none yet; the M1 test set fingerprint goes here)

## Open items
(none)
