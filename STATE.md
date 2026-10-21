# STATE.md

Read this first in every session.

- **Phase:** 1, fine-tuning (supervised).
- **Active milestone:** M1, Memory contradiction judge. Spec: `milestones/M1-contradiction-judge/SPEC.md`.
- **Updated:** 2026-10-21

## Done
- 2026-10-21: Step 1. Repo created with control files (Claude, from Cowork).
- 2026-10-21: Step 3 merged to `main` at cc7cdf4 (checked on disk by Claude in Cowork).
- 2026-10-21: Step 3. Tools installed and first GPU job ran on Modal (Claude Code, branch `m1-03-tool-setup`). uv Python 3.12 project with `modal`; Modal login (profile `e-skora`) and Hugging Face login (`eskora`) work; Modal secret `huggingface` (key `HF_TOKEN`) created; `scripts/hello_gpu.py` ran on an L4 (22.1 GiB VRAM, Unsloth 2025.7.8), about 2 minutes, about $0.03. How to run jobs: `docs/how-to-run.md`.

## Next single action
Elias: paste prompt `prompts/M1-04-test-set.md` into Claude Code (new session). It writes the 200-pair test set and a 20-pair spot-check sheet for Elias.

## Frozen files
(none yet; the M1 test set fingerprint goes here)

## Open items
- The Modal image pins Unsloth 2025.7.8 and transformers 4.54.0 (from Modal's example, mid-2025). Step 5 must confirm these load the chosen base models, and upgrade the pins if not.
