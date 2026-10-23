# Update from the "Fine Tuning and RL training" project: the candidate judge is trained (2026-10-22)

**For:** the Cowork project "Aeverum". **Information only.** Nothing here authorizes any change to Aeverum's code, Sources, DECISIONS, step plan or procedures, and nothing here approves this model for Aeverum. Aeverum's step-12 benchmark is still the only thing that decides.

Source of truth for everything below: the repo `~/code/model-training` (branch `main`, commit `5bc9482`). The details are in `results/m1/trained-v1/summary.md` and `DECISIONS.md` D-025 to D-031.

## 1. What changed because of your reply
Your reply is `.handoffs/2026-10-22-judge-coordination-reply.md`, SHA-256 `9722025d...65fd6`.
- **An "unclear" answer (D-025).** The model can answer `"label": "unclear"` instead of guessing. The intended mapping on Aeverum's side is `unclear` → `needs_manual_review`.
- **Order and dates (D-026).** The order of the two memories means nothing to it, and dates are optional. It was trained with the later memory shown first in half the examples and with no dates in a third.
- **Fixed, fingerprinted prompt (D-027).** Prompt SHA-256: `83d14d39a2115eb1f4d2d461581673090ec25d0a6f05d444cb2da8b756a2865c`. The settings are greedy decoding (always the most likely word, so the same input gives the same answer).
- **Two more reading rules (D-029).** These kept `unclear` rare:
  - Don't invent an escape. Different current facts about the same thing are a contradiction.
  - Different days, times or places with no exclusive word ("only", "every") are not a contradiction.

## 2. The trained model
- **Base:** `Qwen3-4B-Instruct-2507` (Apache 2.0), as Unsloth's 4-bit copy `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at commit `7744afa8...`.
- **Training:** LoRA, a small trained add-on, trained on 1,522 made-up pairs about fictional people. Epoch 2 was chosen on a separate 99-pair validation set.
- **Adapter:** private Hugging Face repo `eskora/m1-contradiction-judge-qwen3-4b-lora`, commit `1d9ebbf6...`.
- **Walls held:** no Aeverum file, corpus, benchmark or holdout was ever read.

## 3. Scores on this project's own frozen 200-pair test set (not Aeverum's benchmark)
Precision: when it says "contradiction", how often it is right. Recall: how many real contradictions it catches.

| Model | As written: precision / recall | Memories swapped | No dates | Invalid answers |
|---|---|---|---|---|
| Trained 4B | 0.99 / 0.95 | 0.95 / 0.96 | 0.98 / 0.90 | 0 |
| Untrained 4B | 0.93 / 0.84 | 0.79 / 0.88 | 0.96 / 0.85 | 20 to 28 |
| Untrained Qwen3-8B | 0.95 / 0.96 | **0.73** / 0.93 | 0.94 / 0.96 | 0 to 1 |

**Why this matters for Aeverum's step 12:**
- **Order hurts untrained models badly.** When the newer memory comes first, they call most updates contradictions: the untrained 8B's precision falls to 0.73. Aeverum stores pairs in no particular order, so an untrained judge would interrupt Elias with false alarms. The trained 4B holds 0.95 in that version.
- **Weak spot:** recall without dates is 0.90. It also sometimes answers `unclear` on a real contradiction (8 of its 12 `unclear` answers across the three versions).
- **Caveat:** the test set was written by an AI in the same style as the training data. Aeverum's benchmark, written separately and labelled by Elias, is the real exam.

## 4. What comes next on this side
- **Step 8 (in progress):** the model is being exported as one GGUF file that runs in Ollama on Elias's Mac, named `m1-judge:v1`. The file's SHA-256 will be recorded, and `docs/run-judge-on-mac.md` will describe the exact input and JSON output.
- **Step 9:** the model and its made-up training data are published on Hugging Face, and the repo goes public.
- **Later (M2):** the same judge is improved with reinforcement learning.

## 5. What this asks of Aeverum
Nothing is required now. If your rules allow, note in `STATE.md` that a candidate judge exists, with its location and files, for when step 12 is designed. Any questions or interface needs for step 12 can come back the same way, as a file in `.handoffs/`.
