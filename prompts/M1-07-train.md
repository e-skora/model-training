# Prompt M1-07: Fine-tune the judge and score it

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `main`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`. Read `AGENTS.md` first and follow it, then `STATE.md`, `milestones/M1-contradiction-judge/SPEC.md`, `DECISIONS.md` (especially D-021 to D-028) and `LESSONS.md`. This task is M1 Step 7: fine-tune `unsloth/Qwen3-4B-Instruct-2507` on the training set, then score it on the frozen test set. Do not start Step 8 (saving for the Mac) or publish anything.

Create and switch to a branch named `m1-07-train`. Commit after each part.

**Rules for this task**
- **The test set is the final exam. Use it once, at the end.** Choose every training setting (epochs, learning rate and so on) by looking only at the validation set (`data/m1/valid.jsonl`), never the test set. If you train more than one version, pick between them on the validation set, then score only the chosen one on the test set.
- Check every claim about Unsloth, TRL, Modal or Hugging Face against current docs (Unsloth's fine-tuning guide and its Qwen3 notes), and write the URLs in your report. Our pins are in `src/judge/modal_image.py`.
- Use an L4 GPU and set a timeout on every Modal function. First do a short trial run (about 20 training steps) to measure speed and memory, then estimate the full cost. If the whole step would cost more than $3, stop and ask Elias.
- Never print or save the Hugging Face token.

## Part A: training
1. **Training script** `scripts/train.py` (Modal, L4):
   - Load the base model in 4-bit with Unsloth and add LoRA adapters (QLoRA, D-011), starting from Unsloth's recommended LoRA settings for this model.
   - Build the training examples with `judge.format.format_pairs` (fixed seed; D-026 order and dates). Use the model's own chat template, so the training input looks exactly like the scoring input from `judge.prompt`.
   - Train only on the answer part (the assistant's JSON), not on the prompt. Unsloth's docs show how ("train on responses only").
   - Measure the loss on the validation set during training. Loss is the model's error score: lower is better. Rising validation loss while training loss falls means overfitting.
   - Fix the random seed and save these to a `.meta.json`:
     - every training setting, plus the seed
     - the SHA-256 of `train.jsonl` and of `valid.jsonl`
     - the prompt SHA-256
     - the base model's commit
     - the training and validation loss curves
2. **Where the result goes:** the trained adapter (the small add-on file LoRA produces) is uploaded from Modal to a **private** Hugging Face repo, `eskora/m1-contradiction-judge-qwen3-4b-lora`, using the `huggingface` Modal secret. It stays private until Step 9. Do not put model files in git (`.gitignore` already blocks them).
3. Run the trial, estimate the cost, then run the full training. Log every GPU run in `runs/RUNLOG.md`.

## Part B: scoring
4. **Scoring script** `scripts/eval_trained.py` (Modal, L4): load the base model plus the trained adapter, and answer the frozen test set in all three versions (as is, swapped, no dates), with the same greedy settings and the same answer loop (`judge.generate`) as the baseline. Check the test set's SHA-256 against `STATE.md` first (`judge.frozen`). Save the raw answers and `.meta.json` under `results/m1/trained-v1/`.
5. **Results:** `results/m1/trained-v1/summary.md`, in plain language:
   - **The table:** a side-by-side comparison of the trained 4B against the untrained 4B and untrained 8B rows from `results/m1/baseline-v2/summary.md`, for all three test versions, with the same columns.
   - **The verdict:** does it meet SPEC "Done means" item 1? That means precision 0.85 and recall 0.80, 0 invalid answers, beating its untrained self, and how close it comes to the untrained 8B.
   - **The losses:** the training and validation loss, as a few numbers per epoch.
   - **Mistakes:** 5 example mistakes with a one-line note each.

## Close
6. Update `STATE.md` (Step 7 done; next action: Codex reviews the training and scoring code, then Step 8), `GLOSSARY.md` (loss, epoch, learning rate, adapter upload, and any other new terms), and `LESSONS.md`. Run `uv run pytest` until it passes. Add offline tests for anything in `scripts/train.py` that can run without a GPU, such as building the examples and masking the prompt part. Push the branch. Do not merge.

**Report back** in plain language: the comparison table, whether it meets the goal, the training settings you chose and why (from the validation set only), total GPU minutes and cost, the private Hugging Face repo link, and the doc URLs you checked.
