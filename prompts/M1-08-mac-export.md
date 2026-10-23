# Prompt M1-08: Save the judge so it runs on the Mac

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `main`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`. Read `AGENTS.md` first and follow it, then `STATE.md`, `milestones/M1-contradiction-judge/SPEC.md` (the "Fitting Aeverum" section), `DECISIONS.md` (D-027 and D-031) and `results/m1/trained-v1/summary.md`. This task is M1 Step 8: turn the trained judge into a single file that Ollama can run on Elias's Mac, and prove that it gives the same answers. Do not publish anything (that is Step 9).

Create and switch to a branch named `m1-08-mac-export`. Commit after each part.

**Rules for this task**
- Check every claim about Unsloth's GGUF export, llama.cpp or Ollama against current docs. Unsloth's "Saving to GGUF" and "Saving to Ollama" pages are the place to start. Write the URLs in your report. Ollama 0.24.0 is installed on the Mac at `/opt/homebrew/bin/ollama`.
- Use Modal for the export (an L4, or a CPU machine if the docs say that's enough), with a timeout. If the whole step would cost more than $2, stop and ask Elias.
- No model files in git. Keep the downloaded model on the Mac in `~/models/m1-judge/`, which is outside the repo.
- **The test set is not used to fix anything.** Check that the export works using the validation set. Use the test set once at the end, only to report the shipped file's score.

**Do this**
1. **Export on Modal.**
   - Load the base model with the chosen adapter. The base model and adapter commit are recorded in `results/m1/trained-v1/qwen3-4b-trained-v1.meta.json` and `train.meta.json`.
   - Merge the adapter into the model and save it as GGUF with `q4_k_m` quantization, the way Unsloth's docs describe. GGUF is the single-file model format Ollama runs, and q4_k_m is a common 4-bit setting that keeps it small.
   - Upload the GGUF to a **private** Hugging Face repo, `eskora/m1-contradiction-judge-qwen3-4b-gguf`.
   - Record the export settings and the GGUF file's SHA-256 (D-027) in `results/m1/gguf-v1/export.meta.json`. Log the GPU runs in `runs/RUNLOG.md`.
2. **Set it up on the Mac.**
   - Download the GGUF to `~/models/m1-judge/` and check its SHA-256.
   - Write `ollama/Modelfile` in the repo. It points at that file and uses **the same chat template the model was trained with**; Unsloth warns that a different template breaks answers. It also sets temperature 0.
   - Create the Ollama model `m1-judge:v1` from it.
3. **Prove it gives the same answers.** Write `scripts/run_ollama.py`, which answers pairs through Ollama's local API using `judge.prompt` (same prompt, same SHA-256). It never logs the memory text anywhere except the results file (Aeverum §12.4). Then:
   - Run it on the validation set in all three versions (as is, swapped, no dates). Compare with the adapter's own validation answers from Step 7.
   - Report the share of labels that match, and the precision and recall on validation.
   - If label agreement is below 97% or there are any invalid answers, stop. Find the cause (template, quantization) using the validation set only, and report before going further.
4. **Final report score.** Only once step 3 passes, run the frozen test set through Ollama in all three versions (check the SHA-256 first with `judge.frozen`). Save the answers to `results/m1/gguf-v1/` and write `results/m1/gguf-v1/summary.md`, in plain language:
   - a table comparing the GGUF on the Mac with the trained adapter on Modal (`results/m1/trained-v1/`)
   - the speed on the Mac, in seconds per pair
   - the file size
   - the GGUF SHA-256
5. **How to use it.** Write `docs/run-judge-on-mac.md`: how to recreate `m1-judge:v1` from the Hugging Face file and the Modelfile, and how Aeverum's harness would call it. Give the input it expects and the JSON it returns, including `unclear`.

**Close**
- Update `STATE.md`: Step 8 done; next action is Step 9, publishing.
- Add new terms to `GLOSSARY.md`, such as GGUF, quantization, q4_k_m and Modelfile.
- Add lessons to `LESSONS.md`.
- Run `uv run pytest` until it passes. Add offline tests for `scripts/run_ollama.py`, using a fake local API.
- Push the branch. Do not merge.

**Report back** in plain language:
- validation agreement and the final comparison table
- seconds per pair on the Mac
- file size and SHA-256
- total GPU minutes and cost
- the doc URLs you checked
