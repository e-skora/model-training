# Prompt M1-09: Get everything ready to publish (but keep it private)

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `main`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`. Read `AGENTS.md` first and follow it, then `STATE.md`, `milestones/M1-contradiction-judge/SPEC.md`, `DECISIONS.md` (D-007, D-018, D-031, D-032), `results/m1/trained-v1/summary.md`, `results/m1/gguf-v1/summary.md` and `docs/run-judge-on-mac.md`. This task is M1 Step 9: prepare everything that will be published.

**Do not make anything public.** Every repo stays private. Claude in Cowork switches them to public only after Elias says so (D-007).

Create and switch to a branch named `m1-09-publish-prep`. Commit after each part.

**Rules for public text**
- No em dashes anywhere.
- Plain language a non-engineer can follow, with no hype.
- Every number must come from the results files. Name the test version each number comes from.
- Say the limits honestly, including that the test set was written by an AI in the same style as the training data.
- Check Hugging Face's model card and dataset card docs for the current metadata fields (license, base_model, tags, pipeline_tag, and so on), and write the URLs in your report.

**Do this**
1. **Dataset repo.** Upload `data/m1/train.jsonl`, `valid.jsonl` and `test.jsonl` to a new **private** Hugging Face dataset repo, `eskora/memory-contradiction-pairs`. Write its dataset card. It should cover:
   - what a pair is, and the labels and types
   - the five reading rules
   - that every person and memory is made up by an AI and spot-checked by blind second judges
   - the counts, the SHA-256 of each file, and the license
   - that the test split was never trained on
2. **Model cards.** Write the model card (`README.md`) for both private model repos and upload them:
   - **GGUF repo** `eskora/m1-contradiction-judge-qwen3-4b-gguf`. This is the main one people will use.
   - **Adapter repo** `eskora/m1-contradiction-judge-qwen3-4b-lora`.

   Each card covers:
   - **What it does,** with one example input and output.
   - **How to run it with Ollama,** from `docs/run-judge-on-mac.md`.
   - **The exact prompt,** with its SHA-256.
   - **How it was trained.** Base model and license, QLoRA with Unsloth on Modal, data size, epoch chosen on validation, cost.
   - **Results:** a table comparing the trained model with the untrained Qwen3-4B and Qwen3-8B on all three test versions, from the results files.
   - **Limits:** synthetic data, English, everyday topics, the `unclear` dodges, lower recall without dates, and that it is not for high-stakes decisions.
   - **Links** to the dataset and the GitHub repo.

   Use the base model's license (Apache 2.0) and set `base_model` in the metadata.
3. **GitHub README.** Update the repo's `README.md`:
   - the model's status
   - a short results table
   - links to the three Hugging Face repos
   - how the repo is laid out
   - the milestone roadmap
4. **Pre-publish check.** Write `scripts/prepublish_check.py`. It checks the repo's files at HEAD and the three Hugging Face repos, and reports, without changing anything:
   - no secrets, tokens or `.env` files
   - no model files in git
   - no em dashes in public text (READMEs, model and dataset cards)
   - no real personal data and no Aeverum text
   - every number in the cards matches the results files

   Run it and fix anything it finds.
5. **Close.**
   - Update `STATE.md`: Step 9 prepared, everything still private; the next action is Elias's go-ahead to publish.
   - Add lessons to `LESSONS.md`.
   - Run `uv run pytest`.
   - Push the branch. Do not merge.

**Report back** in plain language:
- links to the three private Hugging Face repos
- what the pre-publish check printed
- anything you were unsure how to say publicly
