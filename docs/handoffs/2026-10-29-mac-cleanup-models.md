# Handoff: model files on the Mac that the model-training project needs

**From:** the Cowork project "Fine Tuning and RL training" (repo `~/code/model-training`).
**To:** the Cowork project helping Elias clean up his Mac.
**Checked on the Mac:** 2026-10-29, with `ollama list`, `ls` and `du`. Nothing was deleted or changed.

## Do not delete

| What | Where | Size | Why it must stay |
|---|---|---|---|
| Ollama model `m1-judge:v1` | inside `~/.ollama/models` (remove it only with `ollama rm`, never by deleting files there) | 2.5 GB | The trained judge (v1). Aeverum's step 12 pins this exact model by its digest `4490a9bd0a72...`. M2 compares the new judge against it on the Mac, and `scripts/run_ollama.py` calls it by name. |
| The repo itself | `~/code/model-training` (including its hidden `.venv` and `.git` folders) | | The project's code, data, results and Python setup. |

Inside `~/.ollama/models`, the file that holds `m1-judge:v1` is `blobs/sha256-b6d930ce83c06512d12da2549b219edda8506e57404007acd877de29308172a9`. Never delete files in `~/.ollama` by hand; it breaks the models that use them.

## Safe to delete for this project (Elias decides)

| What | Where | Size | Why it is safe |
|---|---|---|---|
| The downloaded GGUF file | `~/models/m1-judge/` (the `.gguf` file and its small `.cache` folder) | 2.3 GB | A second copy. Ollama keeps its own copy (checked: different file on disk), so `m1-judge:v1` keeps working. It is only needed to rebuild `m1-judge:v1` from scratch; it can be downloaded again from the public Hugging Face repo `eskora/m1-contradiction-judge-qwen3-4b-gguf` (steps and SHA-256 check in `~/code/model-training/docs/run-judge-on-mac.md`). |
| Hugging Face cache entry for the adapter | `~/.cache/huggingface/hub/models--eskora--m1-contradiction-judge-qwen3-4b-lora` | 12 KB | Tiny and never used locally; training runs on Modal. |

## Not this project's models

These are on the Mac but this project never uses them. Another project may; check before deleting:
- Ollama: `llama3.1:8b`, `llama3.1:8b-text-q4_K_M`, `llama3.1:8b-text-q8_0`, `qwen2.5:7b-instruct`, `nomic-embed-text:latest`.
- `~/.cache/huggingface/hub`: everything except the `eskora` entry above (mlx-community Llama and Gemma models, gpt2, faster-whisper, all-MiniLM, pocket-tts).

## Coming later

M2 Step 7 will add the new judge to the Mac: a GGUF file in `~/models/` and an Ollama model `m2-judge:v2`. Once that exists, keep `m2-judge:v2` too, by the same rule as `m1-judge:v1`.

All training and big model files live on Modal and Hugging Face, not on the Mac.
