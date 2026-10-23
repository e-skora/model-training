# Run the M1 judge on the Mac

The trained judge (M1 v1) is one file, a GGUF, that Ollama runs on the Mac. GGUF is the single-file model format llama.cpp and Ollama use. This page shows how to set it up from scratch and how a program (such as Aeverum's harness) asks it about two memories.

What you need: Ollama (0.24.0 was used), the Hugging Face command `hf` logged in as `eskora` (the repo is private), and this repo at `~/code/model-training`.

## The file

| | |
|---|---|
| Hugging Face repo (private) | `eskora/m1-contradiction-judge-qwen3-4b-gguf` |
| File | `m1-contradiction-judge-qwen3-4b-Q4_K_M.gguf` |
| SHA-256 | `b6d930ce83c06512d12da2549b219edda8506e57404007acd877de29308172a9` |
| Size | 2,497,280,320 bytes (2.5 GB) |
| Commit | f9a1ae6ec40ba6b9e1c544b13a317fdf5a452360 |
| Made by | `scripts/export_gguf.py`; settings in `results/m1/gguf-v1/export.meta.json` |

## Set it up (about 5 minutes)

Open Terminal.

1. Download the file into `~/models/m1-judge/` (outside the repo, because model files never go in git):
   ```bash
   hf download eskora/m1-contradiction-judge-qwen3-4b-gguf m1-contradiction-judge-qwen3-4b-Q4_K_M.gguf --revision f9a1ae6ec40ba6b9e1c544b13a317fdf5a452360 --local-dir ~/models/m1-judge
   ```
   If it shows no progress for a minute, stop it (Control-C) and put `HF_HUB_DISABLE_XET=1 ` in front of the same command. On 2026-10-23 Hugging Face's faster "Xet" download hung on this Mac and the plain download worked. At about 1 MB a second, the file takes about 45 minutes.
2. Check its fingerprint. The line it prints must start with the SHA-256 in the table above:
   ```bash
   shasum -a 256 ~/models/m1-judge/m1-contradiction-judge-qwen3-4b-Q4_K_M.gguf
   ```
3. Make the Ollama model from the repo's Modelfile (the recipe file that names the GGUF, the chat template and the settings):
   ```bash
   cd ~/code/model-training && ollama create m1-judge:v1 -f ollama/Modelfile
   ```
4. Check that Ollama runs exactly that file, with the right template and settings, on the validation set (it stops at once if anything differs, and refuses to write over earlier answers):
   ```bash
   uv run python scripts/run_ollama.py answer data/m1/valid.jsonl ~/models/m1-judge/check
   ```

Why the template matters: Unsloth warns that a model run with a different chat template than it was trained with gives poor or broken answers. `ollama/Modelfile` writes the template out in full rather than trusting Ollama to guess it.

## How a program calls it

Ollama answers on `http://127.0.0.1:11434` while the Ollama app is open. Send one request per pair to `/api/chat`, with `"stream": false`:

```json
{
  "model": "m1-judge:v1",
  "stream": false,
  "messages": [
    {"role": "system", "content": "<SYSTEM_MESSAGE from src/judge/prompt.py, exactly>"},
    {"role": "user", "content": "Person: Ana Ruiz\nMemory A (written 2025-03-02): Ana's favourite season is autumn.\nMemory B (written 2025-06-10): Ana's favourite season is spring."}
  ]
}
```

- **System message:** the exact `SYSTEM_MESSAGE` text in `src/judge/prompt.py`. Its fingerprint, with the user-message templates, is the prompt SHA-256 `83d14d39a2115eb1f4d2d461581673090ec25d0a6f05d444cb2da8b756a2865c` (`judge.prompt.prompt_sha256()`). Change one character and you have a different judge.
- **User message:** `Person: <name>`, then one line per memory: `Memory A (written <YYYY-MM-DD>): <text>` and `Memory B (written <YYYY-MM-DD>): <text>`. When a date is unknown, leave it out: `Memory A: <text>`. The order of A and B means nothing.
- **Settings:** do not send `options`; the Modelfile already sets temperature 0 (always the most likely word) and the other settings used in testing. Do not send `format` either: the answers were tested without it.

The reply's `message.content` is the judge's answer, one JSON object:

```json
{"label": "contradiction", "type": null, "reason": "Both say what her favourite season is, and they name different seasons."}
```

| `label` | `type` | Meaning |
|---|---|---|
| `contradiction` | `null` | The two memories cannot both be true. |
| `non_contradiction` | `update` | The later memory replaces the earlier one. |
| `non_contradiction` | `compatible_detail` | Both are true; one adds detail. |
| `non_contradiction` | `merely_related` | Same topic, no clash. |
| `unclear` | `null` | The text alone cannot settle it. Treat as "needs a person to look". |

`reason` is one short sentence. If the content is not valid JSON or has another label, treat it as "needs a person to look" too.

Privacy: the request and the answer contain memory text. Do not write either to a log (Aeverum §12.4). `scripts/run_ollama.py` writes memory text only to its answers file.

## How well it does

See `results/m1/gguf-v1/summary.md` for the scores on the frozen test set, the comparison with the trained adapter on Modal, and the speed on the Mac.
