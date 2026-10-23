# M1 Step 8: the Mac file did not pass the validation check (yet)

Date: 2026-10-23. Validation set only (`data/m1/valid.jsonl`, 99 pairs, in 3 versions: 297 answers). The frozen test set was **not** read.

## The result

The rule was: the GGUF on the Mac must give the same label as the trained adapter on Modal (Step 7) on at least 97% of validation answers, with no invalid answers. It gave the same label on **287 of 297 (96.6%)**, with **0 invalid answers**. So Step 8 stopped here, before the test set, as the prompt says.

## What was ruled out: the chat template

- The GGUF carries the training chat template byte for byte (checked by reading the file).
- `ollama/Modelfile` writes the template out; Ollama shows it back unchanged, and runs the file whose SHA-256 is the export's (`b6d930ce...72a9`).
- The validation set was answered twice: through Ollama's chat route, and with the exact text the training template writes, sent raw. Both gave **the same text in all 297 answers** (`valid/compare-raw-vs-chat.json`). Asked twice, Ollama also gives the same answer, so this is not chance.

## What moves the labels: the merge, then the 4-bit squeeze

Between the adapter on Modal and the file on the Mac there are two steps. A third run (`scripts/check_merge.py`, on Modal) answered the validation set after the first step only: the adapter merged into the original 16-bit weights, without quantization.

| Comparison (297 validation answers) | Same label | What differs between them |
|---|---|---|
| Merged 16-bit against the adapter | 288 (96.97%) | the merge only |
| Mac GGUF against merged 16-bit | 292 (98.3%) | the 4-bit squeeze (q4_k_m) and llama.cpp only |
| Mac GGUF against the adapter | 287 (96.6%) | both |

Why the merge changes answers: training ran the adapter on top of a **4-bit copy** of Qwen3-4B (QLoRA). Unsloth's GGUF export, as its docs describe, adds the adapter into the **original 16-bit weights**, which are slightly different numbers. The adapter learned to correct the 4-bit copy, so on the 16-bit weights some close calls tip the other way. The merge alone already falls just below 97%; the squeeze adds a few more flips.

## Is the Mac file worse?

Not clearly. Against the right answers (validation set), the three versions score about the same; they disagree on different close calls. The Mac file says `unclear` less often (3 times against 10), and catches more contradictions, with a few more false alarms in the swapped version.

| Model | Version | Precision | Recall | Accuracy | Unclear |
|---|---|---|---|---|---|
| Adapter on Modal (Step 7) | as is / swapped / no dates | 1.00 / 0.98 / 0.98 | 0.93 / 0.93 / 0.91 | 0.97 / 0.96 / 0.95 | 3 / 3 / 4 |
| Merged 16-bit | as is / swapped / no dates | 0.96 / 0.93 / 0.96 | 0.93 / 0.93 / 0.96 | 0.95 / 0.94 / 0.96 | 2 / 0 / 2 |
| Mac GGUF (q4_k_m) | as is / swapped / no dates | 0.98 / 0.94 / 0.96 | 0.96 / 0.96 / 0.96 | 0.97 / 0.95 / 0.96 | 1 / 0 / 2 |

The 10 pairs where the Mac file and the adapter differ: in 5 the Mac file is right (v-080, v-016, v-055, v-020, v-021), in 5 the adapter is right (v-101 twice, v-037 twice, v-059). Four of the adapter's wins are `unclear` pairs the Mac file called contradictions. With 46 contradictions in the validation set, one pair moves precision or recall by about 0.02, so these gaps are within a pair or two.

## The fix to try (Elias decides)

Merge the adapter into the **4-bit training weights, unpacked to 16-bit** (the exact numbers it was trained on), instead of the original 16-bit weights, then convert and squeeze to q4_k_m as before. That removes the cause of the larger part of the gap. The squeeze part (about 1.7%) stays, so passing 97% is likely but not certain. Cost: about 10 GPU minutes (about $0.15) and another 45-minute download. It departs from Unsloth's one-line export (it uses Unsloth's manual path: save the merge, then llama.cpp's converter), so it needs a yes first.

The other choice is to accept this file as it is: its validation scores are about the same as the adapter's, but it would ship below the 97% agreement bar Step 8 set.

## Files

- `valid/m1-judge-v1.*.jsonl`: Mac GGUF through Ollama's chat route; settings and speed in `m1-judge-v1.meta.json`.
- `valid/m1-judge-v1-raw.*.jsonl`: the same through the raw route with the training template's text.
- `valid/merged16.*.jsonl`: merged 16-bit model on Modal; settings in `merged16.meta.json`.
- `valid/compare-*.json`: each comparison in full, including which pairs differ.
- Adapter answers: `../trained-v1/valid/epoch-2.*.jsonl` (Step 7).
