# M1 judge v1 on the Mac: the GGUF file through Ollama, on the frozen test set

Date: 2026-10-23. Test set: `data/m1/test.jsonl`, 200 pairs (100 contradictions), SHA-256 `6e2a0832...4574b`, checked against `STATE.md` before the run. It was used once, for this one run, after the validation check (`validation-check.md`) and Elias's decision to ship this file (D-032).

The judge on the Mac is one GGUF file: the Step 7 adapter merged into Qwen3-4B-Instruct-2507 and squeezed to 4 bits (q4_k_m) by Unsloth's standard export. Ollama 0.24.0 ran it as `m1-judge:v1` from `ollama/Modelfile`, on an Apple M4 Pro, with the same prompt (SHA-256 `83d14d39...2865c`) and greedy answers (temperature 0). Before the first question, `scripts/run_ollama.py` checked through Ollama that it runs exactly this file, with the training chat template and the Modelfile's settings. Raw answers: `m1-judge-v1.<version>.jsonl`. Settings and timings: `m1-judge-v1.meta.json`.

## The file

| | |
|---|---|
| File | `m1-contradiction-judge-qwen3-4b-Q4_K_M.gguf` |
| Size | 2,497,280,320 bytes (2.5 GB), against about 8 GB for the 16-bit model |
| SHA-256 | `b6d930ce83c06512d12da2549b219edda8506e57404007acd877de29308172a9` |
| Where | private Hugging Face repo `eskora/m1-contradiction-judge-qwen3-4b-gguf`, commit `f9a1ae6ec40ba6b9e1c544b13a317fdf5a452360`; on the Mac in `~/models/m1-judge/` |
| How it was made | `scripts/export_gguf.py`; every setting in `export.meta.json` |

## The table

The adapter rows are the Step 7 results (`results/m1/trained-v1/summary.md`): the same adapter, on its 4-bit base, on a GPU on Modal.

| Model | Version | Precision | Recall | Accuracy | Type accuracy | Unclear | Invalid answers |
|---|---|---|---|---|---|---|---|
| **GGUF on the Mac** | as is | **0.98** | **0.96** | **0.97** | 0.95 (of 97) | 2 | **0** |
| **GGUF on the Mac** | swapped | **0.94** | **0.96** | **0.95** | 0.96 (of 94) | 2 | **0** |
| **GGUF on the Mac** | no dates | **0.98** | **0.94** | **0.96** | 0.93 (of 98) | 1 | **0** |
| Adapter on Modal | as is | 0.99 | 0.95 | 0.97 | 0.96 (of 98) | 3 | 0 |
| Adapter on Modal | swapped | 0.95 | 0.96 | 0.94 | 0.97 (of 92) | 6 | 0 |
| Adapter on Modal | no dates | 0.98 | 0.90 | 0.94 | 0.96 (of 98) | 3 | 0 |

Exact values for the GGUF (precision, recall, accuracy): as is 96/98, 96/100, 193/200; swapped 96/102, 96/100, 190/200; no dates 94/96, 94/100, 192/200.

**Aeverum's bar (precision 0.85, recall 0.80): passed on all three versions.** The lowest numbers are precision 0.94 (swapped) and recall 0.94 (no dates). All 600 answers are valid JSON.

## How close it is to the adapter

The two gave the same label on 584 of 600 test answers (97.3%): 196, 194 and 194 per version (`compare-test-adapter.json`). On validation it was 96.6% (287 of 297), which is why Step 8 stopped and Elias decided (D-032).

Where they differ (16 answers), the GGUF is right 10 times, the adapter 4 times, and both are wrong twice. Most of the GGUF's wins are contradictions the adapter dodged with `unclear` or explained away: the GGUF says `unclear` 5 times in 600 answers, against 12 for the adapter. That is why its recall without dates is higher (0.94 against 0.90). It pays with one more false alarm in the swapped version (6 against 5) and slightly lower type accuracy. One pair is 0.01 of recall, so each gap is a few pairs.

Why they differ at all (from the validation check): training ran the adapter on a 4-bit copy of the base model, and the export merges it into the original 16-bit weights, so some close calls tip the other way; squeezing to q4_k_m tips a few more. The chat template is the same as in training, byte for byte.

## Speed on the Mac

**0.91 seconds per pair** (mean of the 599 answers after the first; median 0.90 s), one answer at a time, on an Apple M4 Pro. The first answer took 3.7 s, because Ollama loads the model into memory first. All 600 answers took 9.2 minutes.

## What it shows

1. **The Mac file is a fair copy of the trained judge.** It passes the bar with room to spare, gives no invalid answers, and scores about the same as the adapter on a GPU: a little better on recall, a little worse on precision.
2. **It explains away less, but not never.** Its misses are mostly contradictions it calls `compatible_detail` (t-182 and t-144, "both can be true"), the same kind of mistake as t-047 in Step 7. It hardly ever dodges with `unclear` now.
3. **A merged QLoRA model is not exactly the adapter.** The 97% fidelity bar was set before this was known. The scores against the right answers are what Aeverum will feel.
