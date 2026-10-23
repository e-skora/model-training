# M1 trained judge v1: Qwen3-4B after fine-tuning, on the frozen test set

Date: 2026-10-22. Test set: `data/m1/test.jsonl`, 200 pairs (100 contradictions), SHA-256 `6e2a0832...4574b`, checked against `STATE.md` before the run and before scoring. It was used once, for this one run, after every training choice was made on the validation set.

The trained judge is the base model `unsloth/Qwen3-4B-Instruct-2507` (in the ready-made 4-bit copy Unsloth loads, see "Base model files" below) plus a LoRA adapter (the small add-on that training produces), loaded from the private Hugging Face repo `eskora/m1-contradiction-judge-qwen3-4b-lora` at commit `1d9ebbf66e1a2abcad9932932bb7d810a38c59b9`. It answered with the same prompt (SHA-256 `83d14d39...2865c`), the same greedy settings and the same answer loop (`judge.generate`) as the untrained baseline. Raw answers: `qwen3-4b-trained-v1.<version>.jsonl`. Settings and versions: `qwen3-4b-trained-v1.meta.json`. Training record: `train.meta.json`.

## The table

The untrained rows are copied from `results/m1/baseline-v2/summary.md`. Same columns, same meanings.

| Model | Version | Precision | Recall | Accuracy | Type accuracy | Unclear | Invalid answers | Format errors |
|---|---|---|---|---|---|---|---|---|
| **Qwen3-4B trained v1** | as is | **0.99** | **0.95** | **0.97** | **0.96 (of 98)** | 3 | **0** | 0 |
| **Qwen3-4B trained v1** | swapped | **0.95** | **0.96** | **0.94** | **0.97 (of 92)** | 6 | **0** | 0 |
| **Qwen3-4B trained v1** | no dates | **0.98** | **0.90** | **0.94** | **0.96 (of 98)** | 3 | **0** | 0 |
| Qwen3-4B untrained | as is | 0.93 | 0.84 | 0.83 | 0.79 (of 82) | 0 | 22 | 0 |
| Qwen3-4B untrained | swapped | 0.79 | 0.88 | 0.75 | 0.77 (of 62) | 0 | 20 | 0 |
| Qwen3-4B untrained | no dates | 0.96 | 0.85 | 0.79 | 0.70 (of 73) | 0 | 28 | 0 |
| Qwen3-8B untrained | as is | 0.95 | 0.96 | 0.96 | 0.60 (of 95) | 0 | 0 | 0 |
| Qwen3-8B untrained | swapped | 0.73 | 0.93 | 0.80 | 0.53 (of 66) | 0 | 1 | 0 |
| Qwen3-8B untrained | no dates | 0.94 | 0.96 | 0.95 | 0.72 (of 93) | 0 | 1 | 0 |

Exact values for the trained judge (precision, recall, accuracy): as is 95/96, 95/100, 193/200; swapped 96/101, 96/100, 188/200; no dates 90/92, 90/100, 188/200.

A reminder of the words: **precision** is how often "contradiction" was right when the judge said it; **recall** is how many of the 100 real contradictions it caught; **type accuracy** is how often it named the right kind of non-contradiction. One pair is 0.01 of recall, so a gap of one or two pairs is too small to mean much.

## The verdict

**Yes, it meets SPEC "Done means" item 1.**
- **Aeverum's bar (precision 0.85, recall 0.80):** passed on all three versions. The lowest numbers are precision 0.95 (swapped) and recall 0.90 (no dates).
- **0 invalid answers:** yes, on all 600 answers. The untrained 4B had 70.
- **Beats its untrained self:** yes, on precision, recall and accuracy in every version. The biggest gain is the swapped version, where precision went from 0.79 to 0.95. False alarms there fell from 24 to 5.
- **How close to the untrained 8B:** better precision in all three versions (0.99, 0.95, 0.98 against 0.95, 0.73, 0.94). Recall: as is 0.95 against 0.96 (one pair fewer), swapped 0.96 against 0.93 (three more), no dates 0.90 against 0.96 (six fewer). Type accuracy is far higher (about 0.96 against 0.53 to 0.72). Measured against the older Step 5 numbers the SPEC quotes (8B: 0.96 and 0.94), the trained 4B "as is" matches or beats both. Against baseline v2's 8B, recall is lower in two versions: as is (0.95 against 0.96) and without dates (0.90 against 0.96).

## The losses

Loss is the model's error score on the right answers: lower is better. It is measured only on the answer part (the JSON), not the prompt. "Training" is the loss on the pairs it learns from; "validation" is the loss on the 99 held-back pairs it never learns from.

| | Training loss (mean in the epoch) | Training loss (end of the epoch) | Validation loss (end of the epoch) |
|---|---|---|---|
| Before training | | | 2.75 |
| Epoch 1 | 0.54 | 0.46 | 0.30 |
| Epoch 2 | 0.21 | 0.14 | **0.29 (lowest)** |
| Epoch 3 | 0.10 | 0.09 | 0.34 |

In epoch 3 the training loss kept falling while the validation loss rose: the model started memorising the training pairs instead of learning the task (overfitting). The full curves are in `train.meta.json` (a training point every 2 steps, a validation point every 12 steps and at each epoch's end).

## How the trained version was chosen (validation set only)

Training saved an adapter after each of the 3 epochs. Each one answered the 99 validation pairs in all three versions. The pick followed a rule written down before any result was seen (`judge.train_data.CHOICE_RULE`): fewest invalid answers, then the best worst-case precision, then the best worst-case recall, then the best mean accuracy, then the earliest epoch.

| Adapter | Precision (as is / swapped / no dates) | Recall | Invalid | Unclear |
|---|---|---|---|---|
| epoch 1 | 0.93 / 0.93 / 0.92 | 0.91 / 0.93 / 0.96 | 0 | 0 |
| **epoch 2 (chosen)** | 1.00 / 0.98 / 0.98 | 0.93 / 0.93 / 0.91 | 0 | 10 |
| epoch 3 | 0.98 / 0.96 / 0.95 | 0.96 / 0.93 / 0.91 | 0 | 2 |

Epoch 2 is also the one with the lowest validation loss, so both signals agree. Raw validation answers: `valid/`.

## Mistakes

The trained judge got 7, 12 and 12 labels wrong in the three versions. Five examples:

1. **t-020** (all three versions): "Oscar has never owned a pet" against "Oscar had a rabbit as a child". It said `unclear` ("a child's rabbit may or may not count"). That is an invented escape, which reading rule 4 forbids. The same pattern caused most `unclear` answers on real contradictions (t-161, t-123, t-200).
2. **t-151** (no dates only): "Arjun switched to a window seat on the flight" against "Arjun sat in an aisle seat for the whole flight". Without dates it read "switched" as an update. Five of the ten contradictions missed without dates are like this (also t-041, t-060, t-115, t-183).
3. **t-047** (as is and swapped): "three pots of geraniums" against "four pots". It called it `compatible_detail` and explained the count away ("four is three plus one"). t-084 (two languages against three) is the same mistake.
4. **t-096** (swapped only): "piano lessons on Mondays", then "this school year, on Tuesdays", with the newer one shown first. It called this update a contradiction. This is the old order weakness, now much rarer: 3 of 34 updates in the swapped version, against 21 of 34 untrained.
5. **t-188** (all three versions): "Linh eats noodles for lunch most days" against "Linh eats pho for lunch most days". It raised a false alarm because it did not treat pho as a kind of noodle soup.

## What it shows

1. **Training fixed what it aimed at.** The order weakness is mostly gone (swapped precision 0.95), all answers are valid JSON, and the type is right about 96% of the time.
2. **Dropping dates is now the weak spot.** Without dates, a contradiction where one memory carries a change word ("switched", "since") can look like an update. This is the case `LESSONS.md` (2026-10-22) warned about.
3. **`unclear` sometimes dodges a real contradiction.** The model learned to say `unclear` (12 times across the three versions), and 8 of those 12 were real contradictions where it invented an escape. The prompt it was trained and scored with lists reading rules 1 to 3 only. Rules 4 and 5 (D-029) shaped the training labels but are not in the prompt text. Changing the prompt would change its fingerprint (D-027), so that is a decision for later, not part of this step.

## Base model files

Training, validation and this test run name the base files directly: the ready-made 4-bit copy `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at commit `7744afa8566e264af1a92a806d8d9aae00cc7c78`, and they record the name of what was really loaded. The untrained baseline v2 asked for `unsloth/Qwen3-4B-Instruct-2507` and recorded that repo's commit (`99206368...`). Two checks without a GPU say it most likely got the same files. The pinned Unsloth 2026.10.7 code replaces that name with the 4-bit copy when loading in 4-bit, drops the revision, and so reads the copy's `main` branch. On 2026-10-22, the day the baseline ran, `main` was `7744afa8...`: the copy has had no other upload since 2025-08-06. **This is an inference, not a verified fact:** the baseline did not record what it really loaded, and its log was not kept. The evidence, with sources and line numbers: `base-files.md`.

## GPU use for the whole step

About 76 GPU minutes on L4s, about $1.01 (an upper bound: three trial runs are counted at their full app time; L4 $0.000222 a second, https://modal.com/pricing, checked 2026-10-22), plus a few cents of CPU time. Line by line: `runs/RUNLOG.md`.
