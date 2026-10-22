# M1 baseline v2: untrained models, new prompt, three versions of the test set

Date: 2026-10-22. Test set: `data/m1/test.jsonl`, 200 pairs (100 contradictions), SHA-256 `6e2a0832...4574b`, checked against `STATE.md` before the run and before scoring.

Why a second baseline: the prompt changed in Step 6 (D-025 to D-027). It now allows a third answer, `unclear`, says that the order of the two memories means nothing, and says that dates may be missing. So the Step 5 numbers (`results/m1/baseline/`) are no longer a fair "before" score for the trained model. These are.

Prompt SHA-256 (D-027): `83d14d39a2115eb1f4d2d461581673090ec25d0a6f05d444cb2da8b756a2865c` (from `judge.prompt.prompt_sha256()`, also saved in each `.meta.json`).

Each model answered all 200 pairs three times, once per version of the test set (D-026). The versions are made in memory from the frozen file, which is never changed:
- **As is:** the pairs as written, older memory first, with dates.
- **Swapped:** the same pairs with the two memories in the other order (newer memory first).
- **No dates:** the same pairs with both dates removed.

Each model ran on one L4 GPU, loaded in 4-bit with Unsloth 2026.10.7, with greedy decoding (`num_beams=1`, no sampling). Raw answers: `<model>.<version>.jsonl`. Settings, model commit and prompt fingerprint: `<model>.meta.json`. Scores come from `src/judge/score.py`.

| Model | Version | Precision | Recall | Accuracy | Type accuracy | Unclear | Invalid answers | Format errors |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | as is | 0.93 | 0.84 | 0.83 | 0.79 (of 82) | 0 | 22 | 0 |
| Qwen3-4B-Instruct-2507 | swapped | 0.79 | 0.88 | 0.75 | 0.77 (of 62) | 0 | 20 | 0 |
| Qwen3-4B-Instruct-2507 | no dates | 0.96 | 0.85 | 0.79 | 0.70 (of 73) | 0 | 28 | 0 |
| Qwen3-8B (thinking off) | as is | 0.95 | 0.96 | 0.96 | 0.60 (of 95) | 0 | 0 | 0 |
| Qwen3-8B (thinking off) | swapped | 0.73 | 0.93 | 0.80 | 0.53 (of 66) | 0 | 1 | 0 |
| Qwen3-8B (thinking off) | no dates | 0.94 | 0.96 | 0.95 | 0.72 (of 93) | 0 | 1 | 0 |

Exact values (precision, recall, accuracy):
- Qwen3-4B: as is 84/90, 84/100, 166/200; swapped 88/112, 88/100, 150/200; no dates 85/89, 85/100, 158/200.
- Qwen3-8B: as is 96/101, 96/100, 191/200; swapped 93/127, 93/100, 159/200; no dates 96/102, 96/100, 189/200.

| Model | GPU minutes | Cost |
|---|---|---|
| Qwen3-4B-Instruct-2507 | 6.6 | $0.09 |
| Qwen3-8B | 7.9 | $0.11 |

What the columns mean: the same as in `results/m1/baseline/summary.md`, plus:
- **Unclear:** how many times the model answered `unclear`. It is a valid answer but not a "contradiction" (D-025): it never enters precision, and it lowers recall when the right answer was contradiction. The test set has no `unclear` pairs, so every `unclear` answer is a wrong label.
- **Invalid answers** are still neither "yes" nor "no" (D-024). All but one of them used a type word as the label: Qwen3-4B wrote `"label": "update"` 22, 20 and 28 times (10, 6 and 5 of them on real contradictions, lowering recall). Qwen3-8B wrote `"update"` once (swapped) and `"merely_related"` once (no dates).
- **Cost:** minutes times the L4 price, $0.000222 a second (https://modal.com/pricing, checked 2026-10-22).

What it shows:
1. **Order matters to both untrained models, though it should not.** With the newer memory shown first, they call most updates contradictions: Qwen3-4B raised 21 false alarms on the 34 updates, Qwen3-8B 29. That is why precision falls to 0.79 and 0.73. The training set shows the newer memory first in half the examples (`src/judge/format.py`) to teach exactly this.
2. **Missing dates matter less.** Precision and recall stay close to "as is". Qwen3-4B gives more invalid answers without dates (28).
3. **Neither model ever says `unclear`.** Training has to teach it (about 5% of training pairs).
4. **The new prompt changed the "as is" scores a little** from Step 5: Qwen3-4B went from 0.93 / 0.83 with 10 invalid answers to 0.93 / 0.84 with 22; Qwen3-8B from 0.96 / 0.94 to 0.95 / 0.96.

The "before" score the trained Qwen3-4B must beat is now the Qwen3-4B rows above, on all three versions, and the goal (D-023) is to match or beat the Qwen3-8B rows.
