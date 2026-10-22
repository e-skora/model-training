# M1 baseline: untrained models on the frozen test set

Date: 2026-10-22. Test set: `data/m1/test.jsonl`, 200 pairs (100 contradictions), SHA-256 `6e2a0832...4574b`, checked before the run.

Each model read every pair with the same prompt (`src/judge/prompt.py`), loaded in 4-bit with Unsloth 2026.10.7 on one L4 GPU, and answered with greedy decoding (always the most likely next word, no sampling). Raw answers are in the `.jsonl` files next to this one. Scores come from `src/judge/score.py`, which checks the test set's SHA-256 against `STATE.md` before scoring.

Repeatability: these three runs predate the fuller run records. Their `.meta.json` files do not save the generation settings, the chat-template options, the exact model and tokenizer commit, or the prompt as the model saw it, and the code did not yet set `num_beams=1` (one candidate answer at a time) explicitly; it relied on each model's default. From now on, `scripts/run_baseline.py` sets `num_beams=1` and saves all of these, so a rerun can be checked against them. The answers were rescored on 2026-10-22 with the stricter scorer (no GPU): precision, recall and accuracy did not change.

| Model | Precision | Recall | Accuracy | Type accuracy | Invalid answers | Format errors | Minutes | Cost |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 0.93 | 0.83 | 0.88 | 0.88 (of 92) | 10 | 0 | 3.0 | $0.04 |
| **Qwen3-8B** (thinking off) | **0.96** | **0.94** | **0.95** | 0.66 (of 96) | 0 | 0 | 3.4 | $0.05 |
| Phi-4-mini-instruct | 0.89 | 0.85 | 0.88 | 0.89 (of 90) | 0 | 90 | 2.6 | $0.03 |

Exact values: Qwen3-4B precision 83/89, recall 83/100, accuracy 175/200; Qwen3-8B 94/98, 94/100, 190/200; Phi-4-mini 85/95, 85/100, 175/200.

What the columns mean:
- **Precision:** when the model said "contradiction", how often it was right.
- **Recall:** of the 100 real contradictions, how many it caught.
- **Accuracy:** how many of the 200 labels it got right.
- **Type accuracy:** for the non-contradictions it labelled right, how often it also named the right type (update, compatible detail, merely related). The number in brackets is how many pairs that covers.
- **Invalid answers:** answers the scorer cannot read: not JSON (including `NaN` or `Infinity`, or a field name written twice) or a label that does not exist. An invalid answer is neither "contradiction" nor "not a contradiction" (D-024). It lowers recall when the right answer was contradiction, and it lowers accuracy. It does not enter precision, because Aeverum turns an unreadable answer into "needs manual review", not a false alarm. That is why this count always sits next to precision and recall, and why the trained model must have 0. All 10 from Qwen3-4B used "update" as the label instead of as the type; 8 of those pairs were real contradictions (lowering recall) and 2 were not.
- **Format errors:** answers with a readable label that still break the prompt's other rules: "type" must be JSON `null` for a contradiction and one of the three types otherwise, and "reason" must be non-empty text. They do not change precision, recall or accuracy. All 90 from Phi-4-mini are contradiction answers with a type that is not `null`: 50 wrote the text `"null"`, 29 wrote "update" and 11 "compatible_detail".
- **Minutes:** time from asking Modal for the GPU until the answers came back, including loading the model. **Cost:** minutes times the L4 price ($0.000222 a second, https://modal.com/pricing); CPU and memory add a little on top.

Notes:
- Qwen3-8B got 10 labels wrong: it missed 6 contradictions and raised 4 false alarms.
- Phi-4-mini wrapped 5 answers in a markdown code box (```json ... ```). The scorer accepts that and counts it; it is not counted as invalid.
- All three untrained models already pass Aeverum's bar (precision 0.85, recall 0.80) on this test set. Qwen3-8B passes it easily. Training has to beat 0.96 and 0.94 to count as better than the base model.
- Qwen3-8B's main weakness is the type: it often calls "merely related" pairs "compatible detail", and "compatible detail" pairs "update".

## Three mistakes from the best model (Qwen3-8B)

1. **t-012**, right answer: not a contradiction (compatible detail). Model said: contradiction.
   A (2026-05-01): "Cyrus gets up before seven on weekdays." B (2026-05-01): "Cyrus gets up around 6:30 on weekdays."
   Note: 6:30 is before seven, so both are true; the model saw two different times and called it a clash.

2. **t-086**, right answer: contradiction. Model said: not a contradiction (update).
   A (2025-02-02): "At the wedding on 1 February, Selam gave the first speech, and her brother spoke later." B (2025-04-09): "At the wedding on 1 February, Selam spoke last, after her brother and both parents."
   Note: a past event cannot change, so a later memory cannot update it; the model read the later date as a change.

3. **t-036**, right answer: not a contradiction (merely related). Model said: contradiction.
   A (2025-10-01): "Yuki works from home on Fridays." B (2025-10-01): "Yuki works from home on Mondays."
   Note: someone can work from home on both days; the model treated two different days as two answers to one question.
