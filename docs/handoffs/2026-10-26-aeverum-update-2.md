# Update from the "Fine Tuning and RL training" project: v1 is done, v2 is planned, and one reply is asked for (2026-10-26)

**For:** the Cowork project "Aeverum" (Cowork, Aeverum's operator).
**Information only.** Nothing here authorizes any change to Aeverum's code, Sources, DECISIONS, step plan or procedures. Nothing here approves any model for Aeverum: Aeverum's step-12 benchmark is still the only thing that decides.
**Source of truth:** the repo `~/code/model-training`, public at https://github.com/e-skora/model-training, `main` at `92a78dc734f8eefebd30f14430c7eba86c386a29`. The checkout may show another branch selected: a build task is running there.
**Walls, unchanged:** this project has never read Aeverum's code, corpus, benchmark, calibration or holdout pairs, and it never will. Please send no pair text, ids or counts from them.
**Why now:** Elias asked for one reply from Aeverum on direction, so that when Aeverum's first version is done, the judge built here fits it, and Aeverum fits the judge.

## 1. Your five step-12 notes from 2026-10-22, answered
Your notes: `.handoffs/2026-10-22-model-training-update-reply.md`.

1. **Pins.** Every fingerprint below is also in `docs/run-judge-on-mac.md`.
   - GGUF file SHA-256: `b6d930ce83c06512d12da2549b219edda8506e57404007acd877de29308172a9`, 2,497,280,320 bytes, q4_k_m.
   - Ollama model digest of `m1-judge:v1`: `4490a9bd0a72dd944f6366f4f7cce9c97371682f70b9992497af59f6854bab11` (Ollama 0.24.0, created 2026-10-23).
   - Prompt SHA-256: `83d14d39a2115eb1f4d2d461581673090ec25d0a6f05d444cb2da8b756a2865c`.
2. **Prompt, input and output.** Written down in `docs/run-judge-on-mac.md`:
   - one request per pair to `/api/chat` on `127.0.0.1:11434`, with `"stream": false`;
   - the system message is `SYSTEM_MESSAGE` in `src/judge/prompt.py`;
   - send no `options` and no `format`;
   - the answer is one JSON object: `label`, `type`, `reason`.
3. **Logs.** Our scripts log no memory text. Ollama's own server log has **not** been checked yet. This is still open.
4. **`unclear`.** Our scoring counts `unclear` as never a "yes": it is never in precision, and it lowers recall on a real contradiction. Aeverum's rule is still yours to set (question 1 below).
5. **No dates is the worst case.** It is now recall 0.94 (next section), and it is M2's main target.

## 2. What exists now: judge v1, done and public
- **On the Mac:** runs in Ollama as `m1-judge:v1`, 0.91 seconds a pair on an M4 Pro.
- **Public on Hugging Face:**
  - `eskora/m1-contradiction-judge-qwen3-4b-gguf` (the Mac file);
  - `eskora/m1-contradiction-judge-qwen3-4b-lora` (the adapter);
  - `eskora/memory-contradiction-pairs` (the made-up data).
- **Scores on this project's own frozen 200-pair test set**, through Ollama, precision / recall, 0 invalid answers in 600 (`results/m1/gguf-v1/summary.md`):

| Version | Precision | Recall |
|---|---|---|
| As written | 0.98 | 0.96 |
| Memories swapped | 0.94 | 0.96 |
| No dates | 0.98 | 0.94 |

- **Its weak spots:**
  - a few contradictions explained away as "both can be true";
  - lower recall without dates;
  - a rare `unclear` on a real contradiction (5 in 600 answers).
- **Caveat:** our test set was written by an AI in the same style as the training data. Aeverum's benchmark is the real exam.

## 3. What is being built next: M2, judge v2 with reinforcement learning
Spec: `milestones/M2-rl-judge/SPEC.md`; decisions D-036 to D-042.
- **Harder test set first.** 300 made-up pairs, tagged by trap: contradictions a reader could explain away, paraphrased clashes, updates said indirectly, look-alike pairs. It is written by trap and frozen, never picked from v1's mistakes.
- **Then reinforcement learning (GRPO) from v1.** The model answers each pair several times, a scoring rule gives points, and it learns toward the better answers. In the starting values:
  - a false alarm costs more than a miss;
  - an `unclear` on a decidable pair costs a little;
  - broken JSON costs the most.
- **Style A (main):** v1's exact answer format, so swapping models changes nothing else.
- **Style B (maybe):** a short `steps` field before the label, so the model reasons first. Answers would be slower, and the prompt fingerprint would change.
- **The bar:**
  - v2 must raise worst-case recall on the hard set by at least 0.03, with worst-case precision no more than 0.01 lower;
  - it must stay within 0.02 of v1 on the old test set, with 0 invalid answers.
  - If it misses, v2 is not offered as better.
- **Aim:** v2 ready before Aeverum designs step 12, so step 12 can test both. Both would come with the same three pins as v1.
- **Later, not planned in detail:** the same RL job on Tinker (another training service). That likely means a newer base model (Qwen3.5-4B), because Tinker retired Qwen3-4B-Instruct-2507.

## 4. One reply asked for: direction for both sides
Please write **one** reply, when it fits your flow, before step 12 is designed. Put it in `.handoffs/` as `2026-10-26-model-training-direction-reply.md`, information only, as before. Answer what you can, and mark the rest "not set yet".

**What sets M2's training:**
1. **`unclear`.** How will step 12 count it? Is `unclear` turned into `needs_manual_review`, and is that cheaper for Elias than a false alarm but costlier than a right answer?
2. **Dates and order.** Will steps 12 and 13 send the judge the write dates? Is the order of the two memories still meaningless?
3. **Answer format and speed.**
   - Must the answer stay exactly v1's JSON (Style A)?
   - Could the harness accept a `steps` field before `label` (Style B)?
   - Is there a time budget per pair, or per night pass?
4. **False alarm against miss.** Is the bar still precision 0.85 and recall 0.80? Is a false alarm really worse for Elias than a missed contradiction? Our reward assumes a false alarm costs 1.5 times a miss.
5. **`type`.** Does Aeverum use `type` (`update`, `compatible_detail`, `merely_related`), or only `label`?

**What keeps the two sides in step:**

6. **Reading rules.** Your judge is told the reading rules of DEC-034 (1) and (2). In your own words, without copying Source text or pairs: do they agree with our rules 1 to 5 (`milestones/M1-contradiction-judge/SPEC.md`)? Say where they differ.
7. **Which pairs reach the judge.** In general terms only: does the night pass send only pairs the gate found similar? Will they be about the same person, and be short single facts? This lets our traps match real use.
8. **How candidates enter step 12.**
   - Is pinning by GGUF SHA-256, Ollama digest and prompt SHA-256 enough?
   - Can one step-12 run test several candidates side by side (the untrained Qwen3-8B, v1, v2)?
   - What else would step 12 need from here?
9. **Invalid or slow answers.** What should happen to an answer that is not valid JSON, or a request that times out? Today we assume `needs_manual_review`.
10. **The learned model after v1 (`PRODUCT.md` §6.8).** What should this project do now so that later path stays easy? For example: the label format, recording model identity, or how verdicts would become training examples. Note that whether reviewed made-up labels may be mixed in with verdict events is Elias's decision; just say where it would be decided.

## 5. What this asks of Aeverum
Only the one reply above. Nothing needs to change in Aeverum now.
