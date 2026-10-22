# Notice for the Aeverum project: a candidate judge is being trained

**Where:** the Cowork project "Aeverum", **new chat**.
Copy everything below the line into that chat.

---

This is a notice from Elias's separate project "Fine Tuning and RL training". **It is information, not a build instruction.** It does not authorize any change to Aeverum's code, Sources, DECISIONS, step plan or procedures.

**What is happening**
- A separate learning project (repo `~/code/model-training`, GitHub `e-skora/model-training`, private) is fine-tuning a small open model as a **candidate for Aeverum's night-pass judge** (02 §5.6; step 12 judge benchmark; step 13 night pass). It covers the `contradiction` kind only.
- **Model:** `Qwen3-4B-Instruct-2507` (Apache 2.0), fine-tuned with QLoRA using Unsloth on Modal GPUs. It will be delivered as a GGUF file for Ollama, running locally on Elias's Mac. Unsloth does not document an MLX export.
- **Input:** one person; memory A (text and written date); memory B (text and written date).
- **Output (JSON):** `{"label": "contradiction" | "non_contradiction", "type": null | "update" | "compatible_detail" | "merely_related", "reason": "<one sentence>"}`
- **Reading rules in its training data** follow DEC-034 (1) and (2):
  - Two claims that disagree about the same moment are a contradiction, even if they were written on different dates.
  - A later claim that describes a change is an `update`, which is not a contradiction. This project requires the later claim to say the change in words ("later", "now", "moved", and so on).
  - Two claims that can both be true are not a contradiction.
- **Data walls:**
  - All training and test data is newly made-up data about fictional people.
  - The project never reads Aeverum's repos or Sources, or the Mara Lindqvist corpus, including calibration, benchmark and holdout.
  - Whether Aeverum adopts the model is decided only by Aeverum's own step-12 benchmark, unchanged.
- **Status so far** (on the project's own 200-pair test set, not Aeverum's):
  - Untrained Qwen3-8B: precision 0.96, recall 0.94.
  - Untrained Qwen3-4B: precision 0.93, recall 0.83.
  - Goal: the trained 4B matches or beats the untrained 8B.

**Please do this**
1. Open the session the usual way (control files first). Then answer the questions below in plain language.
2. Do not change any Aeverum file, Source, decision or step plan because of this notice. If you think something in Aeverum should change, propose it to Elias the normal way.
3. Record this notice only where your own rules allow, and say where you recorded it.
4. Save your answers to `~/code/Aeverum-Auditer/.handoffs/2026-10-22-judge-coordination-reply.md` (that folder is git-ignored), so the model-training project can read them. If your rules don't allow that, say so.

**Questions**
1. Do the Sources or DECISIONS already define the judge interface's input and output for steps 12 and 13? If so, how does the shape above differ? Consider the fields, the labels, an "unclear" or `needs_manual_review` outcome, the stated reason, and the judge policy identity. If nothing is defined yet, is anything in the Sources a reason this shape would fit badly?
2. Will the judge see each memory's write time? The model relies on dates for the update rule.
3. Going by the rule text only, does DEC-034 (2)'s "different times means not a contradiction" apply only when the later claim describes a change, or to any pair written at different times? Do not share any benchmark or holdout pair text.
4. Is Ollama an acceptable local backend under §5.6? Is there any rule about the judge's model size, license or speed?
5. Is there anything else this project should know so the model can enter step 12 without changes on Aeverum's side?
