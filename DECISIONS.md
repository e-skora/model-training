# DECISIONS.md

A dated record of what is decided. Entries are never edited after they are written. A later entry replaces an earlier one by saying so.

"Elias" entries quote his words. "Default" entries are sensible defaults Claude chose; Elias can override any of them at any time.

### D-001 (2026-10-20, Elias): The tool plan
"1. Unsloth (free open-source fine-tuning software) running on Modal ... 2. Publish every model I make to Hugging Face, so I build public proof of work. 3. Later steps, not now: Together AI (managed fine-tuning, the way companies buy it), then Tinker by Thinking Machines (reinforcement learning practice)."

### D-002 (2026-10-20, Elias): Purpose
"The purpose of this project is for me train useful models/make useful tools, and to learn how to do this."

### D-003 (2026-10-20, Elias): Speed
"i want to finish the fine tuning quickly so i can move on to rl traingin"

### D-004 (2026-10-21, Elias): First model is the one Aeverum needs
"lets build the one that aeverum will need eventually", then "go". That model is the contradiction judge for Aeverum's night pass: it reads two memories and says whether they contradict each other. Whether Aeverum adopts it is decided at Aeverum's step 12 (judge benchmark), not here.

### D-005 (2026-10-20, Elias): Fewer questions
"too many questions - i dont feel like answering any of those now." Ask only what blocks the next step; pick defaults for the rest and log them here.

### D-006 (2026-10-21, Default): Repo location
`~/code/model-training`, next to his other projects. Not in `~/Documents`, because Documents syncs to iCloud, which can corrupt git folders.

### D-007 (2026-10-21, Default): GitHub repo is private until M1 ships
Repo `e-skora/model-training`, private. It goes public when the first model is published (one command; Claude does it when Elias says so).

### D-008 (2026-10-21, Default): Made-up data only
Training and test data are made-up memory pairs about fictional people. Never Mara Lindqvist, never text from Aeverum's corpus, never real memories.

### D-009 (2026-10-21, Default): Our own frozen test set
This project writes its own test set before any training and freezes it. Aeverum's benchmark and holdout pairs are never touched from here.

### D-010 (2026-10-21, Default): Base model
A small open model, 3 to 8 billion parameters, with a license that allows commercial use (Apache 2.0 or MIT). Picked by testing two or three candidates on our test set before training (M1 Step 5).

### D-011 (2026-10-21, Default): Training method and GPU
QLoRA with Unsloth on a Modal L4 GPU. A bigger GPU only if the L4 runs out of memory.

### D-012 (2026-10-21, Default): Who writes the made-up data
Claude Code writes it in batches, so there is no extra API cost. Elias spot-checks 20 pairs from each set.

### D-013 (2026-10-21, Default): Who does what
Claude Code builds. Codex reviews the code-heavy steps (testing and training scripts). Claude in Cowork plans, checks results on disk, merges and pushes.

### D-014 (2026-10-21, Default): Spending
$0 beyond Modal's free $30 a month for M1.

### D-015 (2026-10-21, Default): Next milestone is RL on the same judge
M2 trains the same judge with reinforcement learning (GRPO) using Unsloth on Modal, the same setup as M1. Together AI and Tinker come after; their order is set later.

### D-016 (2026-10-21, Default): How deep the learning goes
Practical first: each idea is explained when it comes up. Each milestone ends with a short "why it works" note in `LESSONS.md`.

### D-017 (2026-10-21, Default): Claude Code stays in the desktop Code panel
No terminal Claude Code for now. Revisit when long jobs need to run unattended.

### D-018 (2026-10-21, Default): Proof of work
Each published model gets a Hugging Face model card, its dataset on Hugging Face, the code on GitHub, and one LinkedIn post that Claude drafts and Elias edits.

### D-019 (2026-10-21, Elias): No Modal spending cap
"skipping the modal part, forget about it dont worry about cap". No workspace budget is set in Modal. D-014 ($0 beyond the free credits) stays the goal; the per-run stop in each prompt (ask before anything over $1) stays.

### D-020 (2026-10-22, Elias): Test set accepted
Elias spot-checked 20 of the 200 test pairs and marked all 20 OK ("spot check done"). The test set is frozen at SHA-256 `6e2a08323f445477f9600e371f1074ad3e2b41ef97d7072d6734ef481f6e574b`.

### D-021 (2026-10-22, Default): Shared prompt and greedy scoring
One function in `src/judge/prompt.py` builds the model's input for scoring, training and later use. Scoring uses greedy decoding, so a model always gives the same answer to the same pair. An answer that is not valid JSON counts as wrong.

### D-022 (2026-10-22, Default): Base model is Qwen3-8B
`unsloth/Qwen3-8B` (Apache 2.0), with thinking turned off (`enable_thinking=False` in the chat template, as Qwen's model card says). Why: on the frozen test set, untrained, it had the best contradiction precision (0.96) and recall (0.94) of the three candidates, ahead of Qwen3-4B-Instruct-2507 (0.93 and 0.83) and Phi-4-mini-instruct (0.89 and 0.85). The gap is too big to call close, so the "choose the smaller one" rule does not apply. It fits training on an L4: in 4-bit it used at most 10.2 GiB of the L4's 22.1 GiB while answering, and Unsloth lists about 6 GB as the minimum for QLoRA on an 8B model (https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements). Full results: `results/m1/baseline/summary.md`.
