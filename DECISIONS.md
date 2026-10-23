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

### D-023 (2026-10-22, Elias): Train the 4B to beat the untrained 8B
"lets go with your rec". This replaces D-022. The base model for training is `unsloth/Qwen3-4B-Instruct-2507` (Apache 2.0). Untrained, it scored precision 0.93, recall 0.83, with 10 invalid answers. The goal is for the trained 4B to match or beat the untrained Qwen3-8B (precision 0.96, recall 0.94) on the frozen test set. Why: a clear before-and-after gain that can be proven, cheaper training and much cheaper RL later, and a faster model on the Mac for Aeverum's nightly runs.

### D-024 (2026-10-22, Default): Invalid answers and the scores
Precision and recall stay standard, the same definitions Aeverum's bar uses (02 §8.4). An invalid answer is neither a "yes" nor a "no". It lowers recall when the right answer was contradiction, and it lowers accuracy, but it does not enter precision, because Aeverum turns an unreadable judge answer into "needs manual review", not a false alarm. So the invalid-answer count is always reported next to precision and recall, and the trained model must have 0 invalid answers. This answers Codex review finding 1 (`reviews/M1-05-codex-review.md`). Claude disagrees with folding invalid answers into precision, because it would mix two different failures into one number.

### D-025 (2026-10-22, Default): A third answer, `unclear`
Aeverum must be able to stage "needs manual review" instead of guessing (Aeverum operator's reply, `~/code/Aeverum-Auditer/.handoffs/2026-10-22-judge-coordination-reply.md`, SHA-256 `9722025d39bec4273ed4512955e0ee5511f71de7f64f878da74f1f6642565fd6`). So the model may answer `"label": "unclear"` when the text alone cannot settle the pair. About 5% of training pairs teach this. For scoring, `unclear` is not a "yes": it never enters precision, it lowers recall when the right label is contradiction, and it is counted on its own line. The frozen test set does not change.

### D-026 (2026-10-22, Default): Order means nothing, and dates may be missing
Aeverum stores the two memories in no particular order and has not promised the judge will see write dates (same reply). The data files keep the earlier memory as `a`. When building training examples, the later memory is shown first about half the time, and dates are left out of about a third of pairs, chosen with a fixed seed. Scoring adds two derived versions of the frozen test set, made in memory without editing the file: order swapped, and dates removed.

### D-027 (2026-10-22, Default): Fixed, fingerprinted prompt and settings
The prompt and the generation settings stay fixed and get a SHA-256 each, so Aeverum can name the judge's policy identity. Step 8 records the model file's SHA-256. Whatever runs the model must not log memory text (Aeverum §12.4); that is Aeverum's side, noted here so we don't build anything that depends on logging.

### D-028 (2026-10-22, Elias): No spot-check of the training set
"skip the spot checking step". Elias does not spot-check the training set. Instead, Claude in Cowork has a separate agent judge a sample blind before Step 6 merges, as it did for the test set.

### D-029 (2026-10-22, Default): Two more reading rules, so `unclear` stays rare and honest
A blind review of the training set (`reviews/M1-06-blind-review.md`) found that about half the `unclear` pairs could actually be decided, and that one pattern was labelled two different ways. Two rules were added to the SPEC. Rule 4: do not invent an escape; different current facts about the same thing are a contradiction, and `unclear` is only for pairs a truly missing fact decides. Rule 5: different days, times or places with no exclusive word are not a contradiction. Both rules agree with the frozen test set (for example t-036 is `merely_related`).

### D-030 (2026-10-22, Default): Keep only `unclear` pairs two judges agree on
After two rounds of fixes, about 31 of 80 `unclear` pairs were still disputed (`reviews/M1-06-blind-review-r2.md`). They were dropped, not relabelled. Train is now 1,522 pairs with 47 `unclear` (3%); valid is 99 pairs. A smaller, trustworthy `unclear` set is better than a larger one that teaches the model to dodge, because precision matters most to Aeverum.

### D-031 (2026-10-22, Default): Trained judge v1 is the epoch-2 adapter
Training (M1 Step 7) saved a LoRA adapter after each of 3 epochs. The validation set alone chose epoch 2, by a rule fixed before any result (`judge.train_data.CHOICE_RULE`: fewest invalid answers, then worst-case precision, then worst-case recall). It also had the lowest validation loss. Settings came from Unsloth's notebook for this model and its LoRA guide, unchanged: the validation set was used to pick the epoch, not to tune other settings, because the first run already passed the bar on validation. The base files are `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at commit `7744afa8566e264af1a92a806d8d9aae00cc7c78`, the copy Unsloth really loads. Results: `results/m1/trained-v1/summary.md`.

### D-032 (2026-10-23, Elias): Ship the q4_k_m GGUF as it is, below the 97% agreement bar
"Accept the current Mac file below the 97% bar and go on to the test-set run." The GGUF (`b6d930ce...72a9`, Unsloth's standard export) gave the same label as the Step 7 adapter on 287 of 297 validation answers (96.6%), with 0 invalid answers. The gap comes mostly from the merge into the original 16-bit weights, then the q4_k_m squeeze, not from the template; validation scores stay about the same (`results/m1/gguf-v1/validation-check.md`). The re-export that merges into the 4-bit training weights is not done. The frozen test set is run once, through Ollama, to report this file's score.
