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

### D-033 (2026-10-23, Default): How M1 is described in public
Choices made while preparing the Hugging Face repos and the README (M1 Step 9). Nothing is public yet (D-007).
- **Dataset license: CC BY 4.0** (`cc-by-4.0`). Anyone may use the pairs, including in paid work, if they credit the author. It is the usual license for open datasets, and the data is all made up, so there is nothing to protect. The models keep the base model's Apache 2.0.
- **The public cards and the README do not name Aeverum or Mara Lindqvist.** They say "a personal memory app", and give its target (precision 0.85, recall 0.80) as that app's target. The pre-publish check fails if either name appears in a card, the README or a post. Many internal files (plans, decisions, reviews) do name Aeverum; whether those go public with the GitHub repo is Elias's call at the go-ahead.
- **Who wrote the data is said plainly:** "Claude, a language model made by Anthropic", rather than "an AI".
- **Card metadata:** the GGUF names `Qwen/Qwen3-4B-Instruct-2507` as `base_model` with `base_model_relation: finetune`; the adapter names the 4-bit copy it attaches to, `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`, with `base_model_relation: adapter`. That copy's own card names the Qwen original, so the Hub links all three.
- **The cards live in `hf/`** and go to Hugging Face only through `scripts/publish_hf.py`, which uploads only to private repos. A card edited on the website would be overwritten by the next upload, and the pre-publish check fails if the two differ.

### D-034 (2026-10-23, Elias): Publish M1
Elias's answer, "aa", to two questions: (1) publish the three Hugging Face repos (the recommendation was yes); (2) make the GitHub repo public as it is, option A, with its internal files that name Aeverum. The code gets the Apache 2.0 license (`LICENSE`), the same as the base model. This carries out D-007.

### D-034 (2026-10-23, Elias): Go public, GitHub included as it is
"yes and a", answering two questions: publish the three Hugging Face repos (yes), and make the GitHub repo public as it is, including the internal files that name Aeverum (option A). The code is under Apache 2.0 (`LICENSE`, the canonical text from apache.org), the same license as the base model. The dataset stays CC BY 4.0 (D-033).

### D-035 (2026-10-26, Elias): How Elias is described on the public cards
"instead of 'learning to fine tune open models' make it 'experimenting with fine tuning different types of models', (and give a link to my linkedin ...) - please write it so its still humble and names me but makes it a little more impressive, like im not some amateur". The two model cards now say: Made by Elias Skora (linked to https://www.linkedin.com/in/eliassskora/), an AI systems builder who experiments with fine-tuning different types of models. Uploaded to both Hugging Face model repos on 2026-10-26.

### D-036 (2026-10-26, Elias): M2 is RL on the same judge, with a harder test set first
"go", answering Claude's top suggestion in the Cowork project chat. The suggestion: make M2 reinforcement learning (GRPO) on the M1 judge with Unsloth on Modal, and build a harder test set first, because the M1 test set is too easy to show a real gain. Tinker (the same job on another service, compared side by side) and Together AI come after M2. This confirms D-015 and adds the hard test set. Spec: `milestones/M2-rl-judge/SPEC.md`.

### D-037 (2026-10-26, Default): RL starts from v1, on the same base model
GRPO starts from v1 (the epoch-2 adapter on `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`, D-031) and keeps training that adapter.
- **Why start from v1:** RL can only reward right answers the model already sometimes gives.
- **Why keep the base:** Unsloth's docs say its fast answer engine (vLLM) does not support Qwen3.5 (https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).
- **Not decided here:** Tinker retired Qwen3-4B-Instruct-2507 on 2026-06-12 (https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/index.md), and Together AI does not offer it (https://docs.together.ai/docs/fine-tuning/supported-models.md). Moving to Qwen3.5-4B is decided when the Tinker milestone is planned.

### D-038 (2026-10-26, Default): The hard test set
- **Size:** 300 pairs: 150 `contradiction`, 50 `update`, 50 `compatible_detail`, 50 `merely_related`. No `unclear` right answers, the same as M1.
- **Trap tags:** every pair has a `hard` tag naming its trap (list in prompt M2-01), so scores can be read per trap.
- **Text alone decides:** every pair can be decided from its text, so the no-dates version keeps one right answer.
- **Why 300:** with 150 contradictions, one pair moves recall by about 0.007 instead of 0.01, and the traps aim at v1's known weak spots.

### D-039 (2026-10-26, Default): Hard by design, not by filtering
The hard test set is written to fit trap categories. It is never picked by which pairs v1 gets wrong. Why: a test built from one model's failures makes that model look worse, and makes any retrained model look better by luck. The training pool may be filtered by v1's answers (SPEC Step 3), because that is how GRPO finds pairs to learn from.

### D-040 (2026-10-26, Default): Reward starting values and two answer styles
- **Reward table:** in `milestones/M2-rl-judge/SPEC.md`. A false alarm costs more than a miss, an honest `unclear` costs less than a wrong answer, and broken JSON costs the most. The values are fixed before training, and a change comes only from validation results, logged here.
- **Style A (main):** keeps v1's answer format exactly, so Aeverum can swap the model in with nothing else changing.
- **Style B (stretch):** adds a short `steps` field before the label, so the model reasons first. It is decided after Style A's full run.
- **No thinking mode:** Qwen3-4B-Instruct-2507 has none ("This model supports only non-thinking mode", https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), so reasoning must live in the answer.

### D-041 (2026-10-26, Default): M2 "Done means", and an honest no-gain result
- **The bar:** v2 must raise worst-case recall on the frozen hard test set by at least 0.03, with worst-case precision no more than 0.01 lower. It must stay within 0.02 of v1 on the M1 test set, with 0 invalid answers.
- **If it misses:** M2 still ends with a written note on what RL did and did not change, and v2 is not published as better.

### D-042 (2026-10-26, Default): A blind agent review checks the hard test set
A separate agent judges a sample blind (it answers before it sees the labels), as for the M1 training set (D-028), instead of Elias spot-checking 20 pairs as he did for the M1 test set (D-020). Why: the 2026-10-26 close-out lesson found that blind reviews caught real problems three times, and the lesson says to keep them for every dataset. Elias can ask for a spot-check as well at any time.

### D-043 (2026-10-26, Default): What Aeverum's direction reply changes here
Aeverum's operator answered the update (`.handoffs/2026-10-26-model-training-direction-reply.md`, SHA-256 `9dce39e447ddf3311f6ff0399a6aa1265b9fb1b704453d9d2873201aae28eb17`; copy in `docs/handoffs/2026-10-26-aeverum-direction-reply.md`). It is information only. Most of what M2 asked is "not set yet" on Aeverum's side. Adopted here:
- **Style A stays the main line** (D-040). If Style B is ever offered, its `steps` text must never reach a log.
- **All three versions stay** (as written, swapped, no dates), because Aeverum has not promised dates.
- **The bar is unchanged** (precision 0.85, recall 0.80). The 1.5-to-1 cost of a false alarm against a miss stays this project's own assumption; only Elias can set one for Aeverum.
- **`unclear` keeps a small cost on decidable pairs** and is never pushed to zero, because Aeverum prefers "not sure" over false certainty.
- **`type` is kept**, though Aeverum's step 12 scores only the label.
- **Frozen before step 12.** Every candidate is frozen and pinned before Aeverum's step 12 measures it. Step-12 results never steer training here unless Elias decides so, so Aeverum's benchmark never becomes training data.
- **Every candidate published with five pins:** GGUF SHA-256, Ollama digest, prompt SHA-256, Ollama version, and a download link where the file's SHA-256 can be checked.
- **New data rows carry `source` and `label_by`** (for example `"source": "synthetic"`, `"label_by": "claude"`), so a later model can mix made-up and verdict-based examples knowingly. Whether that mixing is allowed is Elias's decision, and it would also need an Aeverum product and Source change.
- **The Ollama server-log check** (no memory text in logs) must be settled before the judge runs on real memories. It is done in M2 Step 7.
- **Fixed:** `docs/run-judge-on-mac.md` said the Hugging Face repo is private; it is public.

### D-044 (2026-10-26, Elias): Far-apart pairs with no change shown stay contradictions
"use recommendation." The question: two memories written months apart disagree, and nothing in their words shows a change (for example "Lindiwe has a pet parrot" in March and "Lindiwe has no pets" in November). Is that a contradiction or an update? It stays a **contradiction**, by reading rule 1, as in the frozen hard test set (trap `far_apart_no_change`, 25 pairs).
- **Why:** nothing in the text shows a change, and without dates the judge cannot tell either way, so a flag lets Elias decide. It also matches Aeverum's "same moment" reading for favourites.
- **The known risk:** Aeverum's update reading rests on write times (direction reply, point 6). If Elias's Aeverum labels call such pairs updates, the judge will raise false alarms there at step 12. That is settled in Aeverum's label review before step 12, not here.

### D-045 (2026-10-27, Default): Drop the disputed M2 `unclear` pairs; reuse M1's checked ones
The blind review of M2 Step 3 (`reviews/M2-03-blind-review.md`) agreed with 199 of 200 sampled training pairs and 96 of 97 decided validation pairs, but with only 1 of the 33 `unclear` pairs. Its judges called the other 32 decidable, mostly "X never does A" followed by an event that may or may not be A, where one reading is more natural.
- **What was done:** as in D-030, the 32 were dropped, not relabelled, and kept in `data/m2/dropped-unclear.jsonl`.
- **Why:** `unclear` pairs a careful reader can decide would reward dodging, and dodging is v1's main miss.
- **Where `unclear` comes from instead:** the RL pool's `unclear` share is M1's 47 `unclear` training pairs, which two blind reviews already agreed on (D-030). They get `source` and `label_by` fields when added (D-043).
- **The pattern:** this is the third time AI-written `unclear` pairs drifted toward dodging (see LESSONS 2026-10-22).

### D-046 (2026-10-27, Default): Reward on `unclear` pairs, and how the RL pool is chosen
- **Reward when the right answer is `unclear`** (the D-040 table left this open):
  - `unclear` scores +1.0;
  - `non_contradiction` scores 0;
  - `contradiction` scores -1.5, because it claims a sure clash that the text does not support (a false alarm).
- **The RL pool, chosen by a rule fixed before any answer is seen:**
  1. v1 (the adapter) answers each candidate pair 8 times with some randomness (temperature 1.0).
  2. A pair joins the pool if its 8 rewards are not all the same. Only then can GRPO learn from it.
  3. Pairs where all 8 answers are right are added at 20% of the number of mixed pairs, chosen with a fixed seed, so v1 does not forget them.
  4. Pairs where all 8 answers are wrong in the same way are listed for Claude in Cowork to check for a wrong label, and are not trained on until checked.
- **The candidates:** `data/m2/train-hard.jsonl` (971) plus M1's 47 `unclear` training pairs (D-045). Never the frozen test sets, and never the validation files.

### D-047 (2026-10-27, Default): The always-wrong pairs after the blind review
- **Review:** `reviews/M2-04-always-wrong-review.md`. Two blind agents agreed with the file on 8 of the 9 always-wrong pairs and on all 11 control pairs.
- **tr-1527** (M1 `unclear`, "gets up early" / "alarm at half past seven") is disputed: both agents call it compatible_detail. It is dropped from RL use, not relabelled (as D-030, D-045), and never joins the pool. `data/m1/train.jsonl` is published and is not edited.
- **The other 8** have right labels; v1 is wrong on them. They stay out of the pool, because 8 equal rewards give GRPO nothing to learn from, and are kept as a watch list.
- **The pool is unchanged:** 314 pairs, SHA-256 `2c5c3bf2...dac4e77`.

### D-048 (2026-10-29, Default): The reward fingerprint covers whole source files
- **Why:** Codex's second review (`reviews/M2-04b-codex-review.md`) showed that following names inside functions still misses default values, captured values, helpers reached through a module, and the training wrapper `GRPOReward.__call__`. Each new form needs a new rule, so the name walk is replaced rather than extended.
- **The rule:** the fingerprint is the SHA-256 of the full text of `src/judge/reward.py` and of every `judge` module it imports, followed through their imports (today `judge.score`, `judge.prompt`, `judge.frozen`), in a fixed order, with each file's path. The imports are read from the code itself, not from a hand-made list. If one of those files loads code in a way that cannot be followed (`importlib`, `__import__`, `exec`, `eval`), the fingerprint stops with an error instead of skipping it.
- **What it does not cover:** Python itself and outside packages (such as `json` and `re`). Their versions stay in each run's environment record (`uv.lock`, the pinned image).
- **The cost:** any edit to those files, even a comment, gives a new fingerprint. That means an offline recheck of the saved pool (no GPU), as in `results/m2/pool/reward-recheck.md`. This is accepted: a fingerprint that moves too often is safe; one that misses a change is not.
