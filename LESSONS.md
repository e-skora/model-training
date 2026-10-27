# LESSONS.md

Lessons and mistakes, dated, so nothing gets relearned. Add to the bottom. Never edit an old entry.

## 2026-10-20: Mistake: the setup interview asked too much
The first setup interview asked 8 questions plus a list of defaults. Elias did not want to answer them.
**Lesson:** ask only what blocks the next step. Choose sensible defaults for everything else and write them in `DECISIONS.md`, where he can override them.

## 2026-10-21: Lesson: pick a first model whose answers a program can score
Reinforcement learning (RL) only works when a program can score the model's answers by itself. Picking a fine-tuning task with one checkable answer (a label) means the same task carries straight into RL.

## 2026-10-21: Lesson: what M1 Step 3 taught about Modal
- The first run of a new container image is slower because Modal builds it (here about 36 seconds of building). Later runs reuse the image.
- Modal's Unsloth example builds its image with Python 3.11. That is fine next to our local Python 3.12: the image is a separate computer setup.
- Modal's Unsloth example has no Hugging Face secret. We add our own (`huggingface`, key `HF_TOKEN`) so later jobs can upload models.
- Our free Modal plan does not show cost per run. Estimate it from the run's minutes times the GPU price, and log it in `runs/RUNLOG.md`.
- Newer Hugging Face tokens can be hundreds of characters long, not the short ~37 character ones. Check a token with `hf auth whoami`, not by its length.

## 2026-10-21: Mistake: started a run before its file existed
Claude Code started `modal run scripts/hello_gpu.py` while the file write was still blocked, so the run failed in 1 second before reaching Modal (no cost).
**Lesson:** confirm the file is on disk before starting a run.

## 2026-10-21: Lesson: where compatible_detail ends and merely_related begins
Rereading batch 1 of the M1 test set caught a mislabel. "Works from home on Fridays" and "works from the office on Thursdays" had been marked `compatible_detail`, but they are two separate facts, so they are `merely_related`.
**Lesson:** `compatible_detail` means the second memory adds detail to the same fact ("likes tea", then "drinks green tea in the morning"). `merely_related` means a different fact on the same topic ("hikes", then "owns hiking boots"). The rereads between batches are worth doing.

## 2026-10-21: Mistake: a shuffle that could not finish
The first shuffle of the test set kept reshuffling all 200 pairs until no label came more than 3 times in a row. Pure chance almost never gives that, so it ran until Claude Code stopped it (nothing was written, no harm done).
**Lesson:** when the order has to follow a rule, build it one step at a time with the rule built in, rather than reshuffling and hoping. Also: macOS has no `timeout` command.

## 2026-10-21: Mistake: the first test set followed obvious templates and keyword patterns; an independent check caught it before freezing.
**Lesson:** ask for hard cases by name, check for shortcut patterns, and have a second agent review before freezing.

## 2026-10-22: Lesson: the Unsloth pins from mid-2025 were too old
Modal's Unsloth example pins Unsloth 2025.7.8 and transformers 4.54.0. With them, `unsloth/Qwen3-4B-Instruct-2507` and `unsloth/Phi-4-mini-instruct` would not load: Unsloth stopped with "not supported in your current Unsloth version" (Qwen3-8B loaded). Unsloth 2026.10.7, the newest on PyPI, also refuses transformers 4.54.0. The fix: pin Unsloth 2026.10.7 and let `uv pip compile` (for Linux, Python 3.11) pick the matching versions of torch, transformers, trl and the rest, then write them all in one place, `src/judge/modal_image.py`. torch 2.14.1 brings CUDA 13.0, which Modal's driver supports (up to 13.3). `scripts/hello_gpu.py` still works on the new image.
**Lesson:** an example's pins age fast. Do a smoke test on a few pairs first: it found the problem for about $0.03.

## 2026-10-22: Lesson: the untrained models already pass Aeverum's bar on our test set
Untrained, Qwen3-8B scored precision 0.96 and recall 0.94 on the frozen test set, above the 0.85 and 0.80 bar. The other two candidates passed it too. So on this test set there is little room left for training to improve the contradiction label. The clearest room is the type (Qwen3-8B got it right only 66% of the time), and Aeverum's own benchmark may be harder than ours.
**Lesson:** score the base model before planning training, so the goal is measured against a real starting point.

## 2026-10-22: Lesson: a safety check belongs in the shared loader, not in one script
The frozen-file check lived only in `scripts/run_baseline.py`, so the scoring command could score an edited test set without warning. Codex caught it. Now every reader of the test set goes through `judge.frozen.load_frozen`, which checks against `STATE.md` itself.
**Lesson:** put a check where the data is loaded, so every path gets it, and read the expected value from the one place it is recorded instead of copying it.

## 2026-10-22: Lesson: current transformers does not remember which model commit it loaded
The loaded config and tokenizer no longer carry the Hugging Face commit (`_commit_hash` is dropped). To record which files a run used, look the commit up first (`HfApi().model_info(repo).sha`) and load that exact commit with `revision=`.
**Lesson:** to make a run repeatable, choose and save the exact version before the run, rather than hoping to read it back afterwards.

## 2026-10-22: Lesson: untrained models read order as time
On the frozen test set with the newer memory shown first, both untrained models called most updates contradictions (Qwen3-4B 21 of 34, Qwen3-8B 29 of 34), and precision fell from about 0.95 to 0.73-0.79. Without dates they did almost as well as with them. So order, not dates, is the weak spot, and the training formatter shows the newer memory first in half the examples to teach it.
**Lesson:** test the conditions the real user creates (Aeverum stores pairs in no order), not only the tidy version of the data.

## 2026-10-22: Lesson: when dates can vanish, a contradiction must not look like an update
If dates are hidden, a contradiction where only the newer memory has change wording ("Ravi now lives in Pune" after "Ravi lives in Goa") reads exactly like an update. One of the helper agents caught this in its own batches and rewrote 8 pairs; a search of all 1,600 pairs then found 5 more candidates, all of them real contradictions (the change word was about something else).
**Lesson:** when the training format hides or shuffles information, write each label so it is still right without that information, and search for the cases that break.

## 2026-10-22: Lesson: splitting data writing across helper agents works with one shared brief and a checker
Eight helper agents each wrote 200 pairs from one written brief (rules, hard cases by name, counts, a separate range of names and topics) and ran the checker after every batch of 50. All 8 files passed on their own; merging needed only small fixes: `unclear` pairs had too few change words (the brief asked for them in every other label, not in `unclear`), and one full name came up twice across agents.
**Lesson:** put every count and share the checker enforces into the brief itself, and check the merged file again, because some rules (repeated names, shares across the whole set) only show up after merging.

## 2026-10-22: Mistake: a smoke test failed on a typo before reaching the GPU
The first run of the updated baseline runner stopped on the Mac with a `NameError` (a renamed loop variable), before Modal started any GPU. No cost.
**Lesson:** the runner's local part has no test. A quick `--limit 8` smoke test caught it for free; keep doing that before every full run.

## 2026-10-22: Mistake: `unclear` was used as a way out of hard calls
The blind review found that about half the training set's `unclear` pairs could be decided. Their reasons invented an escape ("maybe she has two dogs", "maybe the inside is yellow and the outside white"), or they treated two activities on the same day with no times given as a possible clash. Almost every reason also started with the same words ("It depends on whether..."). Training on them would teach the model to dodge real contradictions. The fix (M1-06b) relabelled 51 of 79, wrote 52 new ones that are truly undecidable, and added a checker rule for the reason openings.
**Lesson:** a "don't know" label needs a test written down before any pair is written: a fact that is truly missing decides it, and neither reading is more natural. Give writers the near-misses by name (invented escapes, same-day activities with no times), and check the reasons for templates, not only the memories.

## 2026-10-22: Lesson: AI-written "unclear" examples drift toward dodging
Twice, Claude Code wrote `unclear` pairs that a blind second judge could decide. The first time, the reasons invented escapes ("maybe two dogs"). The second time, they argued over what a word means. Labels for contradiction and non-contradiction were almost perfect both times.
**Lesson:** for any "I'm not sure" label, keep only the examples two separate judges agree on, and drop the rest instead of arguing them into a label. Have a blind second agent check every dataset before training. It caught problems three times in M1.

## 2026-10-22: Lesson: a check on the training text caught a hidden template change
The first trial stopped before training because the training-text check failed. The Qwen3 chat template writes an empty `<think></think>` block into the answer when it renders a whole conversation, but not into the prompt the model is scored with. Training on the template's full text would have taught the model to start every answer with that block, and the scorer would have marked all of those answers invalid. Cost: one short GPU run that stopped early.
**Lesson:** build the training text from the exact scoring prompt plus the answer, and check it against what the template produces. Do not trust that the two match.

## 2026-10-22: Lesson: Unsloth may load a different repo than you asked for
Asked for `unsloth/Qwen3-4B-Instruct-2507` in 4-bit, Unsloth loads the ready-made copy `...-unsloth-bnb-4bit` and ignores the `revision`. Only its log says so. So baseline v2's `.meta.json` records the commit of a repo whose files were not the ones used. Step 7 names the 4-bit repo and its commit directly and records `model.config._name_or_path`, the name of what was really loaded.
**Lesson:** record what was loaded, read back from the loaded model, not only what was asked for. Read the warnings in a run's log.

## 2026-10-22: Lesson: the validation set picked the epoch, and both signals agreed
Over 3 epochs the training loss fell from 0.54 to 0.10, but the validation loss was lowest after epoch 2 (0.29) and rose in epoch 3 (0.34): overfitting. Answering the validation set with each epoch's adapter, by a rule written down first, also chose epoch 2. The test set was used once, at the end.
**Lesson:** save a checkpoint each epoch and choose on the validation set. That way one training run gives several versions to pick from, at no extra training cost.

## 2026-10-22: Lesson: what training fixed, and what it did not
Training fixed what the data aimed at. The order weakness is mostly gone (swapped precision 0.79 became 0.95), invalid answers went from 70 to 0, and type accuracy went from about 0.75 to 0.96. Two weak spots are left. Without dates, a contradiction whose memory says "switched" reads like an update. And `unclear` sometimes dodges a real contradiction with an invented escape (8 of 12 `unclear` answers). The prompt lists reading rules 1 to 3 only, while the labels follow rules 4 and 5 too.
**Lesson:** when the labels follow rules, the prompt the model sees should state them as well.

## 2026-10-22: Mistake: the scripts trusted their own records too much
Codex found three ways the Step 7 scripts could quietly spoil saved results. A smoke run used the final file names, so a quick check could write over the final exam. Choosing again kept the upload record of the old choice. And choosing did not check that the validation file and settings were still the ones the training record named. None of it happened, but nothing stopped it. The fixes (M1-07b) check before any remote work starts: no output file may already exist, the upload must be the chosen adapter, and fingerprints and settings must match the record.
**Lesson:** a script that writes a record should check that record before it acts on it, and should refuse to replace a saved result. Do the checks before any paid or remote work, and test them with fake remote calls.

## 2026-10-22: Lesson: say "inferred" when you did not see it
The Step 7 summary said the baseline "used the same base files". The code and the file history make that very likely, but the baseline never recorded what it loaded, so nobody saw it. Codex asked for the claim to say what the evidence proves.
**Lesson:** for every "same as" claim in a result, ask whether a record shows it or it is worked out. If it is worked out, say so and point to the evidence.


## 2026-10-23: Lesson: attach an adapter the way the save function expects
Unsloth's GGUF save merges the adapter only when it is called on the adapter model itself. `judge.load` attaches the adapter with `peft.PeftModel.from_pretrained`, which is fine for answering, but a save call on that model reaches the base model underneath and would write the base without the adapter. Unsloth 2026.10.7 refuses that case, and its source says so. The export loads the adapter folder with `FastLanguageModel.from_pretrained`, as Unsloth asks, and stops unless the result is a PEFT model.
**Lesson:** read the pinned library's source for a step that changes the model (merge, save, export), not only the docs page. The docs page shows the one-line call; the source shows what it needs to be called on.

## 2026-10-23: Mistake: Modal wants local code added last
The first export try stopped on the Mac before reaching Modal: the shared image adds our `judge` code last, and the export added more build steps after it. Modal refuses that order. No GPU was used. `src/judge/modal_image.py` now offers `packages` (the image without our code), so a job can add its own steps and then the code.
**Lesson:** when one job needs a bigger image, build on the image before the local code is added, and add the code last.

## 2026-10-23: Lesson: a stuck download can be the transfer method, not the network
The first download of the GGUF to the Mac sat at 0 bytes for 12 minutes. Hugging Face stores new large files with its "Xet" system, and the `hf` command's Xet path hung here. A plain web request for the same file worked at once, so `HF_HUB_DISABLE_XET=1` was set and the download moved. The Mac's line then gave about 1 MB a second, so 2.5 GB took about 45 minutes.
**Lesson:** when a download shows no progress, test the same file with a plain request before blaming the network, and check the speed early to plan around it.

## 2026-10-23: Lesson: a QLoRA adapter is trained on the 4-bit copy, and the usual merge does not use it
The Mac file agreed with the trained adapter on 96.6% of validation labels, just under the 97% bar. The template was ruled out (raw and chat routes gave identical text on all 297 answers). A run of the merged model at full 16-bit precision, before any squeezing, already agreed only 96.97%: Unsloth's export adds the adapter into the original 16-bit weights, while training ran it on a 4-bit copy of them. The q4_k_m squeeze then flipped a few more (98.3% against the 16-bit merge). Scores against the right answers stayed about the same; the versions disagree on different close calls.
**Lesson:** to find where answers change, run each step on its own (template, merge, squeeze) on the validation set, and compare each with the one before. For a model trained with QLoRA, expect the merge itself to move close calls, and set an agreement bar knowing that.

## 2026-10-23: Lesson: a fidelity bar and a quality bar answer different questions
The Mac file missed the 97% agreement bar on validation (96.6%), yet on the test set it scored as well as the adapter or better (recall without dates 0.94 against 0.90), and the two agreed on 97.3%. Where they differed, the Mac file was right 10 times and the adapter 4. Agreement measures "is it the same model"; scores against the right answers measure "is it as good". Elias chose to ship on the second (D-032).
**Lesson:** when a copy of a model misses an agreement bar, also score it against the right answers on the validation set before deciding, and put both numbers in front of the person who decides.

## 2026-10-23: Lesson: Python rounds 0.965 down
Rescoring the saved answers for the model cards, `f"{193/200:.2f}"` printed 0.96, while the results files say 0.97. The float 0.965 is stored as a hair less than 0.965, so it rounds down. The summaries had been written by hand, half up. Four accuracy numbers would have disagreed between the cards and the results.
**Lesson:** round from the exact counts (`Decimal(top) / Decimal(bottom)`, half up) in one function, and use that function wherever a number is printed for people.

## 2026-10-23: Lesson: a check that passes on its first run proves little
The pre-publish check passed on the first full run. Tests that break a card on purpose (a wrong score, a stray number, a changed prompt line, a token in a deleted file) showed what it really catches, and one of them found a gap: the phone pattern missed a 3-2-3 digit layout.
**Lesson:** for every check, write a test that feeds it the problem and expects a failure, before trusting a PASS.

## 2026-10-23: Lesson: the same count written in many places goes stale
D-030 dropped 31 pairs (train 1,522, valid 99), but `data/m1/README.md` still said 1,550 and 102 on its first lines, and the glossary said 102. Readers of the published dataset would have seen two different counts.
**Lesson:** when a decision changes a count, search the repo for the old number and update every place that states it as current. Better: state it in one place and point to it.

## 2026-10-23: Lesson: run the code in a card exactly as written
The model card's Python example was copied out of the card and run against the real model on the Mac. It printed the recorded test answer, byte for byte. The pre-publish check now also compares the card's prompt with `src/judge/prompt.py` and its Modelfile with `ollama/Modelfile`, so the two cannot drift apart.
**Lesson:** code meant for strangers gets run once, as they would run it, before it is published.

## 2026-10-23: M1 "why it works" note (Claude in Cowork, SPEC Step 10)
- **Why a 4B model beat an untrained 8B on the hard version.** Untrained models read "two different statements" as "a clash", and they lean on the order the memories come in. Fine-tuning showed the 4B about 1,500 worked answers, with the order shuffled and dates sometimes removed. So it learned the actual rule (same moment versus a change over time) instead of a shortcut. Size helps general knowledge; examples teach the specific rule.
- **Why LoRA was enough.** The model already understands language. The job only needed a small steer: about 66 million trained numbers, 1.6% of the model. That is why training took 30 minutes on a cheap GPU.
- **Why learning only from the answer matters.** Masking the prompt makes the model spend all its learning on producing the right JSON, not on repeating the instructions.
- **Why the validation set picked epoch 2.** In epoch 3 the training loss kept falling but the validation loss rose. The model was starting to memorize its practice pairs instead of the rule, which is called overfitting.
- **Why the test set mattered most.** A model can only be as good as the exam that measures it. Three blind reviews of the data caught shortcut patterns and fake "unclear" pairs that would have taught bad habits, before any GPU money was spent.

## 2026-10-26: Session close-out lessons (Claude in Cowork, M1 session 2026-10-20 to 2026-10-23)
- **Where Codex lives.** On Elias's Mac, Codex runs inside the ChatGPT desktop app. There is no `codex` command-line tool. Codex review prompts say "Codex (ChatGPT desktop app), new session, in the folder `~/code/model-training`", and Codex writes its review to `reviews/`, where Claude in Cowork reads it from disk.
- **Talking to the Aeverum project.** Files go into Aeverum's git-ignored `~/code/Aeverum-Auditer/.handoffs/` and are read with their SHA-256. They are information only, and nothing from Aeverum's data or Sources is copied here. A copy of what we sent lives in `docs/handoffs/`.
- **Prompts get pasted fast.** Elias pasted M1-06 before a change to it was committed. Once a prompt is handed over, treat it as live, and fix any difference at the next check instead of asking him to paste again.
- **Blind agent reviews replaced Elias's spot-checks (D-028).** A second agent that answers before it sees the labels found real problems three times (shortcut patterns, fake `unclear` pairs twice). Keep it for every dataset.
- **Reviews only after a real change.** Codex reviewed the code-heavy steps (5 and 7) and found real gaps both times. Fixes that only add safeguards, covered by offline tests, were checked by Claude in Cowork, without a second Codex round.
- **A made-up bar can block a good result.** The 97% agreement bar in Step 8 was a guess. What decides whether to ship a file is how well the file itself scores (D-032).

## 2026-10-26: Mistake: said our base model has a thinking mode
While exploring the next phase, Claude in Cowork told Elias in chat that the judge could "think first" by turning on Qwen's thinking mode. Qwen3-4B-Instruct-2507 has none: its model card says "This model supports only non-thinking mode". This was caught before anything was built. M2 handles it with a `steps` field in the answer instead (D-040). Lesson: check the exact model's card before suggesting a feature of its family.

## 2026-10-26: Lesson: RL learns nothing from pairs the model always gets right
In GRPO, each answer is scored against the group's average. If every answer to a pair is right, every score is the same, and nothing changes. v1 already gets most M1 training pairs right, so M2 needs new hard training pairs, kept only where v1's answers disagree (SPEC Step 3).

## 2026-10-26: Lesson: a test set can wear out
When a model scores 0.94 to 0.98 on a test set with 100 contradictions, a real gain of a few pairs looks like luck. Before trying to improve a model, check that the test can still show the improvement. For M2, that meant a new hard test set (D-038).

## 2026-10-26: Lesson: a time word in a memory must fit the memory's dates
In the M2 hard test set, many updates say when the change happened ("Asel got married in Almaty in September"). Random dates would sometimes put memory a after that September, so the dated version would read as a contradiction, not an update. Claude Code caught this before running any check and gave each such update a date window: memory a before the named time, memory b after it. The same goes for contradictions that say "since January": if the two memories are a year apart, the two Januaries differ and both can be true, so that pair was set to the same day.
**Lesson:** when dates are drawn by a program, read every time word in the text ("in March", "last year", "since January") and make the dates agree with it. Check the dated and the undated reading of each pair.

## 2026-10-26: Lesson: the name wall catches ordinary capitalised words too
The checker's name wall rejects any word that matches a name in the M1 files, wherever it appears. In the M2 hard set it caught place and pet names ("Ho", "Rosario", "Paolo", "Thabo"), a mountain ("Ben" in Ben Nevis), and even "Cherry" at the start of a reason. Widening the wall to M1's training and validation files also showed that those two files share 7 name words, so the wider wall applies to M2 files only; M1 is frozen and published.
**Lesson:** before checking, list every capitalised word in the new texts and test it against the names to avoid, not only the people's names.

## 2026-10-26: Lesson: on the hard set, v1's misses are mostly `unclear` dodges
On the frozen hard test set, v1 (the adapter) missed 16 of 150 contradictions at its worst, and 14 of those 16 were `unclear` answers, not wrong labels. They cluster in two traps, paraphrase and far_apart_no_change, where the clash needs one step of everyday sense ("terrified of dogs" and "cuddles puppies"). The reasons invent a way out ("fear can be mild or strong"), the same habit M1 found in 8 of its 12 `unclear` answers. Its false alarms come from a swapped word read as a clash ("oud" and "piano") and the same thing said two ways ("Bheki" and "Bhekizizwe").
**Lesson:** a hard test set shows which habit to train away. Here it is the dodge, which the M2 reward already prices at -0.5 (D-040). Results: `results/m2/baseline/summary.md`.

## 2026-10-26: Lesson: two forms of the same model differ by a few pairs
v1 on the Mac (the GGUF) and v1 as the adapter on Modal are the same trained model, yet on the hard set their worst-case recall differs by 4 pairs (138 and 134 of 150). The M2 bar asks for 5 more pairs.
**Lesson:** compare v2 with v1 in the same form (adapter with adapter, or GGUF with GGUF), never across forms, or the merge-and-squeeze wobble can look like a gain or a loss.

## 2026-10-26: Lesson: when GPU jobs start together, wall time can be three times the work
Three Modal jobs started within a few minutes of each other. Each did 7 to 11 minutes of loading and answering, but the last one took 25.8 minutes from start to finish, because it waited for an L4. Modal bills while a container loads and answers (https://modal.com/pricing), so the bill sits between the two numbers.
**Lesson:** log both the time inside the function and the wall time, and give the cost as a range, rather than pricing wall time as if it were all GPU work.

## 2026-10-26: Mistake: three numbers in the first summary draft were off
The first draft of `results/m2/baseline/summary.md` said 9 `unclear` pairs where the scorer showed 8, gave the drop from the M1 test set as 0.03 to 0.04 when it was 0.01 to 0.05, and stated a precision floor unclearly. A line-by-line check against the scorer output caught all three before the file was saved.
**Lesson:** after writing a results summary, check every number in it against the printed scorer output, one by one, before saving.

## 2026-10-27: Lesson: separate writers reach for the same everyday ideas
Eleven helper agents wrote the M2 training pairs from one brief, each with its own cultures and topic focus. Every file passed the checker and the labels were right, yet the same ideas kept coming back across files: 8 "vegetarian orders meat" pairs, 7 "never owned a car, drives one", 8 "only child, has a sister", 4 "bakery night shift, works days". One pair was word for word the same as another writer's. The checker could not see this, because it compares each file only with itself and with the walls. Rereading caught 29 repeated ideas, and comparing all 1,133 pairs with each other found 22 more near copies, plus one pair with almost the same sentences as another but a different label.
**Lesson:** after splitting writing across helpers, compare every pair with every other pair across all files, and count repeated ideas, not only repeated words. Check your own rewrites against the whole pool too: one of Claude Code's first rewrites copied an idea from another file.

## 2026-10-27: Lesson: an `unclear` pair needs one act with one missing detail, and both answers ordinary
In round 1, a blind judge kept 5 of 66 `unclear` candidates. Its reasons showed why. "Whose is it?" pairs ("owns no boat" / "rows the boat on the lake") and "a visit or a move?" pairs ("lives in the city" / "spent the summer at the farm") nearly always hint at an answer: "the club's shed" reads as someone else's, "the piano in her front room" as hers, a stay with a purpose as a visit. A new writer was then told what the judge accepted: one memory reports a single act whose deciding detail is left out, and both values are ordinary (was it paid? which language? live or on a screen? homemade or bought?). A fresh judge kept 51 of 60, while still calling none of 40 ordinary pairs `unclear`.
**Lesson:** read the blind judge's reasons for rejecting `unclear` pairs, turn them into a written test, and write the next round to that test. Mix ordinary pairs into every blind round, so you can see the judge is not just saying `unclear` to everything.

## 2026-10-27: Mistake (repeated): AI-written `unclear` pairs drift toward dodging, even with a second agent's check
Claude Code had a second helper agent confirm every new `unclear` pair blind, yet Cowork's blind review called 32 of 33 decidable. A helper working from the same brief shares the writer's habits, so it is not an independent check. **Lesson:** do not write new `unclear` pairs with AI agents. Reuse ones already agreed by independent reviews, or have Elias write and check a few by hand. A check is only independent if the checker never saw the writing brief.

## 2026-10-27: Session close-out lessons (Claude in Cowork, session 2026-10-26 to 2026-10-27)
- **Committing to `main` while Claude Code works.** When Claude Code has a branch checked out in `~/code/model-training`, Claude in Cowork must not touch the working files. It commits control-file changes to `main` with git's "temporary index" (a scratch staging list): `.git/cowork-commit.sh` copies files from `.git/cowork-staging/` into a commit on `main` and pushes, without changing the checkout. Run it through osascript with `/bin/bash`, because editing it with `sed -i` drops its run permission.
- **Check the exact model before suggesting a feature** (see the 2026-10-26 thinking-mode mistake).
- **Test sets wear out.** Measure whether a test can still show a gain before trying to improve a model (D-038).
- **Blind reviews:** the brief lives in `reviews/blind-brief.md`. On the M2 hard test set, the reviewers agreed on 296 of 300 pairs, and the 4 disputes showed the traps work. On the training pairs, the review caught 32 bad `unclear` pairs that a helper agent working from the same brief had passed.
- **Aeverum coordination works through one handoff file and one reply.** The reply set D-043. Elias relays the file between the two Cowork projects.
- **Mistake (caught at once): the temporary-index commit ran while the checkout was on `main`.** Claude Code had not yet made its branch, so moving `main` left the real staging list holding the old files, which showed as staged reverts. It was fixed by restoring the four files from the new commit (`git checkout HEAD -- <files>`), before any other commit. `.git/cowork-commit.sh` now refuses to run when the checkout is on `main`.
- **Mistake: the continuation told Elias to paste Claude Code's report,** after he had said M2-04 was already pasted. A new session reads results from disk (the branch and its `STATE.md`). Never hand Elias a step that is already done, or one a tool can do (R-0008, R-0037).
