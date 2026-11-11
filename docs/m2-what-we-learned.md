# What I learned from training a small judge model with reinforcement learning

**In short:** I trained a small model further with reinforcement learning (RL) and set a pass mark before scoring it. It missed the pass mark by one test pair: 138 of 150 where 139 was needed. It still got better in most ways, and the miss taught me thirteen things that anyone building tools with language models can use. Each one became a fix in my second attempt.

Everything here links to the file in this repo that shows it. All the data is made up, about fictional people.

## 1. Why I ran this

I experiment with fine-tuning different types of models, and this was my step from plain fine-tuning into reinforcement learning. When the result came in one pair short, the most important thing to me was to show what we learned, publish something useful that shows why the experiment was valuable, and make the second attempt use those lessons. This page is that.

*Made by [Elias Skora](https://www.linkedin.com/in/eliassskora/), who builds systems with language models and experiments with fine-tuning different types of models. Written with Claude, a language model made by Anthropic, from the project's records and my own words ([DECISIONS.md](../DECISIONS.md), D-061 and D-063).*

## 2. What we tried, in five lines

- **v1** is a small judge (Qwen3-4B, fine-tuned with LoRA): it reads two memories about a person and says whether they contradict each other, with a type and a one-line reason ([M1 results](../results/m1/trained-v1/summary.md)).
- **M2** trained v1 further with RL, using **GRPO**: the model writes 8 answers to the same pair, a small program scores each one, and training pushes it toward the answers that scored above the group's average ([M2 spec](../milestones/M2-rl-judge/SPEC.md)).
- It practised on **harder pairs**, written around 11 kinds of traps, such as a clash said in different words or two memories that only look like a clash ([the hard data](../data/m2/README.md)).
- A **new hard test set** of 300 pairs was written, reviewed blind and locked before any training.
- **The pass mark was set before any score**, and written as code before the first test answer ([the rule](../scripts/compare_v2.py), [D-060](../DECISIONS.md)).

## 3. What happened

**The pass mark:** on the hard test set, v2 had to catch at least 0.03 more of the 150 real contradictions than v1 in its weakest version, without losing more than 0.01 of precision. It also had to hold its scores on the older M1 test set and give no broken answers. Each test set has three versions: the pairs **as is**, **swapped** (newer memory first) and **no dates**. The **worst case** is the lowest of the three.

| Check | v1 | Needed | v2 | Result |
|---|---|---|---|---|
| Hard set, worst-case recall | 134 of 150 | 139 of 150 | 138 of 150 | **missed by one pair** |
| Hard set, worst-case precision | 134 of 142 | 0.934 | 138 of 143 | passed |
| M1 set, every version within 0.02 of v1 | | | all six numbers at or above v1 | passed |
| Broken answers, every version of both sets | 0 | 0 | 0 | passed |

So v2 is **not** called better than v1, and v1 stays the recommended model ([full result](../results/m2/v2/summary.md)).

**Every score.** **Precision** is how often "contradiction" was right when the model said it. **Recall** is how many real contradictions it caught. "Caught / said" is the two counts behind them.

| Model | Test set | Version | Caught / said | Precision | Recall | Unclear answers | Invalid answers |
|---|---|---|---|---|---|---|---|
| v1 | hard | as is | 138 / 144 | 0.958 | 0.920 | 12 | 0 |
| v1 | hard | swapped | 136 / 142 | 0.958 | 0.907 | 15 | 0 |
| v1 | hard | no dates | 134 / 142 | 0.944 | 0.893 | 15 | 0 |
| v2 | hard | as is | 141 / 145 | 0.972 | 0.940 | 6 | 0 |
| v2 | hard | swapped | 138 / 143 | 0.965 | 0.920 | 7 | 0 |
| v2 | hard | no dates | 143 / 148 | 0.966 | 0.953 | 5 | 0 |
| v1 | M1 | as is | 95 / 96 | 0.990 | 0.950 | 3 | 0 |
| v1 | M1 | swapped | 96 / 101 | 0.950 | 0.960 | 6 | 0 |
| v1 | M1 | no dates | 90 / 92 | 0.978 | 0.900 | 3 | 0 |
| v2 | M1 | as is | 96 / 97 | 0.990 | 0.960 | 2 | 0 |
| v2 | M1 | swapped | 98 / 103 | 0.951 | 0.980 | 1 | 0 |
| v2 | M1 | no dates | 92 / 94 | 0.979 | 0.920 | 1 | 0 |

The hard set has 150 real contradictions in 300 pairs; the M1 set has 100 in 200.

**Pair by pair.** On the hard set's swapped version, v2 caught 4 contradictions that v1 missed, and missed 2 that v1 caught. Without dates it was 9 against 0. (Counted from the saved answers in [results/m2/v2/hard/](../results/m2/v2/hard/) and [results/m2/baseline/trained-v1/](../results/m2/baseline/trained-v1/).)

**Per trap, on the hard set.** Right labels out of each trap's pairs, as is / swapped / no dates.

| Trap (right answer) | Pairs | v1 | v2 |
|---|---|---|---|
| explain_away (contradiction) | 40 | 38 / 38 / 35 | 38 / 39 / 39 |
| paraphrase (contradiction) | 30 | 25 / 25 / 25 | 26 / 26 / 27 |
| small_difference (contradiction) | 30 | 30 / 30 / 30 | 30 / 30 / 30 |
| change_word_contradiction (contradiction) | 25 | 24 / 22 / 23 | 24 / 23 / 23 |
| far_apart_no_change (contradiction) | 25 | 21 / 21 / 21 | 23 / 20 / 24 |
| indirect_update (update) | 25 | 25 / 24 / 23 | 25 / 24 / 24 |
| update_with_overlap (update) | 25 | 25 / 25 / 25 | 25 / 25 / 25 |
| near_identical (compatible_detail) | 20 | 20 / 20 / 20 | 20 / 20 / 20 |
| looks_like_clash (compatible_detail) | 30 | 26 / 28 / 26 | 26 / 28 / 26 |
| near_identical (merely_related) | 25 | 22 / 21 / 22 | 25 / 24 / 25 |
| rule5_bait (merely_related) | 25 | 25 / 23 / 25 | 25 / 24 / 25 |

What each trap means: [data/m2/README.md](../data/m2/README.md).

## 4. What RL changed, and what it did not

**It changed:**
- **It caught more contradictions in every version of both test sets.** The biggest gain was without dates on the hard set: 134 to 143 of 150.
- **It raised fewer false alarms on the hard set:** 4, 5 and 5 against v1's 6, 6 and 8. On the M1 set the false alarms stayed the same.
- **It dodged less.** On the hard set it answered `unclear` 18 times in all, against v1's 42: less than half as often. That was v1's main weakness: 14 of its 16 worst-case misses were `unclear` answers ([baseline](../results/m2/baseline/summary.md)).
- **It kept v1's exact answer format,** with 0 invalid answers, so it could be swapped in with nothing else changing.

**It did not change:**
- **`looks_like_clash` did not move at all.** These are pairs that look like a clash but are not, such as a nickname and a full name. 4 of 30 are still wrong as is and without dates.
- **Far-apart pairs got slightly worse when swapped,** 21 to 20 of 25. That one version decided the miss.
- **The last part of training taught little.** GRPO learns only when the 8 answers to a pair score differently. The share of practice pairs whose 8 answers all scored the same rose from 0.45 in the first 40 steps to about 0.8 (`frac_reward_zero_std`). Checkpoints 250, 300 and 314 scored exactly the same on validation ([training summary](../results/m2/grpo/full-run-summary.md)).

## 5. What we learned, and what we changed because of it

This is the core. Every lesson below got a fix, and each fix has a named place in attempt 2 where it is enforced. The full list, with the reasons and the lessons that matter only to this repo, is in [docs/m2-lessons-and-fixes.md](m2-lessons-and-fixes.md).

| | Lesson | What happened | The fix | Where attempt 2 applies it |
|---|---|---|---|---|
| A1 | **Set the bar and the choice rule before any score.** (Keep) | The pass mark, the checkpoint rule and the test rule were all written as code before any result. That is why "missed by one pair" is a clean result and not an argument. | Keep it: the pass rule is code, committed before the first test answer, and tested on made-up answers. | Steps 6 and 8 |
| A2 | **A test set too small to show the gain you want.** | The bar was 5 pairs. The same model in two forms (a laptop file and the training add-on) already differs by 4 pairs, so a real gain of 4 could not be told from noise. | First decide the smallest gain worth seeing, then size the test so that gain is clearly bigger than the noise. Also report the pair-by-pair count. | Test set 2 has 300 contradictions in 600 pairs, so the same bar is 9 pairs (Step 1); paired count reported (Step 8) |
| A5 | **A validation set too small to pick a checkpoint.** | On 97 validation pairs one false alarm moves precision by about 0.02, so the choice rule needed loose limits, and three checkpoints tied exactly. | A bigger validation set: 200 pairs, 100 of them contradictions, kept apart from the test set. | Steps 1 and 7 |
| B1 | **RL learns nothing from pairs the model always gets right.** (Keep) | If all 8 answers score the same, there is nothing to push toward. | The practice pool keeps only pairs where the starting model's 8 answers disagree. | Step 4 |
| B2 | **The practice pool got too easy part way through.** | The pool was picked once, from v1, before training. As the model improved, more of it became "always right", and the last 60 or so steps taught almost nothing. | Train in two rounds, and rebuild the pool between them from the pairs the new model still gets wrong sometimes. Stop a round early if the "all the same" share stays above 0.7 for 40 steps. | Steps 4 and 5 |
| B4 | **Some traps did not improve.** | `looks_like_clash` did not move, and far-apart pairs got worse when swapped. The pool had few of them, each shown in one fixed order. | More practice pairs for the three weakest traps, shown in both orders. Per-trap results are reported only, so they do not steer the choice. | Steps 3 and 4 |
| C1 | **Many AI writers repeat the same ideas.** | 11 writing helpers, one brief: 8 "vegetarian orders meat" pairs, 7 "never owned a car" pairs and more, caught only by rereading everything. | Plan every pair's situation (person, topic, what changes) before writing, and give each writer its own slice. A checker compares every pair with every other pair across all files. | Steps 1 and 3 |
| C2 | **AI-written "can't tell" pairs drift toward dodging, and an AI check from the same brief is not independent.** | A helper passed all 33 new `unclear` pairs; the blind review rejected 32 of them. | A small human check, below. | Step 3 |
| C3 | **Blind reviews with control pairs catch real problems.** (Keep) | The reviewer answers before seeing the labels, mixed with pairs whose answer is known, so you can tell it is not just agreeing. On the hard test set two blind reviewers matched 296 of 300 labels. | Keep it for the test set, the validation set and a sample of the training pairs. The reviewer never sees the writing brief. | Steps 1 and 3 |
| D1 | **A review never ends until you say what a check must withstand.** | A code fingerprint took 5 review rounds, because each round found one more deliberate trick to fool it. | Every safety check states up front what it must catch (honest mistakes) and what is a known limit (tricks written on purpose to fool it). | Step 5's review prompt |
| D2 | **Each fix added code that the next review had to check.** | The training script also took 5 review rounds; most new findings were in recovery code that earlier fixes had added. | Reuse the accepted script with the smallest changes, make the smallest fix for each finding, then review only the fixes. | Step 5 |
| E1 | **A protection we counted on in a library never ran on our path.** | A sampling safeguard in Unsloth only runs for a different training method. Our sampling was right for another reason, found only by reading the installed code. | For any setting or patch you rely on, read the installed source of the pinned version, and record what the run really used. | Step 5 |
| F1 | **Early cost estimates were wrong.** | The slow first step and counting the checkpoint saves twice put the first estimate too high. | Estimate from the steady step rate plus one-off start-up, and compare every estimate with the real bill afterwards. | Step 5's trial, every run log line |

**The human check (C2).** A "can't tell" (`unclear`) pair is a judgment call about what an ordinary reader can decide, and that is exactly where AI writing and AI checking failed together. So in attempt 2, the AI writes 45 candidates to a written test, a blind reviewer gives its view on each, and I keep or drop each one on a one-page sheet in about 15 minutes. No `unclear` pair is used without my keep; every other kind of pair stays AI-written with cross-pair checks and blind review.

## 6. What it cost

Every M2 GPU run, from [runs/RUNLOG.md](../runs/RUNLOG.md). All ran on NVIDIA L4 GPUs rented from [Modal](https://modal.com). "Billed" is Modal's own bill; the five early runs have only estimates.

| Run | GPU minutes | Cost |
|---|---|---|
| Test runs of the scoring, 3 models | 1.6 and 2.4 | about $0.05 to $0.19 (estimated) |
| v1 and two untrained models answer the hard test set | 7.2, 8.9 and 11.0 | about $0.37 to $0.74 (estimated) |
| Two small test runs of pool sampling (the first ran out of memory) | | $0.05 billed |
| Building the practice pool: v1 answers 1,018 pairs 8 times each | 51.9 | $0.80 billed |
| Reading the installed library code (no GPU) | | under $0.01 billed |
| Training trial, 20 steps | 10.5 | $0.18 billed |
| Full training, 314 steps | 65.8 | $0.97 billed |
| Choosing the checkpoint: 8 candidates on validation | 37.6 | $0.57 billed |
| Upload to Hugging Face | | $0.01 billed |
| v2 answers both test sets once, plus a small test run | 14.3 | $0.21 billed |

**Total: about $3.21 to $3.72.** $2.79 of it is billed; the rest is the five estimated runs. The plan before M2 guessed $2 to $8.

**Estimates against the bills:**

| Run | Estimate | Billed |
|---|---|---|
| Building the practice pool | $0.73 | $0.80 |
| Full training | $1.02 | $0.97 |
| Choosing the checkpoint | $0.71 | $0.57 |

## 7. Attempt 2

- **Start from v2,** and still measure against v1, the recommended model.
- **A bigger, fresh hard test set:** 600 pairs with 300 contradictions, so the same bar is 9 pairs, about twice the noise between two forms of one model. Plus a 200-pair validation set.
- **Two rounds of training,** with the practice pool rebuilt from the new model's answers between them, and an early stop when the pool stops teaching.
- **A human check on every `unclear` pair,** and more practice on the traps that did not improve.
- **The same honest bar,** fixed in code before any score. If it misses again, that is written up the same way.

The plan, each step tagged with the lessons it applies: [milestones/M3-rl-judge-attempt-2/SPEC.md](../milestones/M3-rl-judge-attempt-2/SPEC.md).

## 8. My takeaway

We missed by one pair, and I am publishing the miss. The bar was fixed before any score, so a near miss is still a real result, not a story told afterwards. Every lesson and every thing that did not go well now has a fix, and the second attempt is built on them.

## Links

- The model from this attempt (v2, missed the bar): [eskora/m2-contradiction-judge-qwen3-4b-grpo-lora](https://huggingface.co/eskora/m2-contradiction-judge-qwen3-4b-grpo-lora)
- The recommended model (v1): [eskora/m1-contradiction-judge-qwen3-4b-gguf](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-gguf) (runs on a laptop) and [eskora/m1-contradiction-judge-qwen3-4b-lora](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-lora)
- The hard data, as a small benchmark: [eskora/memory-contradiction-pairs-hard](https://huggingface.co/datasets/eskora/memory-contradiction-pairs-hard)
- Parts you can reuse: the reward function [src/judge/reward.py](../src/judge/reward.py), the checkpoint rule (`choose_checkpoint` in [scripts/train_grpo.py](../scripts/train_grpo.py)), the pass rule [scripts/compare_v2.py](../scripts/compare_v2.py), and the run checks in [scripts/train_grpo.py](../scripts/train_grpo.py)
