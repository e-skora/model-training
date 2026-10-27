# M2: Judge v2, improved with reinforcement learning

## What and why
M1's judge (v1, `m1-judge:v1`) learned by copying right answers. That is supervised fine-tuning (SFT). M2 trains the same judge further with reinforcement learning (RL), using GRPO:
- For each pair of memories, the model writes several answers (a "group").
- A small program, the reward function, gives each answer points.
- Training pushes the model toward the answers that scored above the group's average.

Two goals:
1. Learn RL hands-on (D-003, D-036).
2. Fix v1's weak spots for Aeverum without adding false alarms:
   - contradictions it misses when there are no dates;
   - "both can be true" excuses (t-182, t-144 in M1);
   - dodges with `unclear`.

## Why a new test set comes first
- The M1 test set has 100 contradictions, so one pair moves a score by 0.01.
- v1 already scores 0.94 to 0.98 on it.
- A real gain of a few pairs would look like luck.

So M2 writes a harder test set first, freezes it, then measures v1 on it. The M1 test set stays as a second check that nothing got worse.

## The hard test set (Step 1, D-038, D-039)
- **300 pairs:** 150 `contradiction`, 50 `update`, 50 `compatible_detail`, 50 `merely_related`. There are no `unclear` right answers, the same as the M1 test set.
- **A `hard` field on every pair** names its trap. The trap list and counts are in prompt M2-01. Scores are also reported per trap.
- **Every pair can be decided from its text alone**, so the no-dates version still has one right answer:
  - an update shows the change in its words;
  - a contradiction never depends on the dates.
- **Hard by design, not by filtering.** Pairs are written to fit a trap. They are never picked by which pairs v1 gets wrong (D-039).
- **Fresh people, kept apart from M1.** No name or near copy of any M1 file. A separate agent reviews the set blind (D-042); then it is frozen.

## Starting point (D-037)
- RL starts from v1: the epoch-2 adapter on `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` (D-031), and keeps training that same adapter.
- RL needs a model that already sometimes answers right; v1 does.
- The base model stays Qwen3-4B-Instruct-2507.

## Reward (Step 4, starting values, fixed before any training; D-040)

| What the answer did | Points |
|---|---|
| Right label | +1.0 |
| Right type as well (non-contradictions) | +0.25 |
| Said `contradiction` when it is not (false alarm) | -1.5 |
| Said `non_contradiction` on a real contradiction (miss) | -1.0 |
| `unclear` on a pair that can be decided | -0.5 |
| `unclear` when the right answer is `unclear` (training pairs only) | +1.0 |
| Not valid JSON in the exact format | -2.0 |

Why these values:
- A false alarm costs more than a miss, because a false alarm interrupts Elias.
- An honest `unclear` costs less than a wrong answer, but more than nothing, so dodging never pays.
- The `reason` sentence earns nothing, because no program can check it.

Values change only from validation results, and each change is logged in `DECISIONS.md`.

## Two answer styles (D-040)
- **Style A (main):** the answer stays exactly as in v1, so Aeverum can swap the model in with nothing else changing.
- **Style B (stretch):** a short `steps` field is added before the label, so the model reasons first. Whether to run it is decided after Style A's full run.
- Qwen3-4B-Instruct-2507 has no built-in thinking mode. Its model card says: "This model supports only non-thinking mode and does not generate `<think></think>` blocks" (https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). So any reasoning has to live in the answer itself, and Style B gets a new prompt fingerprint.

## Training data (Step 3)
- GRPO only learns from pairs where the model's tries disagree. If all of its answers to a pair are right, they score the same, and nothing changes.
- v1 gets most M1 training pairs right, so they teach RL almost nothing.

So Step 3:
1. Write about 1,000 new hard training pairs and 100 new validation pairs from the same trap list.
2. Keep them apart from both test sets, as in M1.
3. Have a separate agent review a sample blind.
4. Have v1 answer each training pair 8 times with some randomness (temperature 1). Keep the pairs where its answers disagree, plus a small share it always gets right, so it does not forget them.
5. Shuffle the order and remove dates for training examples as in M1 (D-026).

## Steps

| Step | What | Who |
|---|---|---|
| 1 | Write the hard test set; blind review; freeze | Claude Code (prompt M2-01), then Claude in Cowork |
| 2 | Score v1 (the GGUF on the Mac and the adapter on Modal) and the untrained Qwen3-8B on it: the M2 baseline | Claude Code |
| 3 | Write the hard training and validation pairs; blind review; find the pairs where v1's answers disagree | Claude Code, then Claude in Cowork |
| 4 | Reward function and its tests (no GPU) | Claude Code; Codex reviews |
| 5 | GRPO trial run, then full run from v1. Pick the checkpoint on validation by a rule written first | Claude Code; Codex reviews the training code |
| 6 | Score v2 once on the hard test set and on the M1 test set (as is, swapped, no dates) | Claude Code |
| 7 | GGUF for the Mac: `m2-judge:v2` in Ollama | Claude Code |
| 8 | Publish to Hugging Face; write the "what RL changed" note; draft the LinkedIn post | Claude Code, then Claude in Cowork, then Elias edits |

## Done means (D-041)
1. **v2 beats v1 on the frozen hard test set.** Its worst-case recall goes up by at least 0.03 (5 pairs). Worst-case means the lowest of the three versions. Its worst-case precision is no more than 0.01 lower.
2. **v2 holds on the M1 test set.** Its precision and recall are each no more than 0.02 below v1's, in all three versions.
3. **0 invalid answers** in every version.
4. **It runs on the Mac** in Ollama.
5. **Everything is public:**
   - the model, the hard test set and the new training pairs on Hugging Face;
   - the code on GitHub;
   - a "what RL changed" note in `LESSONS.md`;
   - a LinkedIn draft.

If v2 misses item 1, M2 still ends. v2 is not published as better, and the note says plainly what RL changed and what it did not. That is a real result too.

## Cost
- An L4 on Modal by default.
- About $2 to $8 for all of M2. This is Claude's estimate from Modal's L4 price ($0.000222 a second, https://modal.com/pricing) and Unsloth's guidance of at least 300 steps; it is not measured.
- It stays inside Modal's free $30 a month (D-014).
- The per-run stop stays: a prompt asks Elias before any single run over $1 (D-019).

## Data rules (the same as M1)
- Everything is made up, about fictional people. Never Mara Lindqvist, and never text from Aeverum.
- Everyday topics only.
- The M1 reading rules 1 to 5 hold (`milestones/M1-contradiction-judge/SPEC.md`).

## Fitting Aeverum
The M1 points still hold:
- order means nothing, and dates may be missing;
- the model can say `unclear`;
- the prompt and settings are fixed and fingerprinted;
- it runs in Ollama;
- precision matters most.

Aeverum's step-12 benchmark decides whether it uses v1 or v2. Nothing here reads Aeverum's benchmark or its held-back test pairs.
