# M1: Memory contradiction judge

## What the model does
It reads two memories about the same person, each with the date it was written, and answers in JSON:

```json
{"label": "contradiction", "type": null, "reason": "Both say what her favourite season is, and they name different seasons."}
```

- `label` is `contradiction` or `non_contradiction`.
- `type` is only filled for `non_contradiction`:
  - `update`: the later memory replaces the earlier one ("She moved to Bergen" after "She lives in Oslo").
  - `compatible_detail`: both can be true together ("She likes tea" and "She drinks green tea in the morning").
  - `merely_related`: same topic, no clash ("She hikes" and "She owns hiking boots").
- `reason` is one short sentence.

## Reading rules
1. Two memories that state a current favourite, choice or fact as true at the same moment, and disagree, are a `contradiction`.
2. If the second memory was written later and describes a change ("later", "now", "moved", "switched"), it is an `update`, not a contradiction.
3. If both can be true together, it is not a contradiction.

## Why Aeverum needs it
Aeverum's night pass compares memories two at a time and flags contradictions for Elias to review. It runs a local model on the Mac. Aeverum's bar, measured on its own benchmark at its step 12: when the judge says "contradiction" it is right at least 85% of the time (precision 0.85), and it catches at least 80% of real contradictions (recall 0.80).

## Done means
1. The fine-tuned model reaches precision 0.85 and recall 0.80 on our own frozen test set, and beats the base model.
2. It runs on Elias's Mac (Ollama or MLX).
3. The model and its training data are published on Hugging Face with a model card. The code is on GitHub.
4. A LinkedIn post is drafted.

## Data rules
- All memories are made up, about many different fictional people. Never Mara Lindqvist, never text from Aeverum.
- Mix of labels: about half `contradiction`, and the rest split across the three non-contradiction types. The hard cases matter most: updates and near-misses that look like contradictions but are not.
- Test set: 200 pairs, written first, checked, then frozen.
- Training set: about 1,500 pairs, written separately, never sharing a pair with the test set.

## Steps
| Step | What | Who |
|---|---|---|
| 1 | Create the repo and control files | Claude (Cowork). Done 2026-10-21 |
| 2 | Connect `~/code/model-training` to the Cowork project; paste the project instructions into the project settings; set Modal budget to $30 at modal.com/settings/usage | Elias |
| 3 | Install tools and prove a GPU job runs on Modal | Claude Code (prompt M1-03) |
| 4 | Write the test set; Elias spot-checks 20 pairs; freeze it | Claude Code, then Elias |
| 5 | Score 2 or 3 base models on the test set (the baseline); pick one | Claude Code; Codex reviews the scoring code |
| 6 | Write the training set; Elias spot-checks 20 pairs | Claude Code, then Elias |
| 7 | Fine-tune on Modal; score it on the test set | Claude Code; Codex reviews the training code |
| 8 | Save the model in a format the Mac can run; test it locally | Claude Code |
| 9 | Publish model and dataset to Hugging Face; write the model card | Claude Code |
| 10 | Draft the LinkedIn post; write the "why it works" note | Claude (Cowork), then Elias edits |
