---
pretty_name: Memory contradiction pairs
license: cc-by-4.0
language:
- en
task_categories:
- text-classification
tags:
- synthetic
- contradiction-detection
- memory
annotations_creators:
- machine-generated
source_datasets:
- original
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: valid.jsonl
  - split: test
    path: test.jsonl
---

# Memory contradiction pairs

Pairs of short memories about made-up people, each with a label that says whether the two memories contradict each other. It was made to train and test a small judge model for a personal memory app, which compares stored memories and flags clashes for a person to review.

Every person and every memory in it is made up. There is no real personal data.

Models trained on it: [eskora/m1-contradiction-judge-qwen3-4b-gguf](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-gguf) (the ready-to-run file) and [eskora/m1-contradiction-judge-qwen3-4b-lora](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-lora) (the training add-on). Code: [github.com/e-skora/model-training](https://github.com/e-skora/model-training).

## What a pair is

One line of a file is one pair: a person's name, two memories about that person (each with the date it was written), the right label, a type, and a one-sentence reason. Here is a line from the test split:

```json
{"id": "t-022", "person": "Baraka Otieno", "a": {"text": "Baraka's flat is in Kisumu.", "written_at": "2025-05-03"}, "b": {"text": "Baraka moved to Mombasa for a new job and lives there now.", "written_at": "2025-12-14"}, "label": "non_contradiction", "type": "update", "reason": "The later memory says Baraka moved to Mombasa, so it replaces the earlier home."}
```

| Field | What it holds |
|---|---|
| `id` | `tr-` for train, `v-` for validation, `t-` for test |
| `person` | The made-up person's full name |
| `a`, `b` | The two memories: `text`, and `written_at` (a date). Memory `a` is always the earlier one, or written the same day |
| `label` | `contradiction`, `non_contradiction` or `unclear` |
| `type` | For a `non_contradiction`: `update`, `compatible_detail` or `merely_related`. Otherwise `null` |
| `reason` | One short sentence that explains the label |

## Labels and types

| Label | Type | Meaning | Example |
|---|---|---|---|
| `contradiction` | `null` | The two memories cannot both be true | "Her favourite season is autumn" and "Her favourite season is spring" |
| `non_contradiction` | `update` | The later memory replaces the earlier one | "She lives in Oslo", then later "She moved to Bergen" |
| `non_contradiction` | `compatible_detail` | Both are true, and one adds detail to the same fact | "She likes tea" and "She drinks green tea in the morning" |
| `non_contradiction` | `merely_related` | A different fact on the same topic, with no clash | "She hikes" and "She owns hiking boots" |
| `unclear` | `null` | The text alone cannot settle it, because it depends on a fact neither memory gives | "Is vegan" and "takes honey in her tea" |

`unclear` appears only in the train and validation splits.

## The five reading rules

Every label follows these rules.

1. Two memories that state a current favourite, choice or fact as true at the same moment, and disagree, are a `contradiction`.
2. If the second memory was written later and describes a change ("later", "now", "moved", "switched"), it is an `update`, not a contradiction.
3. If both can be true together, it is not a contradiction.
4. Do not invent an escape. If the two memories state different current facts about the same thing, it is a `contradiction`, even if you can imagine a way out ("maybe she has two dogs"). `unclear` is only for pairs where a fact that is truly missing decides it, and neither reading is more natural.
5. Different days, times or places with no exclusive word ("only", "always", "every day", "never") can both be true, so they are not a contradiction: "works from home on Fridays" and "works from home on Mondays" is `merely_related`.

## Splits and counts

| Label | Type | Train | Validation | Test |
|---|---|---|---|---|
| contradiction | | 697 | 46 | 100 |
| non_contradiction | update | 255 | 17 | 34 |
| non_contradiction | compatible_detail | 253 | 16 | 33 |
| non_contradiction | merely_related | 270 | 18 | 33 |
| unclear | | 47 | 2 | 0 |
| **Total** | | **1,522** | **99** | **200** |

Every pair has its own person: 1,522, 99 and 200 different names. No first name or surname of a test-split person appears in the train or validation split.

| File | Pairs | SHA-256 |
|---|---|---|
| `train.jsonl` | 1,522 | `a41de541d2a25c564009e232b2d75ccb714c5a668dcd30ef73004fae778231b1` |
| `valid.jsonl` | 99 | `0476bc57e19460a10f1a3299f17e570b0d7926cb893d6ce5d57d49f356e042d3` |
| `test.jsonl` | 200 | `6e2a08323f445477f9600e371f1074ad3e2b41ef97d7072d6734ef481f6e574b` |

A SHA-256 is a fingerprint of a file's exact contents: change one letter and it changes. Check a download with `shasum -a 256 test.jsonl`.

To load it in Python:

```python
from datasets import load_dataset
pairs = load_dataset("eskora/memory-contradiction-pairs")
```

## How it was made

- **Who wrote it.** Claude, a language model made by Anthropic, wrote every person and every memory, working in Claude Code from a written brief. The topics are everyday ones: home, work, food, hobbies, travel, pets, routines, belongings, tastes and places. Names come from many cultures. Nothing was copied from real people, real memories or other datasets.
- **The test split came first.** It was written and checked before any training. A second Claude session read all 200 pairs blind (it wrote its own answers before seeing the labels). The first round found no wrong labels but found shortcut patterns, so 86 pairs were rewritten; the second round found no wrong labels and no wrong types. Then a person (the author) checked 20 pairs by hand, and the split was frozen.
- **The train and validation splits came next.** Several Claude sessions wrote them from one shared brief, and a checker script ran after every batch. A separate Claude session then judged samples blind in two rounds, including every `unclear` pair. It agreed on the contradiction and non-contradiction labels. It disagreed on many `unclear` pairs, so rules 4 and 5 were added and the pairs were fixed; the `unclear` pairs it still disputed after that were dropped, not relabelled.
- **Walls between the splits.** A checker script stops if a train or validation pair uses a test-split name, or shares more than 60% of its words with a test pair, or repeats a pair from the other split.
- **Hard cases on purpose.** Change words ("now", "moved", "switched", "since") appear in every label, not only in updates, and some updates use none. Apart from updates, about half the pairs of each label have both memories written on the same day. So a model cannot score well by spotting one word.

## The test split was never trained on

The test split was frozen before training, at the SHA-256 above, and never edited after. The model was trained on the train split only. The validation split was used only to choose which saved version of the model to keep. The test split was then used once per model, to report its score.

## Limits

- **Synthetic, and one writer.** All three splits were written by the same language model, in the same style. The test split is no exception: it was written by Claude in the same style as the training data. A model trained on the train split will likely score lower on memories written by real people.
- **English only, everyday topics only.** No medical, legal, money or other sensitive topics.
- **Two short memories at a time,** both about one named person.
- **Labels follow the five rules above.** Someone with other rules could label some pairs differently, above all the `unclear` ones.
- **`unclear` is rare:** 47 train pairs, 2 validation pairs, none in test.
- **Order and dates in the files.** Memory `a` is always the earlier one, and an update's memory `b` is always dated later. A model could learn that shortcut. The training code shows the memories in the other order in half the examples and leaves the dates out of a third.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): use it for anything, including commercial work, and credit "Memory contradiction pairs by Elias Skora".
