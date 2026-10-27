---
license: apache-2.0
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
language:
- en
datasets:
- eskora/memory-contradiction-pairs
tags:
- lora
- qlora
- unsloth
- qwen3
- sft
- contradiction-detection
---

# Memory contradiction judge (Qwen3-4B, LoRA adapter)

The training add-on (a LoRA adapter) that turns Qwen3-4B-Instruct-2507 into a judge: it reads two memories about a person and says whether they contradict each other, with a one-sentence reason. Made by [Elias Skora](https://www.linkedin.com/in/eliassskora/), an AI systems builder who experiments with fine-tuning different types of models, as the contradiction checker for a personal memory app he is building.

**Most people want the ready-to-run file instead:** [eskora/m1-contradiction-judge-qwen3-4b-gguf](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-gguf), one 2.5 GB file for Ollama on a laptop. This repo is for running the model on a GPU with Python, or for training it further. The data is in [eskora/memory-contradiction-pairs](https://huggingface.co/datasets/eskora/memory-contradiction-pairs) and the code in [github.com/e-skora/model-training](https://github.com/e-skora/model-training).

## What it does

You give it a person's name and two memories, each with the date it was written if you know it. It answers with one line of JSON.

Input:

```
Person: Baraka Otieno
Memory A (written 2025-05-03): Baraka's flat is in Kisumu.
Memory B (written 2025-12-14): Baraka moved to Mombasa for a new job and lives there now.
```

Its answer (this exact text, from the test run on a GPU):

```json
{"label": "non_contradiction", "type": "update", "reason": "The newer memory says Baraka moved to Mombasa and lives there now."}
```

| `label` | `type` | Meaning |
|---|---|---|
| `contradiction` | `null` | The two memories cannot both be true |
| `non_contradiction` | `update` | The later memory replaces the earlier one |
| `non_contradiction` | `compatible_detail` | Both are true, and one adds detail |
| `non_contradiction` | `merely_related` | Same topic, no clash |
| `unclear` | `null` | The text alone cannot settle it. Treat it as "a person should look" |

The order of the two memories does not matter, and dates may be left out.

## How to run it with Ollama

Ollama runs the GGUF version of this adapter, not the adapter itself. In short (full steps and a Python example on the [GGUF model page](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-gguf)):

1. Download `m1-contradiction-judge-qwen3-4b-Q4_K_M.gguf` from [eskora/m1-contradiction-judge-qwen3-4b-gguf](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-gguf) at commit `f9a1ae6ec40ba6b9e1c544b13a317fdf5a452360`, and check that its SHA-256 is `b6d930ce83c06512d12da2549b219edda8506e57404007acd877de29308172a9`.
2. Make the Ollama model from the Modelfile on that page: `ollama create m1-judge:v1 -f Modelfile`.
3. Send the exact prompt below to `http://127.0.0.1:11434/api/chat`, with no `options` or `format`.

## How to run it on a GPU with Python

The adapter sits on top of Unsloth's 4-bit copy of the base model, `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at commit `7744afa8566e264af1a92a806d8d9aae00cc7c78`. The scores below were measured by loading it this way, with Unsloth 2026.10.7 and PEFT on one NVIDIA L4 (this is a shortened copy of `src/judge/load.py` and `src/judge/generate.py` in the GitHub repo):

```python
import unsloth  # import first, as Unsloth asks
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    revision="7744afa8566e264af1a92a806d8d9aae00cc7c78",
    max_seq_length=2048, load_in_4bit=True)
model = PeftModel.from_pretrained(model, "eskora/m1-contradiction-judge-qwen3-4b-lora",
                                  revision="1d9ebbf66e1a2abcad9932932bb7d810a38c59b9", is_trainable=False)
FastLanguageModel.for_inference(model)

messages = [{"role": "system", "content": SYSTEM_MESSAGE},  # the exact prompt below
            {"role": "user", "content": "Person: Baraka Otieno\n"
                                        "Memory A (written 2025-05-03): Baraka's flat is in Kisumu.\n"
                                        "Memory B (written 2025-12-14): Baraka moved to Mombasa for a new job and lives there now."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to("cuda")
output = model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

`do_sample=False` and `num_beams=1` mean it always picks the most likely word, so the same pair always gets the same answer.

## The exact prompt

The model was trained and scored with this system message. Then the user message is `Person: <name>`, then `Memory A (written <YYYY-MM-DD>): <text>` and `Memory B (written <YYYY-MM-DD>): <text>`, one per line. When a date is unknown, the line is `Memory A: <text>`.

```
You judge memories. You read two memories about the same person. Each memory may show the date it was written, but dates can be missing. The order of the two memories means nothing: either one may be the older. You decide whether the two memories contradict each other.

Reading rules:
1. Two memories that state a current favourite, choice or fact as true at the same moment, and disagree, are a contradiction.
2. If one memory was written later and describes a change ("later", "now", "moved", "switched"), it is an update, not a contradiction. When dates are missing, judge an update from the wording alone.
3. If both can be true together, it is not a contradiction.

Labels:
- "contradiction": the two memories cannot both be true. The type is null.
- "non_contradiction": the two memories can both be true. The type is one of:
  - "update": the later memory replaces the earlier one ("She moved to Bergen" after "She lives in Oslo").
  - "compatible_detail": both are true together, and one adds detail to the same fact ("She likes tea" and "She drinks green tea in the morning").
  - "merely_related": a different fact on the same topic, with no clash ("She hikes" and "She owns hiking boots").
- "unclear": the text alone cannot settle it, because whether they clash depends on facts the memories do not give, such as the scope or the time. The type is null. Do not use it when rule 1 already decides.

Answer with JSON only, with no other text, in exactly this format:
{"label": "contradiction" or "non_contradiction" or "unclear", "type": null or "update" or "compatible_detail" or "merely_related", "reason": "one short sentence"}
```

Its SHA-256 (the system message and the line templates together, as computed by `prompt_sha256()` in `src/judge/prompt.py`, and by the Python code on the GGUF page) is `83d14d39a2115eb1f4d2d461581673090ec25d0a6f05d444cb2da8b756a2865c`. Change one character and you have a different judge, with untested scores.

## How it was trained

- **Base model:** [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), about 4 billion parameters, Apache 2.0 license, in Unsloth's 4-bit copy named above.
- **Method:** QLoRA with [Unsloth](https://github.com/unslothai/unsloth) (version 2026.10.7) on one NVIDIA L4 GPU rented from [Modal](https://modal.com). QLoRA trains this small add-on on top of a 4-bit copy of the model, so it fits on a cheap GPU. Settings: rank 32, alpha 32, on all 7 projection layers; learning rate 2e-4; 16 examples per step; 3 epochs (passes through the data); seed 3407. It learned from the JSON answer only, not from the prompt.
- **Data:** the 1,522 train pairs of [eskora/memory-contradiction-pairs](https://huggingface.co/datasets/eskora/memory-contradiction-pairs). 761 of them were shown with the newer memory first and 507 without dates, so the model would not depend on order or dates.
- **Which epoch:** a copy was saved after each of the 3 epochs. The 99 validation pairs alone chose epoch 2, by a rule written down before any result: fewest invalid answers, then the best worst-case precision, then the best worst-case recall. Epoch 2 also had the lowest validation loss (0.29). This repo holds epoch 2.
- **Time and cost:** the training run took 29.6 minutes of training, and about $0.45 of GPU time. All GPU runs for this model, including trial runs, the untrained comparisons and scoring, cost about $1.66.
- **Files:** `adapter_model.safetensors` and `adapter_config.json` (the adapter), the tokenizer and chat template it was trained with, and `train.meta.json` (every setting, the loss curves and the validation scores).

## Results

Scored once on the 200 test pairs (100 contradictions), which the model never trained on. The test set was frozen before training at SHA-256 `6e2a08323f445477f9600e371f1074ad3e2b41ef97d7072d6734ef481f6e574b`. Each model answered every pair in three test versions:

- **as is:** the pairs as written, older memory first, with dates;
- **swapped:** the same pairs with the newer memory first;
- **no dates:** the same pairs with both dates removed.

**Precision** is how often "contradiction" was right when the model said it. **Recall** is how many of the 100 real contradictions it caught. **Accuracy** is how often the label was right. An **invalid answer** is one that is not valid JSON or uses a label that does not exist. All models used the same prompt and always picked the most likely word. The untrained models ran in 4-bit on a GPU; Qwen3-8B with its thinking turned off.

| Model | Test version | Precision | Recall | Accuracy | Unclear answers | Invalid answers |
|---|---|---|---|---|---|---|
| **This adapter (on a GPU)** | as is | **0.99** | **0.95** | **0.97** | 3 | **0** |
| **This adapter (on a GPU)** | swapped | **0.95** | **0.96** | **0.94** | 6 | **0** |
| **This adapter (on a GPU)** | no dates | **0.98** | **0.90** | **0.94** | 3 | **0** |
| GGUF version (Ollama on a Mac) | as is | 0.98 | 0.96 | 0.97 | 2 | 0 |
| GGUF version (Ollama on a Mac) | swapped | 0.94 | 0.96 | 0.95 | 2 | 0 |
| GGUF version (Ollama on a Mac) | no dates | 0.98 | 0.94 | 0.96 | 1 | 0 |
| Qwen3-4B-Instruct-2507, untrained | as is | 0.93 | 0.84 | 0.83 | 0 | 22 |
| Qwen3-4B-Instruct-2507, untrained | swapped | 0.79 | 0.88 | 0.75 | 0 | 20 |
| Qwen3-4B-Instruct-2507, untrained | no dates | 0.96 | 0.85 | 0.79 | 0 | 28 |
| Qwen3-8B, untrained | as is | 0.95 | 0.96 | 0.96 | 0 | 0 |
| Qwen3-8B, untrained | swapped | 0.73 | 0.93 | 0.80 | 0 | 1 |
| Qwen3-8B, untrained | no dates | 0.94 | 0.96 | 0.95 | 0 | 1 |

What it shows:

- **Training fixed the order problem.** With the newer memory first, the untrained models called most updates contradictions (precision 0.79 and 0.73). This adapter stays at 0.95.
- **It always answers in the right format:** 0 invalid answers in 600, against 70 for the untrained Qwen3-4B.
- **Against the untrained Qwen3-8B,** a model about twice its size: higher precision in every test version; recall one pair lower as is (0.95 against 0.96), higher swapped (0.96 against 0.93) and lower without dates (0.90 against 0.96).
- The memory app's target is precision 0.85 and recall 0.80. This adapter is above it in all three test versions.
- The GGUF version is made from this adapter. Because it adds the adapter to the 16-bit weights instead of the 4-bit copy it was trained on, the two disagree on a few close calls: the same label on 584 of 600 answers.

The test set is small: one pair is 0.01 of recall, so gaps of one or two hundredths are one or two pairs.

## Limits

- **Made-up data, one writer.** The training and test data were written by Claude, a language model, about made-up people. The test set was written by the same model in the same style as the training data, so these scores are likely higher than the model would get on memories written by real people. It has not been tested on real memories.
- **English only, everyday topics only:** home, work, food, hobbies, travel, pets, routines, belongings, tastes and places.
- **It sometimes says `unclear` on a real contradiction,** by imagining a way both could be true (for example, reading "has never owned a pet" and "had a rabbit as a child" as unclear). Every `unclear` answer on the test set was wrong, since the test set has no `unclear` pairs: 12 of the 600 answers here. The prompt lists only reading rules 1 to 3, while the training labels also follow rule 4, "do not invent an escape".
- **It sometimes explains a clash away** as compatible detail, for example "three pots of geraniums" and "four pots" as "four is three plus one".
- **Lower recall without dates.** Without dates, a contradiction where one memory says "switched" or "since" can look like an update. Recall drops from 0.95 to 0.90 here.
- **Two short memories about one named person at a time.** It was not tested on long texts, more than two memories, or memories about several people.
- **Not for high-stakes decisions.** Use it to flag pairs for a person to check, not to decide anything about anyone on its own, and not for medical, legal, money or safety matters.

## Links

- Ready-to-run file (GGUF, for Ollama): [eskora/m1-contradiction-judge-qwen3-4b-gguf](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-gguf)
- Dataset: [eskora/memory-contradiction-pairs](https://huggingface.co/datasets/eskora/memory-contradiction-pairs)
- Code, raw answers and full results: [github.com/e-skora/model-training](https://github.com/e-skora/model-training)

License: Apache 2.0, the same as the base model.
