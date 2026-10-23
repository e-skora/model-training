---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
base_model_relation: finetune
library_name: gguf
pipeline_tag: text-generation
language:
- en
datasets:
- eskora/memory-contradiction-pairs
tags:
- gguf
- ollama
- qwen3
- unsloth
- contradiction-detection
---

# Memory contradiction judge (Qwen3-4B, GGUF)

A small model that reads two memories about a person and says whether they contradict each other, with a one-sentence reason. It is Qwen3-4B-Instruct-2507, fine-tuned for this one job, in a single 2.5 GB file that runs on a laptop with Ollama.

It was built as the contradiction checker for a personal memory app: the app compares stored memories and shows the clashes to a person to review. Made by Elias Skora while learning to fine-tune open language models.

This repo holds the ready-to-run file. The training add-on is in [eskora/m1-contradiction-judge-qwen3-4b-lora](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-lora), the data in [eskora/memory-contradiction-pairs](https://huggingface.co/datasets/eskora/memory-contradiction-pairs), and the code in [github.com/e-skora/model-training](https://github.com/e-skora/model-training).

## What it does

You give it a person's name and two memories, each with the date it was written if you know it. It answers with one line of JSON.

Input:

```
Person: Baraka Otieno
Memory A (written 2025-05-03): Baraka's flat is in Kisumu.
Memory B (written 2025-12-14): Baraka moved to Mombasa for a new job and lives there now.
```

Its answer (this exact text, from the test run on a Mac):

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

The order of the two memories does not matter, and dates may be left out. If an answer is not valid JSON, treat it as "a person should look" too.

## How to run it with Ollama

You need [Ollama](https://ollama.com) (version 0.24.0 was used) and the `hf` command from the [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/guides/cli) package.

1. Download the file (2.5 GB):
   ```bash
   hf download eskora/m1-contradiction-judge-qwen3-4b-gguf m1-contradiction-judge-qwen3-4b-Q4_K_M.gguf --revision f9a1ae6ec40ba6b9e1c544b13a317fdf5a452360 --local-dir ~/models/m1-judge
   ```
2. Check its fingerprint. The line must start with `b6d930ce83c06512d12da2549b219edda8506e57404007acd877de29308172a9`:
   ```bash
   shasum -a 256 ~/models/m1-judge/m1-contradiction-judge-qwen3-4b-Q4_K_M.gguf
   ```
3. Save this as a file named `Modelfile`. It names the file, writes out the chat template the model was trained with, and makes the answers repeatable (temperature 0 means it always picks the most likely word):
   ```
   FROM ~/models/m1-judge/m1-contradiction-judge-qwen3-4b-Q4_K_M.gguf
   TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}
   {{ .Content }}<|im_end|>
   {{ end }}<|im_start|>assistant
   """
   PARAMETER temperature 0
   PARAMETER num_ctx 2048
   PARAMETER num_predict 256
   PARAMETER repeat_penalty 1
   PARAMETER stop "<|im_end|>"
   PARAMETER stop "<|endoftext|>"
   ```
4. Make the Ollama model:
   ```bash
   ollama create m1-judge:v1 -f Modelfile
   ```
5. Ask it about a pair with the Python code in the next section. Ollama answers on `http://127.0.0.1:11434` while it is open. Send the prompt exactly as below, and do not add `options` or `format` to the request: the scores were measured without them.

On an Apple M4 Pro it took 0.91 seconds per pair, one pair at a time.

## The exact prompt

The model was trained and scored with this prompt, and it works best with exactly this text. The SHA-256 of the prompt (the system message and the three line templates, as computed by `prompt_sha256()` below) is:

`83d14d39a2115eb1f4d2d461581673090ec25d0a6f05d444cb2da8b756a2865c`

Change one character and you have a different judge, with untested scores. This code holds the prompt, checks its SHA-256 and asks the model about one pair. It needs only Python's standard library.

```python
import hashlib
import json
import urllib.request

SYSTEM_MESSAGE = """You judge memories. You read two memories about the same person. Each memory may show the date it was written, but dates can be missing. The order of the two memories means nothing: either one may be the older. You decide whether the two memories contradict each other.

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
{"label": "contradiction" or "non_contradiction" or "unclear", "type": null or "update" or "compatible_detail" or "merely_related", "reason": "one short sentence"}"""

MEMORY_LINE = "Memory {name} (written {written_at}): {text}"
MEMORY_LINE_NO_DATE = "Memory {name}: {text}"
USER_TEMPLATE = "Person: {person}\n{memory_a}\n{memory_b}"


def prompt_sha256():
    template = {"system": SYSTEM_MESSAGE, "user": USER_TEMPLATE,
                "memory": MEMORY_LINE, "memory_no_date": MEMORY_LINE_NO_DATE}
    return hashlib.sha256(json.dumps(template, sort_keys=True).encode("utf-8")).hexdigest()


def memory_line(name, text, written_at=None):
    if written_at:
        return MEMORY_LINE.format(name=name, written_at=written_at, text=text)
    return MEMORY_LINE_NO_DATE.format(name=name, text=text)


def judge(person, memory_a, memory_b, date_a=None, date_b=None):
    """Ask the model about one pair. Dates are 'YYYY-MM-DD' or None."""
    user = USER_TEMPLATE.format(person=person, memory_a=memory_line("A", memory_a, date_a),
                                memory_b=memory_line("B", memory_b, date_b))
    body = {"model": "m1-judge:v1", "stream": False, "messages": [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]}
    request = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())["message"]["content"]


assert prompt_sha256() == "83d14d39a2115eb1f4d2d461581673090ec25d0a6f05d444cb2da8b756a2865c"
print(judge("Baraka Otieno", "Baraka's flat is in Kisumu.",
            "Baraka moved to Mombasa for a new job and lives there now.", "2025-05-03", "2025-12-14"))
```

The answer is a string of JSON; read it with `json.loads`. When a date is unknown, pass `None` and the line becomes `Memory A: <text>`.

## How it was trained

- **Base model:** [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), about 4 billion parameters, Apache 2.0 license. Training used Unsloth's 4-bit copy of it, `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at commit `7744afa8566e264af1a92a806d8d9aae00cc7c78`.
- **Method:** QLoRA with [Unsloth](https://github.com/unslothai/unsloth) (version 2026.10.7) on one NVIDIA L4 GPU rented from [Modal](https://modal.com). QLoRA trains a small add-on (a LoRA adapter) on top of a 4-bit copy of the model, so it fits on a cheap GPU. Settings: rank 32, alpha 32, on all 7 projection layers; learning rate 2e-4; 16 examples per step; 3 epochs (passes through the data); seed 3407. It learned from the JSON answer only, not from the prompt.
- **Data:** the 1,522 train pairs of [eskora/memory-contradiction-pairs](https://huggingface.co/datasets/eskora/memory-contradiction-pairs). 761 of them were shown with the newer memory first and 507 without dates, so the model would not depend on order or dates.
- **Which epoch:** a copy was saved after each of the 3 epochs. The 99 validation pairs alone chose epoch 2, by a rule written down before any result: fewest invalid answers, then the best worst-case precision, then the best worst-case recall. Epoch 2 also had the lowest validation loss (0.29).
- **Time and cost:** the training run took 29.6 minutes of training, and about $0.45 of GPU time. All GPU runs for this model, including trial runs, the untrained comparisons and scoring, cost about $1.66.
- **This file:** Unsloth's `save_pretrained_gguf` added the epoch-2 adapter into the original 16-bit weights (`unsloth/Qwen3-4B-Instruct-2507`) and squeezed the result to 4 bits per weight (q4_k_m) with llama.cpp `b11798-mix-f8a7950`. File: `m1-contradiction-judge-qwen3-4b-Q4_K_M.gguf`, 2,497,280,320 bytes, SHA-256 `b6d930ce83c06512d12da2549b219edda8506e57404007acd877de29308172a9`.

Because training ran on a 4-bit copy and this file was made from the 16-bit weights, the file and the adapter disagree on a few close calls. On the test set they gave the same label on 584 of 600 answers.

## Results

Scored once on the 200 test pairs (100 contradictions), which the model never trained on. The test set was frozen before training at SHA-256 `6e2a08323f445477f9600e371f1074ad3e2b41ef97d7072d6734ef481f6e574b`. Each model answered every pair in three test versions:

- **as is:** the pairs as written, older memory first, with dates;
- **swapped:** the same pairs with the newer memory first;
- **no dates:** the same pairs with both dates removed.

**Precision** is how often "contradiction" was right when the model said it. **Recall** is how many of the 100 real contradictions it caught. **Accuracy** is how often the label was right. An **invalid answer** is one that is not valid JSON or uses a label that does not exist. All models used the same prompt and always picked the most likely word. The untrained models ran in 4-bit on a GPU; Qwen3-8B with its thinking turned off.

| Model | Test version | Precision | Recall | Accuracy | Unclear answers | Invalid answers |
|---|---|---|---|---|---|---|
| **This file (GGUF, Ollama on a Mac)** | as is | **0.98** | **0.96** | **0.97** | 2 | **0** |
| **This file (GGUF, Ollama on a Mac)** | swapped | **0.94** | **0.96** | **0.95** | 2 | **0** |
| **This file (GGUF, Ollama on a Mac)** | no dates | **0.98** | **0.94** | **0.96** | 1 | **0** |
| Trained adapter (on a GPU) | as is | 0.99 | 0.95 | 0.97 | 3 | 0 |
| Trained adapter (on a GPU) | swapped | 0.95 | 0.96 | 0.94 | 6 | 0 |
| Trained adapter (on a GPU) | no dates | 0.98 | 0.90 | 0.94 | 3 | 0 |
| Qwen3-4B-Instruct-2507, untrained | as is | 0.93 | 0.84 | 0.83 | 0 | 22 |
| Qwen3-4B-Instruct-2507, untrained | swapped | 0.79 | 0.88 | 0.75 | 0 | 20 |
| Qwen3-4B-Instruct-2507, untrained | no dates | 0.96 | 0.85 | 0.79 | 0 | 28 |
| Qwen3-8B, untrained | as is | 0.95 | 0.96 | 0.96 | 0 | 0 |
| Qwen3-8B, untrained | swapped | 0.73 | 0.93 | 0.80 | 0 | 1 |
| Qwen3-8B, untrained | no dates | 0.94 | 0.96 | 0.95 | 0 | 1 |

What it shows:

- **Training fixed the order problem.** With the newer memory first, the untrained models called most updates contradictions (precision 0.79 and 0.73). This file stays at 0.94.
- **It always answers in the right format:** 0 invalid answers in 600, against 70 for the untrained Qwen3-4B.
- **Against the untrained Qwen3-8B,** a model about twice its size: higher precision in every test version; recall the same as is, higher swapped (0.96 against 0.93) and lower without dates (0.94 against 0.96).
- The memory app's target is precision 0.85 and recall 0.80. This file is above it in all three test versions.

The test set is small: one pair is 0.01 of recall, so gaps of one or two hundredths are one or two pairs.

## Limits

- **Made-up data, one writer.** The training and test data were written by Claude, a language model, about made-up people. The test set was written by the same model in the same style as the training data, so these scores are likely higher than the model would get on memories written by real people. It has not been tested on real memories.
- **English only, everyday topics only:** home, work, food, hobbies, travel, pets, routines, belongings, tastes and places.
- **It sometimes says `unclear` on a real contradiction,** by imagining a way both could be true (for example, reading "has never owned a pet" and "had a rabbit as a child" as unclear). Every `unclear` answer on the test set was wrong, since the test set has no `unclear` pairs: 5 of the 600 answers here.
- **It sometimes explains a clash away** as compatible detail, for example two different counts of the same thing.
- **Lower recall without dates.** Without dates, a contradiction where one memory says "switched" or "since" can look like an update. Recall drops from 0.96 to 0.94 here.
- **Two short memories about one named person at a time.** It was not tested on long texts, more than two memories, or memories about several people.
- **Not for high-stakes decisions.** Use it to flag pairs for a person to check, not to decide anything about anyone on its own, and not for medical, legal, money or safety matters.

## Links

- Dataset: [eskora/memory-contradiction-pairs](https://huggingface.co/datasets/eskora/memory-contradiction-pairs)
- Training add-on (LoRA adapter): [eskora/m1-contradiction-judge-qwen3-4b-lora](https://huggingface.co/eskora/m1-contradiction-judge-qwen3-4b-lora)
- Code, raw answers and full results: [github.com/e-skora/model-training](https://github.com/e-skora/model-training)

License: Apache 2.0, the same as the base model.
