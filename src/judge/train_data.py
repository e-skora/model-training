"""Training examples for the judge, the check that only the answer is trained, and the loss curves.

This file imports no GPU libraries, so tests can run it with a fake
tokenizer. The GPU job (`scripts/train.py`) calls it with the real one.

Each example is the model's own chat template applied to the shared prompt
(`judge.prompt`) plus the right JSON answer (`judge.format`). The prompt part
is exactly the text the model sees when it is scored (`judge.generate`), and
the answer ends with the end-of-turn token, so the model learns to stop there.
"""

import hashlib
import json

from judge.format import answer_text, format_pairs
from judge.generate import TEMPLATE_BASE

# Labels set to this value are not trained on (the Hugging Face convention).
IGNORE = -100
# The Qwen3 template writes this empty thinking block into the last assistant turn of a
# whole conversation, but not into the scoring prompt. Qwen3-4B-Instruct-2507 never thinks
# out loud, so the model is trained to answer straight after the prompt, without it.
EMPTY_THINK = "<think>\n\n</think>\n\n"


def read_pairs(path):
    """The pairs in a JSONL file and the SHA-256 of the exact bytes they were read from.

    One read gives both, so a recorded fingerprint always belongs to the pairs
    that were used, even if the file changes later.
    """
    with open(path, "rb") as f:
        data = f.read()
    pairs = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
    return pairs, hashlib.sha256(data).hexdigest()


def load_pairs(path):
    return read_pairs(path)[0]


def prompt_text(tokenizer, messages, template_kwargs=None):
    """The scoring input for these messages, built exactly as `judge.generate.answer_items` builds it."""
    return tokenizer.apply_chat_template(messages, **{**TEMPLATE_BASE, **(template_kwargs or {})})


def build_rows(pairs, tokenizer, seed=None, template_kwargs=None):
    """One training row per pair: {"id", "swapped", "dates", "prompt", "answer", "text"}.

    `text` is the prompt, then the answer, then the end-of-turn token. It is
    checked against the template's own rendering of the whole conversation,
    so it is the model's own format, not a copy of it. The one difference
    allowed is EMPTY_THINK before the answer, which `text` leaves out.
    """
    examples = format_pairs(pairs) if seed is None else format_pairs(pairs, seed=seed)
    end = tokenizer.eos_token
    rows = []
    for example, pair in zip(examples, pairs):
        prompt = prompt_text(tokenizer, example["messages"][:-1], template_kwargs)
        answer = answer_text(pair)
        assert example["messages"][-1]["content"] == answer
        full = tokenizer.apply_chat_template(
            example["messages"], **{**TEMPLATE_BASE, **(template_kwargs or {}), "add_generation_prompt": False}
        )
        text = prompt + answer + end
        # The template may close the turn with a newline after the end token; we stop at the token.
        allowed = {text, text + "\n", prompt + EMPTY_THINK + answer + end, prompt + EMPTY_THINK + answer + end + "\n"}
        if full not in allowed:
            raise ValueError(f"{example['id']}: the chat template's full conversation is not prompt + answer + "
                             f"{end!r}, so training would not match scoring")
        rows.append({"id": example["id"], "swapped": example["swapped"], "dates": example["dates"],
                     "prompt": prompt, "answer": answer, "text": text})
    return rows


def mask_problem(input_ids, labels, prompt_ids, answer_ids):
    """Say why one tokenized row does not train on exactly the answer, or return None.

    Right means: the row is the prompt's tokens then the answer's tokens
    (the answer ends with the end-of-turn token); every prompt label is
    IGNORE; and every answer label is the answer token itself.
    """
    n = len(prompt_ids)
    if len(labels) != len(input_ids):
        return "labels and input_ids differ in length"
    if list(input_ids[:n]) != list(prompt_ids):
        return "the row does not start with the prompt's tokens"
    if list(input_ids[n:]) != list(answer_ids):
        return "the tokens after the prompt are not the answer's tokens"
    if any(label != IGNORE for label in labels[:n]):
        return "some prompt tokens are trained on"
    if list(labels[n:]) != list(answer_ids):
        return "some answer tokens are not trained on"
    return None


def check_masks(rows, dataset, tokenizer):
    """Check every tokenized row of a training or validation set. Returns a short summary.

    `dataset` is what the trainer will use, in the same order as `rows`, each
    with "input_ids" and "labels". Stops with ValueError at the first problem,
    so a bad mask never reaches training.
    """
    if len(dataset) != len(rows):
        raise ValueError(f"{len(rows)} rows built but {len(dataset)} in the trainer's dataset (some were dropped)")
    trained = 0
    longest = 0
    for row, item in zip(rows, dataset):
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(row["answer"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        problem = mask_problem(item["input_ids"], item["labels"], prompt_ids, answer_ids)
        if problem:
            raise ValueError(f"{row['id']}: {problem}")
        trained += len(answer_ids)
        longest = max(longest, len(item["input_ids"]))
    return {"rows": len(rows), "trained_tokens": trained, "longest_row_tokens": longest}


def plain(value):
    """Turn a result into plain JSON values, so the Mac can read it without PyTorch installed.

    A number held in a PyTorch or NumPy object becomes a Python number (its
    `.item()`); anything else unknown becomes text.
    """
    return json.loads(json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def loss_curves(log_history):
    """Split the trainer's log into the training and validation loss curves.

    Numbers are read with float(), because Unsloth's log may hold them as text.
    """
    def point(entry, key):
        return {"step": int(entry["step"]), "epoch": round(float(entry["epoch"]), 3), "loss": round(float(entry[key]), 4)}

    train = [point(e, "loss") for e in log_history if "loss" in e]
    valid = [point(e, "eval_loss") for e in log_history if "eval_loss" in e]
    return {"train": train, "valid": valid}


def per_epoch(curves, epochs):
    """A few numbers per epoch: mean training loss in that epoch, and the validation loss at its end."""
    summary = []
    for epoch in range(1, epochs + 1):
        in_epoch = [p["loss"] for p in curves["train"] if epoch - 1 < p["epoch"] <= epoch]
        at_end = [p["loss"] for p in curves["valid"] if abs(p["epoch"] - epoch) < 0.02]
        summary.append({
            "epoch": epoch,
            "train_loss_mean": round(sum(in_epoch) / len(in_epoch), 4) if in_epoch else None,
            "train_loss_last": in_epoch[-1] if in_epoch else None,
            "valid_loss_end": at_end[-1] if at_end else None,
        })
    return summary


CHOICE_RULE = (
    "fewest invalid answers in total; then the highest worst-case precision across the three versions "
    "(a false alarm interrupts Elias, so precision matters most); then the highest worst-case recall; "
    "then the highest mean accuracy; then the earliest (least trained) version"
)


def choose(candidates):
    """Pick one trained version from its validation scores, by CHOICE_RULE. Returns the chosen name.

    `candidates` maps a name (such as "epoch-2") to {version: metrics} for the
    three versions of the validation set. The rule was fixed before any
    result was seen.
    """
    def key(item):
        index, (name, by_version) = item
        metrics = list(by_version.values())
        return (
            sum(m["invalid"] for m in metrics),
            -min(m["precision"] or 0 for m in metrics),
            -min(m["recall"] or 0 for m in metrics),
            -sum(m["accuracy"] or 0 for m in metrics) / len(metrics),
            index,
        )

    return min(enumerate(candidates.items()), key=key)[1][0]
