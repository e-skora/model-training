"""Tests for judge.train_data and the GPU-free parts of scripts/train.py. No GPU, no network.

The fake tokenizer writes Qwen's chat format (ChatML) and turns each
character into one token, with the special markers as single tokens.
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

from judge.format import answer_text, format_pairs
from judge.generate import TEMPLATE_BASE
from judge.train_data import (
    CHOICE_RULE,
    EMPTY_THINK,
    IGNORE,
    build_rows,
    check_masks,
    choose,
    loss_curves,
    mask_problem,
    per_epoch,
    plain,
)

REPO = Path(__file__).resolve().parents[1]
SPECIAL = ["<|im_start|>", "<|im_end|>"]


class ChatMLTokenizer:
    eos_token = "<|im_end|>"

    def __init__(self, trailing_newline=True):
        self.trailing_newline = trailing_newline

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        end = "\n" if self.trailing_newline else ""
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>{end}" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            for n, mark in enumerate(SPECIAL):
                if text.startswith(mark, i):
                    ids.append(-(n + 1))
                    i += len(mark)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": self.encode(text)}


def pair(i, label="contradiction", kind=None):
    return {
        "id": f"x-{i:03d}",
        "person": "Test Person",
        "a": {"text": f"Test Person likes colour {i}.", "written_at": "2025-01-01"},
        "b": {"text": f"Test Person likes colour {i + 1}.", "written_at": "2025-01-01"},
        "label": label,
        "type": kind,
        "reason": "They name different favourite colours.",
    }


PAIRS = [pair(i) for i in range(12)]


def masked(row, tokenizer):
    """What train_on_responses_only should produce for one row: prompt labels IGNORE, answer labels kept."""
    ids = tokenizer.encode(row["text"])
    n = len(tokenizer.encode(row["prompt"]))
    return {"input_ids": ids, "labels": [IGNORE] * n + ids[n:]}


def test_text_is_scoring_prompt_then_answer_then_end_token():
    tokenizer = ChatMLTokenizer()
    rows = build_rows(PAIRS, tokenizer)
    examples = format_pairs(PAIRS)
    for row, example, p in zip(rows, examples, PAIRS):
        # The prompt is exactly what judge.generate.answer_items gives the model when scoring.
        assert row["prompt"] == tokenizer.apply_chat_template(example["messages"][:-1], **TEMPLATE_BASE)
        assert row["prompt"].endswith("<|im_start|>assistant\n")
        assert row["answer"] == answer_text(p)
        assert row["text"] == row["prompt"] + row["answer"] + "<|im_end|>"
    # The format_pairs seed decides order and dates, the same as judge.format.
    assert [(r["swapped"], r["dates"]) for r in rows] == [(e["swapped"], e["dates"]) for e in examples]


def test_template_without_trailing_newline_also_matches():
    rows = build_rows(PAIRS, ChatMLTokenizer(trailing_newline=False))
    assert all(r["text"].endswith("<|im_end|>") for r in rows)


def test_template_that_differs_from_scoring_stops():
    class OddTokenizer(ChatMLTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            text = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return text if add_generation_prompt else text.replace("assistant\n", "assistant\n<think></think>")

    with pytest.raises(ValueError, match="would not match scoring"):
        build_rows(PAIRS, OddTokenizer())


def test_qwen3_empty_think_block_is_left_out_of_the_answer():
    # The real Qwen3-4B-Instruct-2507 template does this (checked on Modal, 2026-10-22).
    class Qwen3Tokenizer(ChatMLTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            text = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            if add_generation_prompt:
                return text
            head, _, last = text.rpartition("<|im_start|>assistant\n")
            return head + "<|im_start|>assistant\n" + EMPTY_THINK + last

    rows = build_rows(PAIRS, Qwen3Tokenizer())
    assert all("<think>" not in r["text"] for r in rows)
    assert all(r["text"] == r["prompt"] + r["answer"] + "<|im_end|>" for r in rows)


def test_right_mask_passes_and_counts_answer_tokens():
    tokenizer = ChatMLTokenizer()
    rows = build_rows(PAIRS, tokenizer)
    summary = check_masks(rows, [masked(r, tokenizer) for r in rows], tokenizer)
    assert summary["rows"] == len(PAIRS)
    assert summary["trained_tokens"] == sum(len(tokenizer.encode(r["answer"])) + 1 for r in rows)
    assert summary["longest_row_tokens"] == max(len(tokenizer.encode(r["text"])) for r in rows)


def test_a_trained_prompt_token_stops():
    tokenizer = ChatMLTokenizer()
    rows = build_rows(PAIRS, tokenizer)
    data = [masked(r, tokenizer) for r in rows]
    data[3]["labels"][5] = data[3]["input_ids"][5]
    with pytest.raises(ValueError, match=f"{rows[3]['id']}: some prompt tokens are trained on"):
        check_masks(rows, data, tokenizer)


def test_an_untrained_answer_token_stops():
    tokenizer = ChatMLTokenizer()
    rows = build_rows(PAIRS, tokenizer)
    data = [masked(r, tokenizer) for r in rows]
    data[0]["labels"][-1] = IGNORE  # the end token is not learned, so the model would not stop
    with pytest.raises(ValueError, match="some answer tokens are not trained on"):
        check_masks(rows, data, tokenizer)


def test_missing_end_token_or_dropped_row_stops():
    tokenizer = ChatMLTokenizer()
    rows = build_rows(PAIRS, tokenizer)
    data = [masked(r, tokenizer) for r in rows]
    data[1] = {"input_ids": data[1]["input_ids"][:-1], "labels": data[1]["labels"][:-1]}
    with pytest.raises(ValueError, match="not the answer's tokens"):
        check_masks(rows, data, tokenizer)
    with pytest.raises(ValueError, match="some were dropped"):
        check_masks(rows, data[:-1], tokenizer)


def test_mask_problem_cases():
    prompt, answer = [1, 2, 3], [7, 8, 9]
    assert mask_problem(prompt + answer, [IGNORE] * 3 + answer, prompt, answer) is None
    assert mask_problem(prompt + answer, [IGNORE] * 6, prompt, answer) == "some answer tokens are not trained on"
    assert mask_problem([0, 2, 3] + answer, [IGNORE] * 3 + answer, prompt, answer).startswith("the row does not start")
    assert mask_problem(prompt + answer, [IGNORE] * 3, prompt, answer).startswith("labels and input_ids")


def test_loss_curves_and_per_epoch():
    log = [
        {"loss": 1.0, "step": 2, "epoch": 0.5},
        {"eval_loss": 0.9, "step": 2, "epoch": 0.5},
        {"loss": 0.6, "step": 4, "epoch": 1.0},
        {"eval_loss": 0.5, "step": 4, "epoch": 1.0},
        {"loss": 0.3, "step": 6, "epoch": 1.5},
        {"loss": 0.2, "step": 8, "epoch": 2.0},
        {"eval_loss": 0.55, "step": 8, "epoch": 2.0},
        {"train_runtime": 10.0, "train_loss": 0.5, "step": 8, "epoch": 2.0},
    ]
    log[0] = {"loss": "1.0", "step": 2, "epoch": "0.5"}  # Unsloth may log numbers as text
    curves = loss_curves(log)
    assert [p["loss"] for p in curves["train"]] == [1.0, 0.6, 0.3, 0.2]
    assert [p["loss"] for p in curves["valid"]] == [0.9, 0.5, 0.55]
    assert per_epoch(curves, 2) == [
        {"epoch": 1, "train_loss_mean": 0.8, "train_loss_last": 0.6, "valid_loss_end": 0.5},
        {"epoch": 2, "train_loss_mean": 0.25, "train_loss_last": 0.2, "valid_loss_end": 0.55},
    ]


def test_plain_turns_tensor_numbers_into_python_numbers():
    class Scalar:  # stands in for a 0-d torch tensor
        def item(self):
            return 0.25

    assert plain({"loss": Scalar(), "steps": [1, Scalar()], "other": object}) == {
        "loss": 0.25, "steps": [1, 0.25], "other": str(object)}


def metrics(precision, recall, invalid=0, accuracy=0.9):
    return {"precision": precision, "recall": recall, "invalid": invalid, "accuracy": accuracy}


def three(precision, recall, invalid=0, accuracy=0.9):
    return {v: metrics(precision, recall, invalid, accuracy) for v in ("as_is", "swapped", "no_dates")}


def test_choose_follows_the_written_rule():
    # Fewest invalid answers wins over better precision.
    assert choose({"epoch-1": three(0.99, 0.99, 1), "epoch-2": three(0.90, 0.90)}) == "epoch-2"
    # Then the worst-case precision across versions.
    weak = three(0.99, 0.9)
    weak["swapped"] = metrics(0.80, 0.95)
    assert choose({"epoch-1": weak, "epoch-2": three(0.95, 0.9)}) == "epoch-2"
    # Then worst-case recall, then mean accuracy, then the earliest.
    assert choose({"epoch-1": three(0.95, 0.85), "epoch-2": three(0.95, 0.9)}) == "epoch-2"
    assert choose({"epoch-1": three(0.95, 0.9, 0, 0.8), "epoch-2": three(0.95, 0.9, 0, 0.85)}) == "epoch-2"
    assert choose({"epoch-1": three(0.95, 0.9), "epoch-2": three(0.95, 0.9)}) == "epoch-1"
    assert "precision" in CHOICE_RULE


def load_train_script():
    spec = importlib.util.spec_from_file_location("train_script", REPO / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # builds Modal objects lazily; no network, no GPU
    return module


def test_train_settings():
    script = load_train_script()
    baseline = json.loads((REPO / "results/m1/baseline-v2/qwen3-4b-instruct-2507.meta.json").read_text())
    # Names only: the trained base is the 4-bit copy Unsloth maps the baseline's repo name to. This does
    # not prove the baseline loaded the same weight files; that evidence is in results/m1/trained-v1/base-files.md.
    assert script.BASE_REPO == baseline["model"] + "-unsloth-bnb-4bit"
    assert len(script.BASE_REVISION) == 40
    assert script.HF_REPO == "eskora/m1-contradiction-judge-qwen3-4b-lora"
    assert script.TRAIN["per_device_train_batch_size"] * script.TRAIN["gradient_accumulation_steps"] == 16
    assert 1 <= script.TRAIN["num_train_epochs"] <= 3
    assert script.LORA["random_state"] == script.SEED
    assert script.planned_steps(1522) == 3 * math.ceil(1522 / 16)
    assert script.planned_steps(1522, max_steps=20) == 20
    assert script.warmup_steps(288) == 14 and script.warmup_steps(20) == 1
    # The script reads only the training and validation sets, never the test set.
    assert "test.jsonl" not in (REPO / "scripts" / "train.py").read_text()


def test_valid_items_cover_three_versions():
    script = load_train_script()
    items = script.valid_items(PAIRS[:4])
    assert [i["id"] for i in items][:5] == ["as_is/x-000", "as_is/x-001", "as_is/x-002", "as_is/x-003", "swapped/x-000"]
    assert len(items) == 12
    assert "written" not in items[-1]["messages"][1]["content"]
