"""Tests for judge.generate with fake model and tokenizer objects. No GPU, no download."""

import hashlib

from judge.generate import GENERATION, answer_items


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeIds:
    def __init__(self, rows, length):
        self.rows = rows
        self.shape = (rows, length)


class FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self):
        self.template_calls = []
        self.call_kwargs = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        return "PROMPT:" + messages[-1]["content"]

    def __call__(self, texts, **kwargs):
        self.call_kwargs.append(kwargs)
        return FakeInputs(input_ids=FakeIds(len(texts), 3))

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return "".join(tokens)


class FakeModel:
    def __init__(self):
        self.generate_kwargs = []

    def generate(self, **kwargs):
        self.generate_kwargs.append(kwargs)
        rows = kwargs["input_ids"].rows
        # Three prompt tokens, then the answer.
        return [["p", "p", "p", "{", "}"] for _ in range(rows)]


def items(n):
    return [{"id": f"p{i}", "messages": [{"role": "user", "content": f"pair {i}"}]} for i in range(n)]


def test_settings_are_passed_through_and_saved():
    model, tokenizer = FakeModel(), FakeTokenizer()
    results, settings = answer_items(model, tokenizer, items(5), {"enable_thinking": False}, batch_size=2, device="cpu")

    assert [r["id"] for r in results] == ["p0", "p1", "p2", "p3", "p4"]
    assert all(r["raw"] == "{}" for r in results)  # prompt tokens cut off

    # Greedy decoding, explicitly: one beam, no sampling, in every batch.
    assert len(model.generate_kwargs) == 3
    for kwargs in model.generate_kwargs:
        assert kwargs["num_beams"] == 1
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == GENERATION["max_new_tokens"]
        assert kwargs["pad_token_id"] == tokenizer.pad_token_id

    expected_template = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert all(call == expected_template for call in tokenizer.template_calls)
    assert all(call["add_special_tokens"] is False for call in tokenizer.call_kwargs)
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<eos>"

    assert settings["generation"] == GENERATION
    assert settings["generation"]["num_beams"] == 1
    assert settings["chat_template_options"] == expected_template
    assert settings["first_prompt_sha256"] == hashlib.sha256(b"PROMPT:pair 0").hexdigest()
    assert settings["first_prompt"] is None  # only saved in full for runs of one batch


def test_small_run_saves_the_first_prompt_in_full():
    _, settings = answer_items(FakeModel(), FakeTokenizer(), items(2), {}, batch_size=16, device="cpu")
    assert settings["first_prompt"] == "PROMPT:pair 0"
    assert settings["chat_template_options"] == {"tokenize": False, "add_generation_prompt": True}
