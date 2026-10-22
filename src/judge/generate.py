"""Ask a loaded model to answer pairs, and record the settings used.

This file imports no GPU libraries, so tests can run it with fake model and
tokenizer objects. The GPU job (`scripts/run_baseline.py`) loads the real
model and calls `answer_items` inside `torch.no_grad()`.
"""

import hashlib

# Greedy decoding: always the most likely next word. num_beams=1 says so
# explicitly instead of trusting the downloaded model's default settings.
GENERATION = {
    "max_new_tokens": 256,
    "do_sample": False,
    "num_beams": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}
# Fixed options for every chat template; a model can add its own (such as enable_thinking).
TEMPLATE_BASE = {"tokenize": False, "add_generation_prompt": True}


def answer_items(model, tokenizer, items, template_kwargs, batch_size, device="cuda"):
    """Answer each item ({"id", "messages"}). Returns (results, settings).

    `settings` holds what a later reader needs to check the run: the
    generation settings, the chat-template options, and the SHA-256 of the
    first prompt as the model saw it.
    """
    # Pad on the left so every answer in a batch starts right after its prompt.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    template = {**TEMPLATE_BASE, **template_kwargs}

    results = []
    first_prompt = None
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        texts = [tokenizer.apply_chat_template(item["messages"], **template) for item in batch]
        if first_prompt is None:
            first_prompt = texts[0]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        output = model.generate(**inputs, **GENERATION, pad_token_id=tokenizer.pad_token_id)
        prompt_length = inputs["input_ids"].shape[1]
        for item, tokens in zip(batch, output):
            results.append({"id": item["id"], "raw": tokenizer.decode(tokens[prompt_length:], skip_special_tokens=True)})

    settings = {
        "generation": dict(GENERATION),
        "chat_template_options": template,
        "first_prompt_sha256": hashlib.sha256(first_prompt.encode("utf-8")).hexdigest() if first_prompt else None,
        "first_prompt": first_prompt if len(items) <= batch_size else None,
    }
    return results, settings
