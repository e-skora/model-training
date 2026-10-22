"""M1 Step 5: answer every test pair with untrained candidate models (the baseline).

Each candidate runs on its own L4 at the same time. Answers use greedy
decoding (always the most likely next word), so a rerun gives the same answers.

Run from the repo folder:
    uv run modal run scripts/run_baseline.py
Try a few pairs first, written somewhere else:
    uv run modal run scripts/run_baseline.py --limit 8 --out /some/scratch/folder
Score the answers afterwards with `uv run python -m judge.score`.
"""

import modal

from judge.modal_image import image

app = modal.App("m1-baseline")

# Short name -> Hugging Face repo, and extra settings for its chat template.
CANDIDATES = {
    "qwen3-4b-instruct-2507": ("unsloth/Qwen3-4B-Instruct-2507", {}),
    # Qwen3-8B thinks out loud by default; Qwen's model card turns it off this way.
    "qwen3-8b": ("unsloth/Qwen3-8B", {"enable_thinking": False}),
    "phi-4-mini-instruct": ("unsloth/Phi-4-mini-instruct", {}),
}
TEST_PATH = "data/m1/test.jsonl"
TEST_SHA256 = "6e2a08323f445477f9600e371f1074ad3e2b41ef97d7072d6734ef481f6e574b"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 256


@app.function(
    image=image,
    gpu="L4",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
)
def answer_pairs(repo, template_kwargs, items):
    import time

    start = time.time()
    import unsloth  # import first, as Unsloth asks
    import torch
    import transformers
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=repo, max_seq_length=2048, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    loaded = time.time()

    # Pad on the left so every answer in a batch starts right after its prompt.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
        texts = [
            tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=True, **template_kwargs
            )
            for item in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy decoding
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        for item, tokens in zip(batch, new_tokens):
            results.append({"id": item["id"], "raw": tokenizer.decode(tokens, skip_special_tokens=True)})

    return {
        "results": results,
        "load_seconds": round(loaded - start, 1),
        "answer_seconds": round(time.time() - loaded, 1),
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "unsloth": unsloth.__version__,
        "transformers": transformers.__version__,
        "first_prompt": texts[0] if len(items) <= BATCH_SIZE else None,
    }


@app.local_entrypoint()
def main(models: str = ",".join(CANDIDATES), limit: int = 0, out: str = "results/m1/baseline"):
    import hashlib
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from judge.prompt import build_messages
    from judge.score import load_jsonl

    digest = hashlib.sha256(Path(TEST_PATH).read_bytes()).hexdigest()
    if digest != TEST_SHA256:
        raise SystemExit(f"STOP: {TEST_PATH} SHA-256 is {digest}, not the frozen {TEST_SHA256}")

    names = [name.strip() for name in models.split(",")]
    unknown = [name for name in names if name not in CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; choose from {list(CANDIDATES)}")

    pairs = load_jsonl(TEST_PATH)
    if limit:
        pairs = pairs[:limit]
    items = [{"id": pair["id"], "messages": build_messages(pair)} for pair in pairs]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run(name):
        repo, template_kwargs = CANDIDATES[name]
        start = time.time()
        try:
            result = answer_pairs.remote(repo, template_kwargs, items)
        except Exception as error:  # report and keep the other models going
            return name, None, repr(error), time.time() - start
        return name, result, None, time.time() - start

    with ThreadPoolExecutor(len(names)) as pool:
        outcomes = list(pool.map(run, names))

    for name, result, error, wall in outcomes:
        if error:
            print(f"{name}: FAILED after {wall / 60:.1f} min: {error}")
            continue
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for line in result.pop("results"):
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        meta = {"model": CANDIDATES[name][0], "pairs": len(items), "wall_seconds": round(wall, 1), **result}
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(items)} answers in {wall / 60:.1f} min (load {result['load_seconds']} s, "
              f"answer {result['answer_seconds']} s, peak VRAM {result['peak_vram_gib']} GiB)")
