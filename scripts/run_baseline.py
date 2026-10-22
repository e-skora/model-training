"""M1 Steps 5 and 6: answer every test pair with untrained candidate models (the baseline).

Each candidate runs on its own L4 at the same time. It answers all three
versions of the test set (`judge.score.VERSIONS`: as is, swapped, no dates),
loading the model once. Answers use greedy
decoding (always the most likely next word, `judge.generate.GENERATION`).
Each run's .meta.json saves the generation settings, chat-template options,
the model commit it loaded, the SHA-256 of the prompt template (D-027) and
of the first prompt, so a rerun can be checked against it.

Run from the repo folder:
    uv run modal run scripts/run_baseline.py --models qwen3-4b-instruct-2507,qwen3-8b
Try a few pairs first, written somewhere else:
    uv run modal run scripts/run_baseline.py --limit 8 --out /some/scratch/folder
Answers go to <out>/<model>.<version>.jsonl, settings to <out>/<model>.meta.json.
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
TEST_PATH = "data/m1/test.jsonl"  # checked against STATE.md's "Frozen files" before any run
BATCH_SIZE = 16


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
    from huggingface_hub import HfApi
    from unsloth import FastLanguageModel

    from judge.generate import answer_items

    # Pin the exact Hugging Face commit, so the record says which files were loaded.
    # Unsloth passes `revision` to both the model and the tokenizer.
    revision = HfApi().model_info(repo).sha
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=repo, max_seq_length=2048, load_in_4bit=True, revision=revision
    )
    FastLanguageModel.for_inference(model)
    loaded = time.time()

    with torch.no_grad():
        results, settings = answer_items(model, tokenizer, items, template_kwargs, BATCH_SIZE)

    return {
        "results": results,
        "load_seconds": round(loaded - start, 1),
        "answer_seconds": round(time.time() - loaded, 1),
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "unsloth": unsloth.__version__,
        "transformers": transformers.__version__,
        "model_revision": revision,
        "tokenizer_revision": revision,
        **settings,
    }


@app.local_entrypoint()
def main(models: str = ",".join(CANDIDATES), limit: int = 0, out: str = "results/m1/baseline-v2"):
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from judge.frozen import FrozenFileError, load_frozen
    from judge.prompt import build_messages, prompt_sha256
    from judge.score import VERSIONS, derive

    try:
        pairs = load_frozen(TEST_PATH)
    except FrozenFileError as error:
        raise SystemExit(f"STOP: {error}")

    names = [name.strip() for name in models.split(",")]
    unknown = [name for name in names if name not in CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; choose from {list(CANDIDATES)}")

    if limit:
        pairs = pairs[:limit]
    # One item per pair and version; the id says which version ("swapped/t-001").
    items = [
        {"id": f"{version}/{shown['id']}", "messages": build_messages(shown)}
        for version in VERSIONS
        for shown in derive(pairs, version)
    ]
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
        answers = {version: [] for version in VERSIONS}
        for line in result.pop("results"):
            version, pair_id = line["id"].split("/", 1)
            answers[version].append({"id": pair_id, "raw": line["raw"]})
        for version, lines in answers.items():
            with open(out_dir / f"{name}.{version}.jsonl", "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
        meta = {
            "model": CANDIDATES[name][0],
            "pairs": len(pairs),
            "versions": list(VERSIONS),
            "prompt_sha256": prompt_sha256(),
            "wall_seconds": round(wall, 1),
            **result,
        }
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(items)} answers ({len(pairs)} pairs x {len(VERSIONS)} versions) in {wall / 60:.1f} min (load {result['load_seconds']} s, "
              f"answer {result['answer_seconds']} s, peak VRAM {result['peak_vram_gib']} GiB)")
