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
    uv run modal run scripts/run_baseline.py --smoke --out /some/scratch/folder  # 8 validation pairs, not the test set
Another frozen test file (default: the M1 test set), checked against STATE.md the same way:
    uv run modal run scripts/run_baseline.py --models qwen3-8b --test data/m2/test-hard.jsonl --out results/m2/baseline/qwen3-8b
Answers go to <out>/<model>.<version>.jsonl, settings to <out>/<model>.meta.json
(<model>-smoke for a smoke run, so it never shares a name with a real run).
Before anything runs, the script stops if one of its output files already exists.
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
VALID_PATH = "data/m1/valid.jsonl"  # a smoke run answers the first 8 of these, never a test set
SMOKE_PAIRS = 8
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


def output_paths(out_dir, name, smoke=False):
    """The files a run writes for one model: one answer file per version, then the settings file."""
    from pathlib import Path

    from judge.score import VERSIONS

    stem = name + ("-smoke" if smoke else "")
    return [Path(out_dir) / f"{stem}.{version}.jsonl" for version in VERSIONS] + [Path(out_dir) / f"{stem}.meta.json"]


def check_free(paths):
    """Stop if any output file exists, so saved answers are never replaced."""
    taken = [str(p) for p in paths if p.exists()]
    if taken:
        raise SystemExit(f"STOP: these files already exist and would be replaced: {', '.join(taken)}. "
                         "Nothing ran. Use another --out folder.")


def run_models(names, out, answer, limit=0, smoke=False, test_path=TEST_PATH, valid_path=VALID_PATH,
               state_path=None):
    """Check the names, the data and the output folder, then answer and save. `answer` runs one GPU job."""
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from judge.frozen import REPO, FrozenFileError, frozen_entries, frozen_key, load_frozen
    from judge.prompt import build_messages, prompt_sha256
    from judge.score import VERSIONS, derive
    from judge.train_data import read_pairs

    unknown = [name for name in names if name not in CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; choose from {list(CANDIDATES)}")
    # Every check runs before the test set is read and before anything starts on Modal.
    paths = {name: output_paths(out, name, smoke) for name in names}
    check_free([p for name in names for p in paths[name]])

    if smoke:
        pairs, data_sha256 = read_pairs(valid_path)
        pairs, source = pairs[:SMOKE_PAIRS], str(valid_path)
    else:
        try:
            pairs = load_frozen(test_path, state_path)
            source = frozen_key(test_path, state_path)
            data_sha256 = frozen_entries(Path(state_path) if state_path else REPO / "STATE.md")[source]
        except FrozenFileError as error:
            raise SystemExit(f"STOP: {error}")
    if limit:
        pairs = pairs[:limit]
    # One item per pair and version; the id says which version ("swapped/t-001").
    items = [
        {"id": f"{version}/{shown['id']}", "messages": build_messages(shown)}
        for version in VERSIONS
        for shown in derive(pairs, version)
    ]

    def run(name):
        repo, template_kwargs = CANDIDATES[name]
        start = time.time()
        try:
            result = answer(repo, template_kwargs, items)
        except Exception as error:  # report and keep the other models going
            return name, None, repr(error), time.time() - start
        return name, result, None, time.time() - start

    with ThreadPoolExecutor(len(names)) as pool:
        outcomes = list(pool.map(run, names))

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, result, error, wall in outcomes:
        if error:
            print(f"{name}: FAILED after {wall / 60:.1f} min: {error}")
            continue
        answers = {version: [] for version in VERSIONS}
        for line in result.pop("results"):
            version, pair_id = line["id"].split("/", 1)
            answers[version].append({"id": pair_id, "raw": line["raw"]})
        for version, path in zip(VERSIONS, paths[name]):
            with open(path, "x", encoding="utf-8") as f:  # "x": fail rather than replace a file
                for line in answers[version]:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
        meta = {
            "model": CANDIDATES[name][0],
            "data": source,
            "data_sha256": data_sha256,
            "pairs": len(pairs),
            "versions": list(VERSIONS),
            "prompt_sha256": prompt_sha256(),
            "wall_seconds": round(wall, 1),
            **result,
        }
        with open(paths[name][-1], "x", encoding="utf-8") as f:
            f.write(json.dumps(meta, indent=2) + "\n")
        print(f"{name}: {len(items)} answers ({len(pairs)} pairs of {source} x {len(VERSIONS)} versions) in "
              f"{wall / 60:.1f} min (load {result['load_seconds']} s, answer {result['answer_seconds']} s, "
              f"peak VRAM {result['peak_vram_gib']} GiB)")
    return outcomes


@app.local_entrypoint()
def main(models: str = ",".join(CANDIDATES), limit: int = 0, out: str = "results/m1/baseline-v2",
         smoke: bool = False, test: str = TEST_PATH):
    run_models([name.strip() for name in models.split(",")], out, answer_pairs.remote, limit=limit, smoke=smoke,
               test_path=test)
