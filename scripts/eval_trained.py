"""M1 Step 7 Part B: answer the frozen test set with the trained judge (base model plus adapter).

The adapter is loaded from the private Hugging Face repo at the exact commit
`scripts/train.py --stage upload` recorded in results/m1/trained-v1/train.meta.json,
on top of the same base-model commit the baseline used. It answers all three
versions of the test set (as is, swapped, no dates) with the baseline's
greedy settings and answer loop (`judge.generate`), so the scores compare
fairly with results/m1/baseline-v2/.

The test set is the final exam: it is read once, for the one real run.
Run from the repo folder:
    uv run modal run scripts/eval_trained.py --smoke --out /some/scratch/folder   # 8 validation pairs, not the test set
    uv run modal run scripts/eval_trained.py                                      # the one test-set run
Answers go to <out>/qwen3-4b-trained-v1.<version>.jsonl, settings to <out>/qwen3-4b-trained-v1.meta.json.
Score them with `uv run python -m judge.score data/m1/test.jsonl <as_is> <swapped> <no_dates>`.
"""

import modal

from judge.modal_image import image

app = modal.App("m1-eval-trained")

NAME = "qwen3-4b-trained-v1"
TEST_PATH = "data/m1/test.jsonl"  # checked against STATE.md's "Frozen files" before any run
VALID_PATH = "data/m1/valid.jsonl"
TRAIN_META = "results/m1/trained-v1/train.meta.json"
BATCH_SIZE = 16  # the same as the baseline


@app.function(image=image, gpu="L4", timeout=1800, secrets=[modal.Secret.from_name("huggingface")])
def answer_pairs(base, base_revision, adapter, adapter_revision, items):
    import time

    start = time.time()
    import unsloth
    import peft
    import torch
    import transformers

    from judge.generate import answer_items
    from judge.load import load_judge

    model, tokenizer = load_judge(base, base_revision, adapter=adapter, adapter_revision=adapter_revision)
    loaded = time.time()
    with torch.no_grad():
        results, settings = answer_items(model, tokenizer, items, {}, BATCH_SIZE)
    return {
        "results": results,
        "loaded_model": model.base_model.config._name_or_path,
        "load_seconds": round(loaded - start, 1),
        "answer_seconds": round(time.time() - loaded, 1),
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "unsloth": unsloth.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        **settings,
    }


@app.local_entrypoint()
def main(smoke: bool = False, out: str = "results/m1/trained-v1"):
    import json
    import time
    from pathlib import Path

    from judge.frozen import FrozenFileError, frozen_entries, load_frozen
    from judge.prompt import build_messages, prompt_sha256
    from judge.score import VERSIONS, derive
    from judge.train_data import load_pairs

    train_meta = json.loads(Path(TRAIN_META).read_text(encoding="utf-8"))
    upload = train_meta.get("upload")
    if not upload or not upload.get("private"):
        raise SystemExit("STOP: no private adapter upload recorded in train.meta.json; run train.py --stage upload")
    if train_meta["prompt_sha256"] != prompt_sha256():
        raise SystemExit("STOP: the prompt changed since training")

    if smoke:
        pairs, source = load_pairs(VALID_PATH)[:8], VALID_PATH
    else:
        try:
            pairs, source = load_frozen(TEST_PATH), TEST_PATH
        except FrozenFileError as error:
            raise SystemExit(f"STOP: {error}")

    items = [
        {"id": f"{version}/{shown['id']}", "messages": build_messages(shown)}
        for version in VERSIONS
        for shown in derive(pairs, version)
    ]
    start = time.time()
    result = answer_pairs.remote(train_meta["base_model"], train_meta["base_revision"],
                                 upload["repo"], upload["commit"], items)
    wall = round(time.time() - start, 1)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers = {version: [] for version in VERSIONS}
    for line in result.pop("results"):
        version, pair_id = line["id"].split("/", 1)
        answers[version].append({"id": pair_id, "raw": line["raw"]})
    for version, lines in answers.items():
        with open(out_dir / f"{NAME}.{version}.jsonl", "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    meta = {
        "model": f"{train_meta['base_model']} + {upload['repo']}",
        "base_model": train_meta["base_model"],
        "base_revision": train_meta["base_revision"],
        "adapter_repo": upload["repo"],
        "adapter_revision": upload["commit"],
        "adapter_checkpoint": upload["checkpoint"],
        "data": source,
        "data_sha256": None if smoke else frozen_entries(Path("STATE.md"))[TEST_PATH],
        "pairs": len(pairs),
        "versions": list(VERSIONS),
        "prompt_sha256": prompt_sha256(),
        "wall_seconds": wall,
        **result,
    }
    (out_dir / f"{NAME}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{NAME}: {len(items)} answers ({len(pairs)} pairs of {source} x {len(VERSIONS)} versions) in "
          f"{wall / 60:.1f} min (load {result['load_seconds']} s, answer {result['answer_seconds']} s, "
          f"peak VRAM {result['peak_vram_gib']} GiB)")
