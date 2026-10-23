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
A smoke run uses the name qwen3-4b-trained-v1-smoke instead, so it can never write over the final files.
Before anything runs, the script stops if one of its output files already exists, or if the recorded
upload is not the adapter the validation set chose.
Score them with `uv run python -m judge.score data/m1/test.jsonl <as_is> <swapped> <no_dates>`.
"""

import modal

from judge.modal_image import image

app = modal.App("m1-eval-trained")

NAME = "qwen3-4b-trained-v1"
SMOKE_NAME = NAME + "-smoke"  # a smoke run's files never share a name with the final run's
OUT = "results/m1/trained-v1"
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


def output_paths(out_dir, smoke):
    """The files a run writes: one answer file per version, then the settings file."""
    from pathlib import Path

    from judge.score import VERSIONS

    name = SMOKE_NAME if smoke else NAME
    return [Path(out_dir) / f"{name}.{version}.jsonl" for version in VERSIONS] + [Path(out_dir) / f"{name}.meta.json"]


def check_free(paths):
    """Stop if any output file exists, so saved answers are never replaced."""
    taken = [str(p) for p in paths if p.exists()]
    if taken:
        raise SystemExit(f"STOP: these files already exist and would be replaced: {', '.join(taken)}. "
                         "Nothing ran. Use another --out folder.")


def check_upload(train_meta):
    """Stop unless train.meta.json records a private upload of the adapter the validation set chose."""
    from judge.prompt import prompt_sha256

    upload = train_meta.get("upload")
    if not upload or not upload.get("private"):
        raise SystemExit("STOP: no private adapter upload recorded in train.meta.json; run train.py --stage upload")
    chosen = (train_meta.get("choice") or {}).get("chosen")
    if upload.get("checkpoint") != chosen:
        raise SystemExit(f"STOP: the recorded upload is {upload.get('checkpoint')!r} but the validation set chose "
                         f"{chosen!r}; run train.py --stage upload")
    if train_meta["prompt_sha256"] != prompt_sha256():
        raise SystemExit("STOP: the prompt changed since training")
    return upload


def evaluate(smoke, out, answer, train_meta_path=TRAIN_META, test_path=TEST_PATH, valid_path=VALID_PATH,
             state_path="STATE.md"):
    """Check the record and the output folder, then answer and save. `answer` runs the GPU job."""
    import json
    import time
    from pathlib import Path

    from judge.frozen import FrozenFileError, frozen_entries, load_frozen
    from judge.prompt import build_messages, prompt_sha256
    from judge.score import VERSIONS, derive
    from judge.train_data import load_pairs

    # Every check runs before the test set is read and before anything starts on Modal.
    train_meta = json.loads(Path(train_meta_path).read_text(encoding="utf-8"))
    upload = check_upload(train_meta)
    paths = output_paths(out, smoke)
    check_free(paths)

    if smoke:
        pairs, source = load_pairs(valid_path)[:8], str(valid_path)
    else:
        try:
            pairs, source = load_frozen(test_path, state_path), str(test_path)
        except FrozenFileError as error:
            raise SystemExit(f"STOP: {error}")

    items = [
        {"id": f"{version}/{shown['id']}", "messages": build_messages(shown)}
        for version in VERSIONS
        for shown in derive(pairs, version)
    ]
    start = time.time()
    result = answer(train_meta["base_model"], train_meta["base_revision"], upload["repo"], upload["commit"], items)
    wall = round(time.time() - start, 1)

    Path(out).mkdir(parents=True, exist_ok=True)
    answers = {version: [] for version in VERSIONS}
    for line in result.pop("results"):
        version, pair_id = line["id"].split("/", 1)
        answers[version].append({"id": pair_id, "raw": line["raw"]})
    for version, path in zip(VERSIONS, paths):
        with open(path, "x", encoding="utf-8") as f:  # "x": fail rather than replace a file
            for line in answers[version]:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    meta = {
        "model": f"{train_meta['base_model']} + {upload['repo']}",
        "base_model": train_meta["base_model"],
        "base_revision": train_meta["base_revision"],
        "adapter_repo": upload["repo"],
        "adapter_revision": upload["commit"],
        "adapter_checkpoint": upload["checkpoint"],
        "data": source,
        "data_sha256": None if smoke else frozen_entries(Path(state_path))[source],
        "pairs": len(pairs),
        "versions": list(VERSIONS),
        "prompt_sha256": prompt_sha256(),
        "wall_seconds": wall,
        **result,
    }
    with open(paths[-1], "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2) + "\n")
    print(f"{paths[-1].stem}: {len(items)} answers ({len(pairs)} pairs of {source} x {len(VERSIONS)} versions) in "
          f"{wall / 60:.1f} min (load {result['load_seconds']} s, answer {result['answer_seconds']} s, "
          f"peak VRAM {result['peak_vram_gib']} GiB)")


@app.local_entrypoint()
def main(smoke: bool = False, out: str = OUT):
    evaluate(smoke, out, answer_pairs.remote)
