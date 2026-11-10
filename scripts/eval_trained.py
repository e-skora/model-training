"""M1 Step 7 Part B: answer a frozen test set with the trained judge (base model plus adapter).

The adapter is loaded from the private Hugging Face repo at the exact commit
`scripts/train.py --stage upload` recorded in results/m1/trained-v1/train.meta.json,
on top of the base-model commit training used (the baseline most likely loaded
the same files, but did not record it: results/m1/trained-v1/base-files.md). It answers all three
versions of the test set (as is, swapped, no dates) with the baseline's
greedy settings and answer loop (`judge.generate`), so the scores compare
fairly with results/m1/baseline-v2/.

The test set is the final exam: it is read once, for the one real run.
Run from the repo folder:
    uv run modal run scripts/eval_trained.py --smoke --out /some/scratch/folder   # 8 validation pairs, not the test set
    uv run modal run scripts/eval_trained.py                                      # the one test-set run
    uv run modal run scripts/eval_trained.py --test data/m2/test-hard.jsonl --out results/m2/baseline/trained-v1
--test names another frozen file (default: the M1 test set); it is checked against STATE.md the same way.
Answers go to <out>/qwen3-4b-trained-v1.<version>.jsonl, settings to <out>/qwen3-4b-trained-v1.meta.json.
A smoke run adds "-smoke" to the name (qwen3-4b-trained-v1-smoke), so it can never write over the final files.

M2 Step 6 (D-060): --model v2 answers with the GRPO adapter the validation sets chose (step-250,
eskora/m2-contradiction-judge-qwen3-4b-grpo-lora at 0964d031), read from results/m2/grpo/train.meta.json,
on v1's pinned base. Before loading, the GPU job checks the adapter files' SHA-256 at that commit against the
run's saved step-250 files. Files are named qwen3-4b-grpo-v2.*, and v2 needs its own --out folder:
    uv run modal run scripts/eval_trained.py --model v2 --smoke --out /some/scratch/folder
    uv run modal run scripts/eval_trained.py --model v2 --test data/m2/test-hard.jsonl --out results/m2/v2/hard
    uv run modal run scripts/eval_trained.py --model v2 --test data/m1/test.jsonl --out results/m2/v2/m1
Before anything runs, the script stops if one of its output files already exists, or if the recorded
upload is not the adapter the validation set chose.
Score them with `uv run python -m judge.score <test file> <as_is> <swapped> <no_dates>`.
"""

import modal

from judge.modal_image import image

app = modal.App("m1-eval-trained")

NAME = "qwen3-4b-trained-v1"
OUT = "results/m1/trained-v1"
TEST_PATH = "data/m1/test.jsonl"  # checked against STATE.md's "Frozen files" before any run
VALID_PATH = "data/m1/valid.jsonl"
TRAIN_META = "results/m1/trained-v1/train.meta.json"
BATCH_SIZE = 16  # the same as the baseline

# M2 Step 6 (D-060): v2 is the GRPO checkpoint the validation sets chose, at the commit its upload recorded.
# `--model v2` scores it with the same loader, prompt, greedy settings and answer loop as v1.
V2 = {
    "name": "qwen3-4b-grpo-v2",
    "record": "results/m2/grpo/train.meta.json",
    "repo": "eskora/m2-contradiction-judge-qwen3-4b-grpo-lora",
    "commit": "0964d031073408707397ce7438fc7df7b8a772c0",
    "checkpoint": "step-250",
}
# The adapter files PEFT loads; their SHA-256 at that commit must equal the run's saved step-250 files.
CHECKED_FILES = ("adapter_config.json", "adapter_model.safetensors")


def files_problem(expected, got):
    """None when every expected file has the expected SHA-256, else what differs."""
    wrong = sorted(name for name in expected if got.get(name) != expected[name])
    return f"adapter files differ from the recorded fingerprints: {', '.join(wrong)}" if wrong else None


@app.function(image=image, gpu="L4", timeout=1800, secrets=[modal.Secret.from_name("huggingface")])
def answer_pairs(base, base_revision, adapter, adapter_revision, items, expected_files=None):
    import time

    start = time.time()
    import unsloth
    import peft
    import torch
    import transformers

    from judge.generate import answer_items
    from judge.load import load_judge

    checked = {}
    if expected_files:  # before loading or answering anything
        import hashlib

        from huggingface_hub import hf_hub_download

        for name in expected_files:
            with open(hf_hub_download(adapter, name, revision=adapter_revision), "rb") as handle:
                checked[name] = hashlib.sha256(handle.read()).hexdigest()
        problem = files_problem(expected_files, checked)
        if problem:
            raise RuntimeError(f"STOP: {problem}; nothing was answered")
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
        **({"adapter_files_sha256": checked} if checked else {}),
        **settings,
    }


def output_paths(out_dir, smoke, name=NAME):
    """The files a run writes: one answer file per version, then the settings file."""
    from pathlib import Path

    from judge.score import VERSIONS

    name = name + "-smoke" if smoke else name
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


def check_v2(grpo_meta, v1_meta):
    """Stop unless the GRPO record names v2 exactly (D-060) on v1's pinned base. Returns (upload, files)."""
    upload = check_upload(grpo_meta)
    pinned = {"repo": V2["repo"], "commit": V2["commit"], "checkpoint": V2["checkpoint"]}
    if {k: upload.get(k) for k in pinned} != pinned:
        raise SystemExit(f"STOP: the recorded upload is not v2 {pinned}")
    for key in ("base_model", "base_revision"):
        if grpo_meta[key] != v1_meta[key]:
            raise SystemExit(f"STOP: v2's {key} {grpo_meta[key]!r} is not v1's {v1_meta[key]!r}")
    saved = [s for s in grpo_meta.get("saved", []) if s.get("name") == V2["checkpoint"]]
    if len(saved) != 1 or not all(saved[0]["files"].get(name) for name in CHECKED_FILES):
        raise SystemExit(f"STOP: no fingerprints recorded for {V2['checkpoint']}")
    return upload, {name: saved[0]["files"][name] for name in CHECKED_FILES}


def evaluate(smoke, out, answer, train_meta_path=None, test_path=TEST_PATH, valid_path=VALID_PATH,
             state_path="STATE.md", model="v1", v1_meta_path=TRAIN_META):
    """Check the record and the output folder, then answer and save. `answer` runs the GPU job.

    `model` is "v1" (the M1 adapter, as before) or "v2" (the GRPO adapter, D-060)."""
    import json
    import time
    from pathlib import Path

    from judge.frozen import FrozenFileError, frozen_entries, frozen_key, load_frozen
    from judge.prompt import build_messages, prompt_sha256
    from judge.score import VERSIONS, derive
    from judge.train_data import load_pairs

    # Every check runs before the test set is read and before anything starts on Modal.
    if model not in ("v1", "v2"):
        raise SystemExit(f"STOP: unknown model {model!r}; choose v1 or v2")
    if train_meta_path is None:
        train_meta_path = TRAIN_META if model == "v1" else V2["record"]
    train_meta = json.loads(Path(train_meta_path).read_text(encoding="utf-8"))
    extra, call = {}, {}
    if model == "v1":
        upload, name = check_upload(train_meta), NAME
    else:
        if Path(out).resolve() == Path(OUT).resolve():
            raise SystemExit(f"STOP: {OUT} holds v1's answers; give v2 its own --out folder")
        v1_meta = json.loads(Path(v1_meta_path).read_text(encoding="utf-8"))
        upload, files = check_v2(train_meta, v1_meta)
        name, call = V2["name"], {"expected_files": files}
        from judge.reward import reward_sha256

        # The reward is not used to score; recorded to show src/ is as it was when v2 was trained.
        extra = {"adapter_files_expected": files, "reward_sha256": reward_sha256(),
                 "reward_sha256_training": train_meta.get("reward_sha256")}
    paths = output_paths(out, smoke, name)
    check_free(paths)

    if smoke:
        pairs, source = load_pairs(valid_path)[:8], str(valid_path)
    else:
        try:
            pairs, source = load_frozen(test_path, state_path), frozen_key(test_path, state_path)
        except FrozenFileError as error:
            raise SystemExit(f"STOP: {error}")

    items = [
        {"id": f"{version}/{shown['id']}", "messages": build_messages(shown)}
        for version in VERSIONS
        for shown in derive(pairs, version)
    ]
    start = time.time()
    result = answer(train_meta["base_model"], train_meta["base_revision"], upload["repo"], upload["commit"], items,
                    **call)
    wall = round(time.time() - start, 1)
    if call and files_problem(call["expected_files"], result.get("adapter_files_sha256", {})):
        raise SystemExit("STOP: the GPU job did not report the expected adapter fingerprints; nothing saved")

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
        **extra,
        **result,
    }
    with open(paths[-1], "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2) + "\n")
    print(f"{paths[-1].stem}: {len(items)} answers ({len(pairs)} pairs of {source} x {len(VERSIONS)} versions) in "
          f"{wall / 60:.1f} min (load {result['load_seconds']} s, answer {result['answer_seconds']} s, "
          f"peak VRAM {result['peak_vram_gib']} GiB)")


@app.local_entrypoint()
def main(smoke: bool = False, out: str = OUT, test: str = TEST_PATH, model: str = "v1"):
    evaluate(smoke, out, answer_pairs.remote, test_path=test, model=model)
