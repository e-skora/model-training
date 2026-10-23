"""M1 Step 8: turn the trained judge into one GGUF file that Ollama can run on the Mac.

GGUF is the single-file model format llama.cpp and Ollama run. The export
follows Unsloth's "Saving to GGUF" page (https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf):
load the base model with the chosen adapter, then call
`model.save_pretrained_gguf(folder, tokenizer, quantization_method="q4_k_m")`.
In the pinned Unsloth 2026.10.7 that call merges the adapter into the original
16-bit weights (it downloads `unsloth/Qwen3-4B-Instruct-2507`, not the 4-bit
copy used for training), converts the merge to GGUF and shrinks it to q4_k_m
with a prebuilt llama.cpp from Unsloth's release page, pinned below.

The adapter is loaded from its folder with `FastLanguageModel.from_pretrained`,
as Unsloth asks. Attaching it with `peft.PeftModel.from_pretrained` (as
`judge.load` does for answering) would make the save call reach the base model
and write it without the adapter; Unsloth 2026.10.7 refuses that case.

The q4_k_m file goes to a private Hugging Face repo. Its SHA-256 (D-027) and
the export settings go to results/m1/gguf-v1/export.meta.json.
Run from the repo folder:
    uv run modal run scripts/export_gguf.py
Before anything runs, the script stops if the record file already exists, or if
the recorded adapter upload is not the one the validation set chose.
"""

import modal

from judge.modal_image import image

app = modal.App("m1-export-gguf")

OUT = "results/m1/gguf-v1/export.meta.json"
TRAIN_META = "results/m1/trained-v1/train.meta.json"
GGUF_REPO = "eskora/m1-contradiction-judge-qwen3-4b-gguf"
GGUF_NAME = "m1-contradiction-judge-qwen3-4b-Q4_K_M.gguf"
QUANTIZATION = "q4_k_m"
# The 16-bit weights Unsloth merges the adapter into (it maps the 4-bit name to this one).
MERGE_BASE = "unsloth/Qwen3-4B-Instruct-2507"
# Unsloth fetches a prebuilt, SHA-256-checked llama.cpp from github.com/unslothai/llama.cpp/releases.
# Pinned to the latest release on 2026-10-22 so the same converter runs every time.
LLAMA_CPP_TAG = "b11798-mix-f8a7950"
MAX_SEQ_LENGTH = 2048  # the same as judge.load

# libgomp1 and libcurl4 for the prebuilt llama.cpp tools; git, cmake and a compiler only if Unsloth must build it.
export_image = image.apt_install("git", "curl", "libgomp1", "libcurl4", "cmake", "build-essential")


@app.function(
    image=export_image,
    gpu="L4",  # loading the 4-bit training copy needs a CUDA GPU
    cpu=4,
    memory=24576,
    timeout=2400,
    secrets=[modal.Secret.from_name("huggingface")],
    env={"UNSLOTH_LLAMA_TAG": LLAMA_CPP_TAG},
)
def export(base, base_revision, adapter, adapter_revision, repo, filename):
    import hashlib
    import json
    import os
    import time
    from pathlib import Path

    start = time.time()
    import unsloth
    import peft
    import torch
    import transformers
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from unsloth import FastLanguageModel

    api = HfApi()
    # Unsloth loads the base at its latest commit, so stop unless that is still the commit training used.
    if api.model_info(base).sha != base_revision:
        raise RuntimeError(f"{base} moved on from {base_revision}; the export would not use the training base")
    # A repo left empty by a failed upload may be reused; one that holds the file is never written over.
    if api.repo_exists(repo) and (not api.model_info(repo).private or api.file_exists(repo, filename)):
        raise RuntimeError(f"{repo} already holds {filename} or is public; this export never writes over it")
    merge_base_before = api.model_info(MERGE_BASE).sha

    folder = snapshot_download(adapter, revision=adapter_revision, local_dir="/tmp/adapter")
    adapter_config = json.loads(Path(folder, "adapter_config.json").read_text())
    if adapter_config["base_model_name_or_path"] != base:
        raise RuntimeError(f"the adapter names base {adapter_config['base_model_name_or_path']}, not {base}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=folder, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True
    )
    if not isinstance(model, PeftModel):
        raise RuntimeError("the adapter did not load as a PEFT model, so the export would leave it out")
    loaded_model = model.base_model.config._name_or_path
    loaded = time.time()

    saved = model.save_pretrained_gguf("/tmp/gguf", tokenizer, quantization_method=QUANTIZATION)
    exported = time.time()
    files = [str(f) for f in saved["gguf_files"]]
    q4 = [f for f in files if QUANTIZATION in Path(f).name.lower()]
    if len(q4) != 1:
        raise RuntimeError(f"expected one {QUANTIZATION} file, got {files}")
    path = q4[0]
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    sha256 = digest.hexdigest()
    size = os.path.getsize(path)

    info_file = Path(os.path.expanduser("~/.unsloth/llama.cpp/UNSLOTH_PREBUILT_INFO.json"))
    llama_cpp = json.loads(info_file.read_text()) if info_file.is_file() else None

    api.create_repo(repo, private=True, repo_type="model", exist_ok=True)
    if not api.model_info(repo).private:
        raise RuntimeError(f"{repo} is not private; stopping before the upload")
    commit = api.upload_file(path_or_fileobj=path, path_in_repo=filename, repo_id=repo,
                             commit_message=f"M1 judge v1, {QUANTIZATION}, SHA-256 {sha256}")
    revision = commit.oid
    hub_info = api.get_paths_info(repo, [filename], revision=revision)[0]
    hub_sha256 = hub_info.lfs.sha256 if hub_info.lfs else None
    if hub_sha256 != sha256:
        raise RuntimeError(f"Hugging Face holds SHA-256 {hub_sha256}, not {sha256}")

    return {
        "loaded_model": loaded_model,
        "merge_base": MERGE_BASE,
        "merge_base_revision_before": merge_base_before,
        "merge_base_revision_after": api.model_info(MERGE_BASE).sha,
        "local_files": [{"name": Path(f).name, "bytes": os.path.getsize(f)} for f in files],
        "source_file": Path(path).name,
        "sha256": sha256,
        "bytes": size,
        "repo": repo,
        "repo_private": api.model_info(repo).private,
        "file": filename,
        "commit": revision,
        "commit_url": commit.commit_url,
        "hub_sha256": hub_sha256,
        "llama_cpp": llama_cpp,
        "load_seconds": round(loaded - start, 1),
        "export_seconds": round(exported - loaded, 1),
        "upload_seconds": round(time.time() - exported, 1),
        "gpu": torch.cuda.get_device_name(0),
        "unsloth": unsloth.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }


def check_upload(train_meta):
    """Stop unless train.meta.json records a private upload of the adapter the validation set chose."""
    from judge.prompt import prompt_sha256

    upload = train_meta.get("upload")
    if not upload or not upload.get("private"):
        raise SystemExit("STOP: no private adapter upload recorded in train.meta.json")
    chosen = (train_meta.get("choice") or {}).get("chosen")
    if upload.get("checkpoint") != chosen:
        raise SystemExit(f"STOP: the recorded upload is {upload.get('checkpoint')!r} but the validation set chose {chosen!r}")
    if train_meta["prompt_sha256"] != prompt_sha256():
        raise SystemExit("STOP: the prompt changed since training")
    return upload


def run_export(out, run, train_meta_path=TRAIN_META):
    """Check the record and the output file, then export and save the record. `run` runs the Modal job."""
    import json
    import time
    from pathlib import Path

    from judge.prompt import prompt_sha256

    train_meta = json.loads(Path(train_meta_path).read_text(encoding="utf-8"))
    upload = check_upload(train_meta)
    out = Path(out)
    if out.exists():
        raise SystemExit(f"STOP: {out} already exists and would be replaced. Nothing ran.")

    start = time.time()
    result = run(train_meta["base_model"], train_meta["base_revision"], upload["repo"], upload["commit"],
                 GGUF_REPO, GGUF_NAME)
    meta = {
        "base_model": train_meta["base_model"],
        "base_revision": train_meta["base_revision"],
        "adapter_repo": upload["repo"],
        "adapter_revision": upload["commit"],
        "adapter_checkpoint": upload["checkpoint"],
        "prompt_sha256": prompt_sha256(),
        "method": f"model.save_pretrained_gguf(folder, tokenizer, quantization_method={QUANTIZATION!r})",
        "quantization": QUANTIZATION,
        "llama_cpp_tag": LLAMA_CPP_TAG,
        "max_seq_length": MAX_SEQ_LENGTH,
        "wall_seconds": round(time.time() - start, 1),
        **result,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:  # "x": fail rather than replace a record
        f.write(json.dumps(meta, indent=2) + "\n")
    print(f"{result['file']}: {result['bytes'] / 1e9:.2f} GB, SHA-256 {result['sha256']}, "
          f"{result['repo']} at {result['commit']} (wall {meta['wall_seconds'] / 60:.1f} min)")


@app.local_entrypoint()
def main(out: str = OUT):
    run_export(out, export.remote)
