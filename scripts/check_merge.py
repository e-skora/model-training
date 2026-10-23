"""M1 Step 8 diagnosis: answer the validation set with the merged model at full 16-bit precision.

The GGUF on the Mac agreed with the trained adapter on 96.6% of validation
labels, below the 97% bar. Two steps lie between them:
  1. the merge: Unsloth adds the adapter into the original 16-bit weights,
     while training ran the adapter on the 4-bit copy of those weights;
  2. quantization: the merged model is shrunk to q4_k_m for the Mac.
This job runs step 1 alone: the 16-bit base, plus the adapter merged in, with
no quantization, through the same answer loop as Step 7 (`judge.generate`).
Comparing its answers with the adapter's and with the GGUF's shows which step
moves the labels. Validation set only; the test set is not read.
Run from the repo folder:
    uv run modal run scripts/check_merge.py
Answers go to results/m1/gguf-v1/valid/merged16.<version>.jsonl, settings to merged16.meta.json.
"""

import modal

from judge.modal_image import image

app = modal.App("m1-check-merge")

OUT = "results/m1/gguf-v1/valid"
NAME = "merged16"
VALID_PATH = "data/m1/valid.jsonl"
EXPORT_META = "results/m1/gguf-v1/export.meta.json"
BATCH_SIZE = 16  # the same as Step 7's validation answers
MAX_SEQ_LENGTH = 2048


@app.function(image=image, gpu="L4", timeout=1800, secrets=[modal.Secret.from_name("huggingface")])
def answer_merged(base, base_revision, adapter, adapter_revision, items):
    import time

    start = time.time()
    import unsloth
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from unsloth import FastLanguageModel

    from judge.generate import answer_items

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, revision=base_revision, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    loaded_model = model.config._name_or_path
    model = PeftModel.from_pretrained(model, adapter, revision=adapter_revision, is_trainable=False)
    model = model.merge_and_unload()  # as the export does: the adapter becomes part of the 16-bit weights
    FastLanguageModel.for_inference(model)
    loaded = time.time()
    with torch.no_grad():
        results, settings = answer_items(model, tokenizer, items, {}, BATCH_SIZE)
    return {
        "results": results,
        "loaded_model": loaded_model,
        "dtype": str(next(model.parameters()).dtype),
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
def main(out: str = OUT):
    import json
    import time
    from pathlib import Path

    from judge.prompt import build_messages, prompt_sha256
    from judge.score import VERSIONS, derive
    from judge.train_data import read_pairs

    export = json.loads(Path(EXPORT_META).read_text(encoding="utf-8"))
    if export["merge_base_revision_before"] != export["merge_base_revision_after"]:
        raise SystemExit("STOP: the 16-bit base moved during the export, so its commit is not known")
    paths = [Path(out) / f"{NAME}.{version}.jsonl" for version in VERSIONS] + [Path(out) / f"{NAME}.meta.json"]
    taken = [str(p) for p in paths if p.exists()]
    if taken:
        raise SystemExit(f"STOP: these files already exist and would be replaced: {', '.join(taken)}. Nothing ran.")

    pairs, valid_sha256 = read_pairs(VALID_PATH)
    items = [{"id": f"{version}/{shown['id']}", "messages": build_messages(shown)}
             for version in VERSIONS for shown in derive(pairs, version)]
    start = time.time()
    result = answer_merged.remote(export["merge_base"], export["merge_base_revision_before"],
                                  export["adapter_repo"], export["adapter_revision"], items)
    wall = round(time.time() - start, 1)

    answers = {version: [] for version in VERSIONS}
    for line in result.pop("results"):
        version, pair_id = line["id"].split("/", 1)
        answers[version].append({"id": pair_id, "raw": line["raw"]})
    Path(out).mkdir(parents=True, exist_ok=True)
    for version, path in zip(VERSIONS, paths):
        with open(path, "x", encoding="utf-8") as f:
            for line in answers[version]:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    meta = {
        "model": f"{export['merge_base']} (16-bit) + {export['adapter_repo']}, merged, not quantized",
        "base_model": export["merge_base"],
        "base_revision": export["merge_base_revision_before"],
        "adapter_repo": export["adapter_repo"],
        "adapter_revision": export["adapter_revision"],
        "data": VALID_PATH,
        "data_sha256": valid_sha256,
        "pairs": len(pairs),
        "versions": list(VERSIONS),
        "prompt_sha256": prompt_sha256(),
        "wall_seconds": wall,
        **result,
    }
    with open(paths[-1], "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2) + "\n")
    print(f"{NAME}: {len(items)} answers in {wall / 60:.1f} min (load {result['load_seconds']} s, "
          f"answer {result['answer_seconds']} s, loaded {result['loaded_model']}, {result['dtype']})")
