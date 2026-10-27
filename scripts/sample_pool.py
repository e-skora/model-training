"""M2 Step 3 part B: v1 answers each RL candidate 8 times, then the D-046 rule picks the RL pool.

Candidates: data/m2/train-hard.jsonl plus M1's 47 `unclear` training pairs (D-045).
Never a frozen test set and never a validation file: the script stops if a
candidate file is listed under "Frozen files" in STATE.md.

Two stages. `sample` runs v1 (the epoch-2 adapter, D-031, at the base and adapter
commits scripts/eval_trained.py uses) on a Modal L4: each candidate is shown the way
training will show it (judge.pool.show), and answered 8 times in one generate call at
temperature 1.0 (judge.generate.SAMPLING). `choose` runs on the Mac: it scores every
answer with judge.reward and applies judge.pool's rule, fixed before any answer was seen.

Run from the repo folder:
    uv run modal run scripts/sample_pool.py --stage sample --smoke --out /some/scratch/folder   # 16 candidates
    uv run modal run scripts/sample_pool.py --stage sample                                      # the full run
    uv run python scripts/sample_pool.py                                                        # choose (no GPU)
    uv run python scripts/sample_pool.py --samples <scratch>/v1-samples-smoke.jsonl --out <scratch>  # choose on a smoke run
Every output file is opened with "x", so a saved result is never replaced.
"""

import modal

from judge.modal_image import image

app = modal.App("m2-sample-pool")

HARD_PATH = "data/m2/train-hard.jsonl"
M1_TRAIN_PATH = "data/m1/train.jsonl"
TRAIN_META = "results/m1/trained-v1/train.meta.json"
OUT = "results/m2/pool"
NAME = "v1-samples"
SMOKE_NAME = NAME + "-smoke"
POOL_PATH = "data/m2/rl-pool.jsonl"
WRONG_PATH = "data/m2/rl-pool-always-wrong.jsonl"
SUMMARY_PATH = "results/m2/pool/summary.md"
BATCH_SIZE = 16  # prompts per generate call, so 128 answers at a time
SAMPLE_SEED = 3407  # torch's random seed for the sampling
SMOKE_PAIRS = 16
EXAMPLE_SEED = 5  # which mixed pairs the summary shows
L4_PER_SECOND = 0.000222  # https://modal.com/pricing (as in the M2 SPEC)
# v1 as D-031 pinned it; the upload record must name exactly these.
V1_ADAPTER = ("eskora/m1-contradiction-judge-qwen3-4b-lora", "1d9ebbf66e1a2abcad9932932bb7d810a38c59b9", "epoch-2")


@app.function(image=image, gpu="L4", timeout=3600, secrets=[modal.Secret.from_name("huggingface")])
def sample_pairs(base, base_revision, adapter, adapter_revision, items, n, seed):
    import time

    start = time.time()
    import unsloth
    import peft
    import torch
    import transformers

    from judge.generate import sample_items
    from judge.load import load_judge

    model, tokenizer = load_judge(base, base_revision, adapter=adapter, adapter_revision=adapter_revision)
    loaded = time.time()
    torch.manual_seed(seed)
    with torch.no_grad():
        results, settings = sample_items(model, tokenizer, items, {}, BATCH_SIZE, n)
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
        "torch_seed": seed,
        **settings,
    }


def check_free(paths):
    taken = [str(p) for p in paths if p.exists()]
    if taken:
        raise SystemExit(f"STOP: these files already exist and would be replaced: {', '.join(taken)}. Nothing ran.")


def check_v1(train_meta):
    """Stop unless the training record's upload is v1 exactly as D-031 pinned it, with today's prompt."""
    from judge.prompt import prompt_sha256

    upload = train_meta.get("upload") or {}
    chosen = (train_meta.get("choice") or {}).get("chosen")
    if (upload.get("repo"), upload.get("commit"), upload.get("checkpoint")) != V1_ADAPTER or chosen != V1_ADAPTER[2]:
        raise SystemExit(f"STOP: train.meta.json does not record v1 ({V1_ADAPTER}) as the chosen upload")
    if train_meta["prompt_sha256"] != prompt_sha256():
        raise SystemExit("STOP: the prompt changed since training")
    return upload


def load_candidates(state_path="STATE.md"):
    """The candidate pairs and the SHA-256 of each file read. Stops if a file is frozen."""
    from pathlib import Path

    from judge.frozen import frozen_entries
    from judge.pool import candidates
    from judge.train_data import read_pairs

    frozen = frozen_entries(Path(state_path))
    for path in (HARD_PATH, M1_TRAIN_PATH):
        if path in frozen:
            raise SystemExit(f"STOP: {path} is a frozen test file")
    hard, hard_sha = read_pairs(HARD_PATH)
    m1, m1_sha = read_pairs(M1_TRAIN_PATH)
    return candidates(hard, m1), {HARD_PATH: hard_sha, M1_TRAIN_PATH: m1_sha}


def sample(smoke, out, run):
    """Check v1's record and the output folder, then sample on Modal and save. `run` is the GPU job."""
    import json
    import time
    from pathlib import Path

    from judge.pool import KEEP_SEED, SAMPLES, SHOW_SEED, show
    from judge.prompt import prompt_sha256
    from judge.reward import reward_sha256

    train_meta = json.loads(Path(TRAIN_META).read_text(encoding="utf-8"))
    upload = check_v1(train_meta)
    name = SMOKE_NAME if smoke else NAME
    samples_path, meta_path = Path(out) / f"{name}.jsonl", Path(out) / f"{name}.meta.json"
    check_free([samples_path, meta_path])

    pairs, data_sha = load_candidates()
    shown = show(pairs)
    if smoke:
        shown = shown[:SMOKE_PAIRS]
    items = [{"id": s["id"], "messages": s["messages"]} for s in shown]

    start = time.time()
    result = run(train_meta["base_model"], train_meta["base_revision"], upload["repo"], upload["commit"],
                 items, SAMPLES, SAMPLE_SEED)
    wall = round(time.time() - start, 1)

    by_id = {line["id"]: line["raw"] for line in result.pop("results")}
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(samples_path, "x", encoding="utf-8") as f:
        for s in shown:
            line = {"id": s["id"], "version": s["version"], "swapped": s["swapped"], "dates": s["dates"],
                    "raw": by_id[s["id"]]}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    inside = result["load_seconds"] + result["answer_seconds"]
    meta = {
        "model": f"{train_meta['base_model']} + {upload['repo']}",
        "base_model": train_meta["base_model"],
        "base_revision": train_meta["base_revision"],
        "adapter_repo": upload["repo"],
        "adapter_revision": upload["commit"],
        "adapter_checkpoint": upload["checkpoint"],
        "data_sha256": data_sha,
        "candidates": len(pairs),
        "sampled": len(items),
        "smoke": smoke,
        "samples_per_pair": SAMPLES,
        "show_seed": SHOW_SEED,
        "keep_seed": KEEP_SEED,
        "prompt_sha256": prompt_sha256(),
        "reward_sha256": reward_sha256(),
        "wall_seconds": wall,
        "cost_low": round(inside * L4_PER_SECOND, 2),
        "cost_high": round(wall * L4_PER_SECOND, 2),
        **result,
    }
    if smoke:
        per_pair = result["answer_seconds"] / len(items)
        full_inside = result["load_seconds"] + per_pair * len(pairs)
        meta["projection"] = {
            "answer_seconds_per_pair": round(per_pair, 2),
            "full_minutes_inside": round(full_inside / 60, 1),
            "full_cost": round(full_inside * L4_PER_SECOND, 2),
        }
    with open(meta_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2) + "\n")
    print(f"{name}: {len(items)} candidates x {SAMPLES} answers in {wall / 60:.1f} min wall "
          f"(load {result['load_seconds']} s, answer {result['answer_seconds']} s, peak VRAM {result['peak_vram_gib']} GiB)")
    if smoke:
        print(f"Projection for all {len(pairs)} candidates: {meta['projection']}")


def pct(top, bottom):
    """A share as a whole percent, rounded half up from the exact counts (LESSONS 2026-10-23)."""
    from decimal import ROUND_HALF_UP, Decimal

    if not bottom:
        return "-"
    return f"{(Decimal(100) * top / bottom).quantize(Decimal(1), rounding=ROUND_HALF_UP)}%"


def summary_text(pairs_by_id, judged, pool, wrong, meta, pool_sha, samples_rows):
    """The plain-language summary, built only from the counts, so no number is typed by hand."""
    import random

    from judge.pool import ALWAYS_RIGHT_SHARE, KEEP_SEED, SAMPLES, SHOW_SEED
    from judge.score import trap_key

    by_trap = {}
    for row in judged:
        by_trap.setdefault(trap_key(pairs_by_id[row["id"]]), []).append(row)
    pool_set = set(pool)

    def counts(rows):
        pairs = [pairs_by_id[r["id"]] for r in rows]
        answers = sum(len(r["kinds"]) for r in rows)
        right = sum(k == p["label"] for r, p in zip(rows, pairs) for k in r["kinds"])
        decided = [r for r, p in zip(rows, pairs) if p["label"] != "unclear"]
        decided_answers = sum(len(r["kinds"]) for r in decided)
        unclear_on_decided = sum(k == "unclear" for r in decided for k in r["kinds"])
        return {
            "pairs": len(rows),
            "mixed": sum(r["outcome"] == "mixed" for r in rows),
            "right": sum(r["outcome"] == "always_right" for r in rows),
            "wrong": sum(r["outcome"] == "always_wrong" for r in rows),
            "always_unclear": sum(r["answer_counts"]["unclear"] == len(r["kinds"]) for r in rows),
            "in_pool": sum(r["id"] in pool_set for r in rows),
            "accuracy": pct(right, answers),
            "unclear_on_decided": pct(unclear_on_decided, decided_answers),
            "unclear_on_decided_n": (unclear_on_decided, decided_answers),
            "invalid": sum(r["answer_counts"]["invalid"] for r in rows),
            "answers": answers,
        }

    total = counts(judged)
    m1_unclear = sum(1 for r in judged if pairs_by_id[r["id"]].get("hard") == "m1_unclear")
    lines = [
        "# M2 Step 3 part B: the RL pool",
        "",
        "v1 (the M1 judge, epoch-2 adapter) answered every training candidate 8 times, with some randomness "
        "(temperature 1.0). Each answer got points from the reward function. GRPO, the RL method, can only learn "
        "from a pair where those points differ, so those pairs make up the pool, plus a few pairs v1 always gets right.",
        "",
        "## The result",
        "",
        f"- **Candidates:** {total['pairs']} pairs ({total['pairs'] - m1_unclear} M2 hard training pairs and "
        f"{m1_unclear} M1 `unclear` training pairs).",
        f"- **Mixed** (the 8 rewards differ): {total['mixed']}. All of them are in the pool.",
        f"- **Always right:** {total['right']}. {total['in_pool'] - total['mixed']} of them join the pool "
        f"({int(ALWAYS_RIGHT_SHARE * 100)}% of the mixed count, drawn with seed {KEEP_SEED}), so v1 does not forget them.",
        f"- **Always wrong the same way:** {total['wrong']}. Kept out and listed in `data/m2/rl-pool-always-wrong.jsonl` "
        "for Claude in Cowork to check for a wrong label.",
        f"- **Always `unclear`** (all 8 answers `unclear`, counted inside the groups above): {total['always_unclear']}.",
        f"- **Pool size: {len(pool)} pairs** (`data/m2/rl-pool.jsonl`).",
        f"- **v1's sampled accuracy:** {total['accuracy']} of {total['answers']} answers had the right label.",
        f"- **`unclear` on pairs that can be decided:** {total['unclear_on_decided_n'][0]} of "
        f"{total['unclear_on_decided_n'][1]} answers ({total['unclear_on_decided']}).",
        f"- **Invalid answers:** {total['invalid']} of {total['answers']}.",
        "",
        "Note: \"always right\" means the right label all 8 times. On a non-contradiction, a mix of right and wrong "
        "types gives different rewards (1.25 and 1.0), so that pair counts as mixed.",
        "",
        "## Per trap",
        "",
        "Accuracy is the share of the 8 answers with the right label. \"Unclear\" is the share of answers that said "
        "`unclear` on a pair that can be decided.",
        "",
        "| Trap (right answer) | Pairs | Mixed | Always right | Always wrong | Always unclear | In pool | Accuracy | Unclear |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for trap in sorted(by_trap):
        c = counts(by_trap[trap])
        lines.append(f"| {trap} | {c['pairs']} | {c['mixed']} | {c['right']} | {c['wrong']} | {c['always_unclear']} | "
                     f"{c['in_pool']} | {c['accuracy']} | {c['unclear_on_decided']} |")
    lines.append(f"| **all** | {total['pairs']} | {total['mixed']} | {total['right']} | {total['wrong']} | "
                 f"{total['always_unclear']} | {total['in_pool']} | {total['accuracy']} | {total['unclear_on_decided']} |")

    # Five mixed pairs, drawn with a fixed seed, with all 8 answers.
    raws = {row["id"]: row["raw"] for row in samples_rows}
    versions = {row["id"]: row["version"] for row in samples_rows}
    mixed_ids = sorted(r["id"] for r in judged if r["outcome"] == "mixed")
    examples = random.Random(EXAMPLE_SEED).sample(mixed_ids, min(5, len(mixed_ids)))
    lines += ["", f"## Five mixed pairs and their 8 answers (drawn with seed {EXAMPLE_SEED})", ""]
    judged_by_id = {r["id"]: r for r in judged}
    for pair_id in examples:
        pair, row = pairs_by_id[pair_id], judged_by_id[pair_id]
        right = pair["label"] + (f" / {pair['type']}" if pair["type"] else "")
        lines += [
            f"### {pair_id} ({trap_key(pair)}), shown {versions[pair_id].replace('_', ' ')}",
            "",
            f"- Memory a: {pair['a']['text']} ({pair['a']['written_at']})",
            f"- Memory b: {pair['b']['text']} ({pair['b']['written_at']})",
            f"- Right answer: {right}",
            "",
            "| # | Answer | Points |",
            "|---|---|---|",
        ]
        for i, (raw, points) in enumerate(zip(raws[pair_id], row["rewards"]), 1):
            text = " ".join(raw.split()).replace("|", "\\|")
            lines.append(f"| {i} | `{text}` | {points:+.2f} |")
        lines.append("")

    inside = (meta["load_seconds"] + meta["answer_seconds"]) / 60
    lines += [
        "## GPU time and cost",
        "",
        f"- One run on a Modal {meta['gpu']}: {inside:.1f} minutes inside the function "
        f"(load {meta['load_seconds'] / 60:.1f}, answering {meta['answer_seconds'] / 60:.1f}), "
        f"{meta['wall_seconds'] / 60:.1f} minutes from start to finish.",
        f"- Cost: about ${meta['cost_low']:.2f} to ${meta['cost_high']:.2f} (L4 at $0.000222 a second, "
        "https://modal.com/pricing; Modal bills while the container loads and answers, so the bill sits between the two).",
        "",
        "## Fingerprints and settings",
        "",
        f"- `data/m2/rl-pool.jsonl` SHA-256: `{pool_sha}`",
        f"- Reward fingerprint (`judge.reward.reward_sha256()`): `{meta['reward_sha256']}`",
        f"- Prompt fingerprint: `{meta['prompt_sha256']}`",
        "- Candidate files: " + ", ".join(f"`{p}` `{s}`" for p, s in meta["data_sha256"].items()),
        f"- v1: `{meta['base_model']}` at `{meta['base_revision']}` plus `{meta['adapter_repo']}` at "
        f"`{meta['adapter_revision']}` ({meta['adapter_checkpoint']}); loaded `{meta['loaded_model']}`.",
        f"- Sampling: {meta['generation']}, {SAMPLES} answers per pair in one generate call, torch seed {meta['torch_seed']}.",
        f"- Versions shown (judge.format's rules, seed {SHOW_SEED}): " + ", ".join(
            f"{v} {sum(1 for r in samples_rows if r['version'] == v)}"
            for v in ("as_is", "swapped", "no_dates", "swapped_no_dates")) + ".",
        "- Each pool row keeps its version, its 8 rewards and its answer counts. The 8 raw answers are in "
        "`results/m2/pool/v1-samples.jsonl`.",
        "",
    ]
    return "\n".join(lines)


def choose_pool(samples_path, pool_path, wrong_path, summary_path):
    """Score every saved answer, apply judge.pool's rule, and write the pool, the always-wrong list and the summary."""
    import hashlib
    import json
    from pathlib import Path

    from judge.pool import choose, judge_samples, show
    from judge.reward import reward_sha256

    samples_path = Path(samples_path)
    meta = json.loads(samples_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    check_free([Path(pool_path), Path(wrong_path), Path(summary_path)])
    if meta["reward_sha256"] != reward_sha256():
        raise SystemExit("STOP: the reward changed since sampling")
    pairs, data_sha = load_candidates()
    if data_sha != meta["data_sha256"]:
        raise SystemExit("STOP: a candidate file changed since sampling")
    shown = {s["id"]: s for s in show(pairs)}
    pairs_by_id = {pair["id"]: pair for pair in pairs}

    with open(samples_path, encoding="utf-8") as f:
        samples_rows = [json.loads(line) for line in f if line.strip()]
    if not meta["smoke"] and len(samples_rows) != len(pairs):
        raise SystemExit(f"STOP: {len(samples_rows)} sampled pairs, {len(pairs)} candidates")
    judged = []
    for row in samples_rows:
        if (row["swapped"], row["dates"]) != (shown[row["id"]]["swapped"], shown[row["id"]]["dates"]):
            raise SystemExit(f"STOP: {row['id']} was sampled in another version than judge.pool.show gives")
        if len(row["raw"]) != meta["samples_per_pair"]:
            raise SystemExit(f"STOP: {row['id']} has {len(row['raw'])} answers")
        judged.append({"id": row["id"], **judge_samples(pairs_by_id[row["id"]], row["raw"])})
    pool, wrong = choose(judged)

    by_id = {row["id"]: row for row in samples_rows}
    judged_by_id = {row["id"]: row for row in judged}

    def out_row(pair_id, with_answers=False):
        sample, j = by_id[pair_id], judged_by_id[pair_id]
        row = {**pairs_by_id[pair_id], "version": sample["version"], "swapped": sample["swapped"],
               "dates": sample["dates"], "rewards": j["rewards"], "answer_counts": j["answer_counts"],
               "outcome": j["outcome"]}
        if with_answers:
            row["answers"] = sample["raw"]
        return row

    for path, ids, with_answers in ((pool_path, pool, False), (wrong_path, wrong, True)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x", encoding="utf-8") as f:
            for pair_id in ids:
                f.write(json.dumps(out_row(pair_id, with_answers), ensure_ascii=False) + "\n")
    pool_sha = hashlib.sha256(Path(pool_path).read_bytes()).hexdigest()
    text = summary_text(pairs_by_id, judged, pool, wrong, meta, pool_sha, samples_rows)
    with open(summary_path, "x", encoding="utf-8") as f:
        f.write(text)
    print(f"pool {len(pool)} pairs, always wrong {len(wrong)}; rl-pool SHA-256 {pool_sha}")


@app.local_entrypoint()
def main(stage: str = "sample", smoke: bool = False, out: str = OUT):
    if stage != "sample":
        raise SystemExit("the choose stage runs without Modal: uv run python scripts/sample_pool.py")
    sample(smoke, out, sample_pairs.remote)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Choose the RL pool from saved samples (no GPU).")
    parser.add_argument("--samples", default=f"{OUT}/{NAME}.jsonl")
    parser.add_argument("--out", default=None, help="write the pool, list and summary here instead (for smoke runs)")
    args = parser.parse_args()
    if args.out:
        from pathlib import Path

        paths = [str(Path(args.out) / Path(p).name) for p in (POOL_PATH, WRONG_PATH, SUMMARY_PATH)]
    else:
        paths = [POOL_PATH, WRONG_PATH, SUMMARY_PATH]
    choose_pool(args.samples, *paths)
