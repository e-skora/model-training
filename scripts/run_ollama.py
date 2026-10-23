"""M1 Step 8: answer pairs with the GGUF judge through Ollama on the Mac, and compare answers.

Ollama runs the model on the Mac and answers on a local web address
(http://127.0.0.1:11434). Each pair gets the shared prompt from `judge.prompt`
(same text, same SHA-256 as training and scoring), in the three versions of
`judge.score.VERSIONS` (as is, swapped, no dates).

Memory text is written only to the answers file, never to the screen or the
settings file (Aeverum §12.4). Errors name the pair id, never its text.

Before the first question the script checks, through Ollama itself, that the
model runs the exact GGUF file the export recorded (its SHA-256), with the chat
template and settings written in ollama/Modelfile. A frozen file (the test set)
is checked against STATE.md and needs --final, because it is read once, at the end.

Run from the repo folder, with Ollama open:
    uv run python scripts/run_ollama.py answer data/m1/valid.jsonl results/m1/gguf-v1/valid
    uv run python scripts/run_ollama.py answer data/m1/valid.jsonl results/m1/gguf-v1/valid --raw
    uv run python scripts/run_ollama.py compare data/m1/valid.jsonl \
        results/m1/gguf-v1/valid/m1-judge-v1 results/m1/trained-v1/valid/epoch-2
    uv run python scripts/run_ollama.py answer data/m1/test.jsonl results/m1/gguf-v1 --final
`--raw` skips Ollama's chat template and sends the prompt exactly as the
training template writes it (`render_prompt`). Its answers should be the same
as the normal run's, which proves the Modelfile template matches training.
"""

import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from judge.frozen import FrozenFileError, frozen_entries, load_frozen
from judge.prompt import build_messages, prompt_sha256
from judge.score import VERSIONS, derive, load_jsonl, score
from judge.train_data import read_pairs

REPO = Path(__file__).resolve().parents[1]
HOST = "http://127.0.0.1:11434"
MODEL = "m1-judge:v1"
NAME = "m1-judge-v1"
MODELFILE = REPO / "ollama" / "Modelfile"
EXPORT_META = REPO / "results" / "m1" / "gguf-v1" / "export.meta.json"
TIMEOUT = 300  # seconds for one answer, including loading the model the first time
KEEP_ALIVE = "15m"


class OllamaError(Exception):
    """Ollama could not be reached, or it runs something other than the recorded model."""


def render_prompt(messages):
    """The prompt as the training chat template writes it for a system and a user message, ready for the answer.

    Qwen3-4B-Instruct-2507's template (chat_template.jinja in the adapter repo)
    writes each message as <|im_start|>role, a new line, the text, <|im_end|>
    and a new line, then <|im_start|>assistant and a new line. No start token.
    """
    turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return turns + "<|im_start|>assistant\n"


def parse_modelfile(text):
    """Read FROM, TEMPLATE and PARAMETER lines. Returns {"from", "template", "parameters"}.

    `parameters` maps a name to a list of its values as text, since `stop` may appear more than once.
    """
    template = re.search(r'^TEMPLATE """(.*?)"""', text, re.DOTALL | re.MULTILINE)
    source = re.search(r"^FROM (.+)$", text, re.MULTILINE)
    parameters = {}
    for name, value in re.findall(r"^PARAMETER (\S+) (.+)$", text, re.MULTILINE):
        parameters.setdefault(name, []).append(value.strip().strip('"'))
    return {
        "from": source.group(1).strip() if source else None,
        "template": template.group(1) if template else None,
        "parameters": parameters,
    }


def parse_shown_parameters(text):
    """Read the "parameters" text of /api/show (lines of name and value) into the same shape as parse_modelfile."""
    parameters = {}
    for line in (text or "").splitlines():
        if line.strip():
            name, value = line.split(None, 1)
            parameters.setdefault(name, []).append(value.strip().strip('"'))
    return parameters


def same_values(a, b):
    """Two parameter lists match when each value is equal as a number, or as text."""
    def norm(value):
        try:
            return float(value)
        except ValueError:
            return value
    return sorted(map(str, map(norm, a))) == sorted(map(str, map(norm, b)))


def post(host, path, body=None):
    """Send one request to Ollama and return its JSON reply. Errors say only what Ollama said went wrong."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(host + path, data=data, headers={"Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as reply:
            return json.loads(reply.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            said = json.loads(error.read().decode("utf-8")).get("error")
        except ValueError:
            said = None
        raise OllamaError(f"{path} answered {error.code}: {said}") from None
    except (urllib.error.URLError, OSError) as error:
        raise OllamaError(f"cannot reach Ollama at {host} ({error.__class__.__name__}); is the Ollama app open?") from None


def check_model(host, model, modelfile_path, expected_sha256):
    """Check that `model` runs the recorded GGUF with the Modelfile's template and settings. Returns what it saw."""
    modelfile = parse_modelfile(Path(modelfile_path).read_text(encoding="utf-8"))
    shown = post(host, "/api/show", {"model": model})
    # Ollama names the model file by its SHA-256, so its FROM line shows the digest of the GGUF it runs.
    digests = re.findall(r"sha256[-:]([0-9a-f]{64})", shown.get("modelfile", ""))
    if expected_sha256 not in digests:
        raise OllamaError(f"{model} does not run the exported GGUF (SHA-256 {expected_sha256}); "
                          f"it runs {digests or 'an unknown file'}. Recreate it from ollama/Modelfile.")
    if shown.get("template") != modelfile["template"]:
        raise OllamaError(f"{model} uses a different chat template from ollama/Modelfile. Recreate it.")
    parameters = parse_shown_parameters(shown.get("parameters"))
    if set(parameters) != set(modelfile["parameters"]) or not all(
            same_values(parameters[k], modelfile["parameters"][k]) for k in parameters):
        raise OllamaError(f"{model} settings {parameters} differ from ollama/Modelfile {modelfile['parameters']}")
    if not same_values(parameters.get("temperature", ["missing"]), ["0"]):
        raise OllamaError("the Modelfile must set temperature 0")
    return {
        "model_sha256": expected_sha256,
        "template_sha256": hashlib.sha256(modelfile["template"].encode("utf-8")).hexdigest(),
        "parameters": modelfile["parameters"],
        "ollama_version": post(host, "/api/version").get("version"),
    }


def ask(host, model, messages, raw=False):
    """One answer. Returns (text, timing). Only the default chat route is what Aeverum would call."""
    start = time.perf_counter()
    if raw:
        reply = post(host, "/api/generate", {"model": model, "prompt": render_prompt(messages), "raw": True,
                                             "stream": False, "keep_alive": KEEP_ALIVE})
        text = reply.get("response")
    else:
        reply = post(host, "/api/chat", {"model": model, "messages": messages, "stream": False,
                                         "keep_alive": KEEP_ALIVE})
        text = (reply.get("message") or {}).get("content")
    timing = {
        "seconds": round(time.perf_counter() - start, 3),
        "load_seconds": round(reply.get("load_duration", 0) / 1e9, 3),
        "prompt_tokens": reply.get("prompt_eval_count"),
        "answer_tokens": reply.get("eval_count"),
        "done_reason": reply.get("done_reason"),
    }
    return text, timing


def load_data(path, final, state_path):
    """Pairs and their SHA-256. A frozen file is checked against STATE.md and needs `final`."""
    state_path = Path(state_path)
    try:
        key = Path(path).resolve().relative_to(state_path.resolve().parent).as_posix()
    except ValueError:
        key = None
    if key in frozen_entries(state_path):
        if not final:
            raise SystemExit(f"STOP: {key} is frozen (the final exam). Use --final for the one run, after validation passes.")
        try:
            return load_frozen(path, state_path), frozen_entries(state_path)[key], key
        except FrozenFileError as error:
            raise SystemExit(f"STOP: {error}") from None
    if final:
        raise SystemExit(f"STOP: --final is only for a frozen file, and {path} is not one")
    pairs, sha256 = read_pairs(path)
    return pairs, sha256, str(path)


def output_paths(out_dir, raw):
    name = NAME + ("-raw" if raw else "")
    return [Path(out_dir) / f"{name}.{version}.jsonl" for version in VERSIONS] + [Path(out_dir) / f"{name}.meta.json"]


def machine():
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except OSError:
        chip = ""
    return {"system": platform.system(), "machine": platform.machine(), "chip": chip or None}


def answer(data, out, raw=False, final=False, host=HOST, model=MODEL, modelfile=MODELFILE,
           export_meta=EXPORT_META, state_path=REPO / "STATE.md"):
    """Check everything, answer every pair in every version, then save answers and settings."""
    paths = output_paths(out, raw)
    taken = [str(p) for p in paths if p.exists()]
    if taken:
        raise SystemExit(f"STOP: these files already exist and would be replaced: {', '.join(taken)}. Nothing ran.")
    expected = json.loads(Path(export_meta).read_text(encoding="utf-8"))["sha256"]
    pairs, data_sha256, source = load_data(data, final, state_path)
    try:
        seen = check_model(host, model, modelfile, expected)
    except OllamaError as error:
        raise SystemExit(f"STOP: {error}") from None

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    answers = {version: [] for version in VERSIONS}
    timings = []
    for version in VERSIONS:
        for shown in derive(pairs, version):
            try:
                text, timing = ask(host, model, build_messages(shown), raw=raw)
            except OllamaError as error:
                raise SystemExit(f"STOP at {version}/{shown['id']}: {error}") from None
            answers[version].append({"id": shown["id"], "raw": text})
            timings.append({"id": f"{version}/{shown['id']}", **timing})
        print(f"{version}: {len(answers[version])} answers")

    Path(out).mkdir(parents=True, exist_ok=True)
    for version, path in zip(VERSIONS, paths):
        with open(path, "x", encoding="utf-8") as f:  # "x": fail rather than replace a file
            for line in answers[version]:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    # The first answer includes loading the model into memory, so speed is measured on the rest.
    warm = [t["seconds"] for t in timings[1:]] or [t["seconds"] for t in timings]
    meta = {
        "model": model,
        "route": "/api/generate raw (training template written by render_prompt)" if raw else "/api/chat",
        **seen,
        "prompt_sha256": prompt_sha256(),
        "data": source,
        "data_sha256": data_sha256,
        "pairs": len(pairs),
        "versions": list(VERSIONS),
        "started": started,
        "machine": machine(),
        "seconds_per_answer_mean": round(statistics.mean(warm), 3),
        "seconds_per_answer_median": round(statistics.median(warm), 3),
        "first_answer_seconds": timings[0]["seconds"] if timings else None,
        "total_seconds": round(sum(t["seconds"] for t in timings), 1),
        "timings": timings,
    }
    with open(paths[-1], "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2) + "\n")
    print(f"{paths[-1].name}: {len(timings)} answers, {meta['seconds_per_answer_mean']} s per answer "
          f"(median {meta['seconds_per_answer_median']} s, first {meta['first_answer_seconds']} s)")
    return meta


def compare(data, ours, theirs, state_path=REPO / "STATE.md", final=False):
    """Compare two answer sets (file prefixes, without .<version>.jsonl) on the same pairs, version by version."""
    pairs, _, _ = load_data(data, final, state_path)
    result = {}
    for version in VERSIONS:
        ours_metrics, ours_rows = score(pairs, load_jsonl(f"{ours}.{version}.jsonl"))
        theirs_metrics, theirs_rows = score(pairs, load_jsonl(f"{theirs}.{version}.jsonl"))
        both = list(zip(ours_rows, theirs_rows))
        same_label = [o["id"] for o, t in both if o["label"] is not None and o["label"] == t["label"]]
        non = [(o, t) for o, t in both if o["label"] == t["label"] == "non_contradiction"]
        result[version] = {
            "pairs": len(both),
            "label_agreement": len(same_label) / len(both) if both else None,
            "label_differs": [o["id"] for o, t in both if o["id"] not in same_label],
            "type_agreement": sum(o["type"] == t["type"] for o, t in non) / len(non) if non else None,
            "type_agreement_of": len(non),
            "same_raw_text": sum(o["raw"] == t["raw"] for o, t in both),
            "ours": ours_metrics,
            "theirs": theirs_metrics,
        }
    total = sum(r["pairs"] for r in result.values())
    result["all"] = {
        "pairs": total,
        "label_agreement": sum(r["label_agreement"] * r["pairs"] for r in result.values()) / total if total else None,
        "invalid_ours": sum(result[v]["ours"]["invalid"] for v in VERSIONS),
    }
    return result


def print_comparison(result):
    for version in VERSIONS:
        r = result[version]
        print(f"{version}: labels agree {r['label_agreement']:.3f} ({r['pairs'] - len(r['label_differs'])}/{r['pairs']}), "
              f"same text {r['same_raw_text']}/{r['pairs']}, differ at {', '.join(r['label_differs']) or 'none'}; "
              f"precision {r['ours']['precision']:.3f} vs {r['theirs']['precision']:.3f}, "
              f"recall {r['ours']['recall']:.3f} vs {r['theirs']['recall']:.3f}, invalid {r['ours']['invalid']}")
    print(f"all: labels agree {result['all']['label_agreement']:.3f} of {result['all']['pairs']}, "
          f"invalid answers {result['all']['invalid_ours']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("answer", help="answer every pair in all three versions through Ollama")
    run.add_argument("data")
    run.add_argument("out")
    run.add_argument("--raw", action="store_true", help="send the training template's text instead of chat messages")
    run.add_argument("--final", action="store_true", help="allow the one run on a frozen file")
    run.add_argument("--host", default=HOST)
    run.add_argument("--model", default=MODEL)
    check = commands.add_parser("compare", help="compare two answer sets on the same pairs")
    check.add_argument("data")
    check.add_argument("ours", help="answer files prefix, such as results/m1/gguf-v1/valid/m1-judge-v1")
    check.add_argument("theirs", help="answer files prefix, such as results/m1/trained-v1/valid/epoch-2")
    check.add_argument("--save", help="write the comparison to this JSON file (never replaces one)")
    check.add_argument("--final", action="store_true", help="the data is a frozen file whose answers already exist")
    args = parser.parse_args(argv)

    if args.command == "answer":
        answer(args.data, args.out, raw=args.raw, final=args.final, host=args.host, model=args.model)
    else:
        result = compare(args.data, args.ours, args.theirs, final=args.final)
        print_comparison(result)
        if args.save:
            with open(args.save, "x", encoding="utf-8") as f:
                f.write(json.dumps({"ours": args.ours, "theirs": args.theirs, **result}, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
