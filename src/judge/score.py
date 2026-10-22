"""Score a file of model answers against the test pairs.

Run from the repo folder:
    uv run python -m judge.score data/m1/test.jsonl results/m1/baseline/qwen3-4b.jsonl

The answers file is JSONL with one line per pair: {"id": ..., "raw": "<model text>"}.
"""

import json
import re
import sys

from judge.prompt import LABELS

# One markdown code fence around the whole answer, such as ```json ... ```.
FENCE = re.compile(r"\A```[a-zA-Z]*\s*\n(.*?)\n?```\Z", re.DOTALL)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_answer(raw):
    """Read one model answer. Returns (answer, problem, fenced).

    `answer` is a dict with a known label, or None when the answer is
    invalid; then `problem` says why. `fenced` is True when the JSON was
    wrapped in a markdown code fence, which is accepted but counted.
    """
    if not isinstance(raw, str):
        return None, "no answer", False
    text = raw.strip()
    match = FENCE.match(text)
    fenced = match is not None
    if fenced:
        text = match.group(1).strip()
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        return None, "not valid JSON", fenced
    if not isinstance(answer, dict):
        return None, "not valid JSON", fenced
    if answer.get("label") not in LABELS:
        return None, "unknown label", fenced
    return answer, None, fenced


def ratio(top, bottom):
    return top / bottom if bottom else None


def score(pairs, answers):
    """Compare answers with the right labels. Returns the metrics and one row per pair.

    Invalid or missing answers count as wrong: never a correct label, never
    a caught contradiction.
    """
    by_id = {answer["id"]: answer.get("raw") for answer in answers}

    rows = []
    for pair in pairs:
        answer, problem, fenced = parse_answer(by_id.get(pair["id"]))
        rows.append({
            "id": pair["id"],
            "gold_label": pair["label"],
            "gold_type": pair["type"],
            "label": answer["label"] if answer else None,
            "type": answer.get("type") if answer else None,
            "reason": answer.get("reason") if answer else None,
            "problem": problem,
            "fenced": fenced,
            "raw": by_id.get(pair["id"]),
        })

    said_contradiction = [r for r in rows if r["label"] == "contradiction"]
    real_contradiction = [r for r in rows if r["gold_label"] == "contradiction"]
    caught = [r for r in real_contradiction if r["label"] == "contradiction"]
    label_right = [r for r in rows if r["label"] == r["gold_label"]]
    # Type accuracy: only non-contradictions the model labelled right.
    right_non = [r for r in label_right if r["gold_label"] == "non_contradiction"]
    type_right = [r for r in right_non if r["type"] == r["gold_type"]]
    invalid = [r for r in rows if r["problem"]]

    metrics = {
        "pairs": len(rows),
        "precision": ratio(len(caught), len(said_contradiction)),
        "recall": ratio(len(caught), len(real_contradiction)),
        "accuracy": ratio(len(label_right), len(rows)),
        "type_accuracy": ratio(len(type_right), len(right_non)),
        "type_accuracy_of": len(right_non),
        "invalid": len(invalid),
        "invalid_not_json": sum(1 for r in invalid if r["problem"] in ("not valid JSON", "no answer")),
        "invalid_unknown_label": sum(1 for r in invalid if r["problem"] == "unknown label"),
        "fenced": sum(1 for r in rows if r["fenced"]),
    }
    return metrics, rows


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: python -m judge.score <test.jsonl> <answers.jsonl>")
    metrics, _ = score(load_jsonl(argv[1]), load_jsonl(argv[2]))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(sys.argv)
