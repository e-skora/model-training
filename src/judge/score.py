"""Score a file of model answers against the test pairs.

Run from the repo folder:
    uv run python -m judge.score data/m1/test.jsonl results/m1/baseline/qwen3-4b.jsonl
Score the three versions of the test set side by side (as is, swapped, no dates):
    uv run python -m judge.score data/m1/test.jsonl as_is.jsonl swapped.jsonl no_dates.jsonl

The test file must match its "Frozen files" entry in STATE.md, or scoring stops.
An answers file is JSONL with one line per pair: {"id": ..., "raw": "<model text>"}.
"""

import copy
import json
import re
import sys

from judge.frozen import FrozenFileError, load_frozen
from judge.prompt import LABELS, TYPES

# The frozen test set as is, and two versions built from it in memory (D-026).
# The right answers stay the same: order means nothing, and an update must
# be clear from its wording when the dates are gone.
VERSIONS = ("as_is", "swapped", "no_dates")

# One markdown code fence around the whole answer, such as ```json ... ```.
FENCE = re.compile(r"\A```[a-zA-Z]*\s*\n(.*?)\n?```\Z", re.DOTALL)


class RepeatedField(ValueError):
    pass


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reject_constant(name):
    # Python's json reads NaN, Infinity and -Infinity, which are not JSON.
    raise ValueError(f"{name} is not valid JSON")


def no_repeated_fields(items):
    names = [name for name, _ in items]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise RepeatedField(f"repeated field {repeated}")
    return dict(items)


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
        answer = json.loads(text, parse_constant=reject_constant, object_pairs_hook=no_repeated_fields)
    except RepeatedField:
        return None, "repeated field", fenced
    except ValueError:  # includes json.JSONDecodeError
        return None, "not valid JSON", fenced
    if not isinstance(answer, dict):
        return None, "not valid JSON", fenced
    if answer.get("label") not in LABELS:
        return None, "unknown label", fenced
    return answer, None, fenced


def derive(pairs, version):
    """Return the pairs as they are shown in one version of the test set.

    "swapped" exchanges memories a and b; "no_dates" removes both dates.
    The pairs are copied, so the loaded test set itself never changes.
    """
    if version not in VERSIONS:
        raise ValueError(f"unknown version {version!r}; choose from {VERSIONS}")
    derived = copy.deepcopy(pairs)
    for pair in derived:
        if version == "swapped":
            pair["a"], pair["b"] = pair["b"], pair["a"]
        elif version == "no_dates":
            for which in ("a", "b"):
                pair[which]["written_at"] = None
    return derived


def format_problem(answer):
    """For an answer with a known label, say which required field breaks the prompt's format, or None.

    The prompt asks for "type": JSON null for a contradiction or unclear, and
    one of the three types otherwise, and "reason": a short sentence (non-empty text).
    """
    if "type" not in answer:
        return "no type"
    if answer["label"] in ("contradiction", "unclear") and answer["type"] is not None:
        return f"type not null for {answer['label']}"
    if answer["label"] == "non_contradiction" and answer["type"] not in TYPES:
        return "unknown type"
    reason = answer.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return "no reason text"
    return None


def check_ids(pairs, answers):
    """Stop on answers that repeat a pair or name a pair not in the test set."""
    seen, repeated = set(), set()
    for answer in answers:
        if answer["id"] in seen:
            repeated.add(answer["id"])
        seen.add(answer["id"])
    if repeated:
        raise ValueError(f"more than one answer for pair(s) {sorted(repeated)}")
    unknown = seen - {pair["id"] for pair in pairs}
    if unknown:
        raise ValueError(f"answers for pair(s) not in the test set: {sorted(unknown)}")


def ratio(top, bottom):
    return top / bottom if bottom else None


def score(pairs, answers):
    """Compare answers with the right labels. Returns the metrics and one row per pair.

    Precision and recall are the standard ones, with "contradiction" as the
    "yes" answer (D-024). An invalid or missing answer is neither yes nor no:
    it lowers recall when the right label is contradiction, it lowers
    accuracy, and it never enters precision. So `invalid` is always reported
    next to them. An "unclear" answer is valid but is not a yes either
    (D-025): it never enters precision, it lowers recall when the right
    label is contradiction, and it is counted in `unclear`. `format_errors`
    counts answers with a known label that break the type or reason fields;
    they do not change label accuracy.
    """
    check_ids(pairs, answers)
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
            "format_problem": format_problem(answer) if answer else None,
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
        "unclear": sum(1 for r in rows if r["label"] == "unclear"),
        "invalid": len(invalid),
        "invalid_not_json": sum(1 for r in invalid if r["problem"] in ("not valid JSON", "no answer")),
        "invalid_repeated_field": sum(1 for r in invalid if r["problem"] == "repeated field"),
        "invalid_unknown_label": sum(1 for r in invalid if r["problem"] == "unknown label"),
        "format_errors": sum(1 for r in rows if r["format_problem"]),
        "fenced": sum(1 for r in rows if r["fenced"]),
    }
    return metrics, rows


def score_versions(pairs, answers_by_version):
    """Score each version's answers against the same right labels. Returns {version: metrics}."""
    return {version: score(pairs, answers)[0] for version, answers in answers_by_version.items()}


def side_by_side(metrics_by_version):
    """A plain-text table with one column per version."""
    versions = list(metrics_by_version)
    names = list(next(iter(metrics_by_version.values())))
    lines = ["metric".ljust(24) + "".join(v.rjust(12) for v in versions)]
    for name in names:
        cells = []
        for version in versions:
            value = metrics_by_version[version][name]
            cells.append(("-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)).rjust(12))
        lines.append(name.ljust(24) + "".join(cells))
    return "\n".join(lines)


def main(argv):
    if len(argv) not in (3, 5):
        sys.exit("usage: python -m judge.score <test.jsonl> <answers.jsonl>\n"
                 "   or: python -m judge.score <test.jsonl> <as_is.jsonl> <swapped.jsonl> <no_dates.jsonl>")
    try:
        pairs = load_frozen(argv[1])
    except FrozenFileError as error:
        sys.exit(f"STOP: {error}")
    metrics = {}
    for version, path in zip(VERSIONS, argv[2:]):
        try:
            metrics[version] = score_versions(pairs, {version: load_jsonl(path)})[version]
        except ValueError as error:
            sys.exit(f"STOP: {path}: {error}")
    if len(metrics) == 1:
        print(json.dumps(metrics["as_is"], indent=2))
    else:
        print(side_by_side(metrics))


if __name__ == "__main__":
    main(sys.argv)
