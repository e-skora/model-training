"""M2 Step 6: does v2 beat v1 on the two frozen test sets? The rule is D-060, written here before any test score.

It reads saved answer files only and runs no model. v1's answers are the ones saved in M1 Step 7 and
M2 Step 2 (v1 is never run again); v2's are written by `scripts/eval_trained.py --model v2`.

The rule (D-041, D-060), with exact fractions as in `choose_checkpoint`. "Worst case" is the lowest of
the three versions (as written, swapped, no dates).
1. Hard set: v2's worst-case recall is at least v1's + 0.03, and v2's worst-case precision is at least
   v1's - 0.01.
2. M1 set: in each version, v2's precision and recall are each at least v1's - 0.02 in that version.
3. v2 has 0 invalid answers (the scorer's count, D-024) in every version of both sets.
Answers that break v1's exact format (`judge.reward.read_answer` rejects them) are reported too; they
are not part of the rule.

Precision when a model never says "contradiction" counts as 0 (the strict reading, as in the choice).

Run from the repo folder, after both v2 runs:
    uv run python scripts/compare_v2.py
    uv run python scripts/compare_v2.py --save results/m2/v2/compare.json
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

from judge.frozen import FrozenFileError, load_frozen
from judge.reward import read_answer
from judge.score import VERSIONS, load_jsonl, score, score_by_trap

SETS = {"hard": "data/m2/test-hard.jsonl", "m1": "data/m1/test.jsonl"}
# (folder, file name before ".<version>.jsonl") for each model and set.
V1 = {"hard": ("results/m2/baseline/trained-v1", "qwen3-4b-trained-v1"),
      "m1": ("results/m1/trained-v1", "qwen3-4b-trained-v1")}
V2 = {"hard": ("results/m2/v2/hard", "qwen3-4b-grpo-v2"),
      "m1": ("results/m2/v2/m1", "qwen3-4b-grpo-v2")}

HARD_RECALL_GAIN = Fraction(3, 100)
HARD_PRECISION_DROP = Fraction(1, 100)
M1_DROP = Fraction(2, 100)


def counts(pairs, answers):
    """Exact counts for one version of one set."""
    _, rows = score(pairs, answers)
    return {
        "pairs": len(rows),
        "caught": sum(r["label"] == "contradiction" and r["gold_label"] == "contradiction" for r in rows),
        "said": sum(r["label"] == "contradiction" for r in rows),
        "real": sum(r["gold_label"] == "contradiction" for r in rows),
        "unclear": sum(r["label"] == "unclear" for r in rows),
        "invalid": sum(bool(r["problem"]) for r in rows),
        "format_errors": sum(bool(r["format_problem"]) for r in rows),
        "style_a_breaks": sum(read_answer(r["raw"])[0] is None for r in rows),
    }


def precision(c):
    return Fraction(c["caught"], c["said"]) if c["said"] else Fraction(0)


def recall(c):
    return Fraction(c["caught"], c["real"]) if c["real"] else Fraction(0)


def worst(by_version, rate):
    return min(rate(by_version[v]) for v in VERSIONS)


def show(x):
    """A fraction as "a/b (0.123)"."""
    return f"{x.numerator}/{x.denominator} ({float(x):.3f})"


def item_1(v1, v2):
    """Hard set: worst-case recall up by at least 0.03, worst-case precision down by no more than 0.01."""
    need_r = worst(v1, recall) + HARD_RECALL_GAIN
    need_p = worst(v1, precision) - HARD_PRECISION_DROP
    got_r, got_p = worst(v2, recall), worst(v2, precision)
    return {
        "passed": got_r >= need_r and got_p >= need_p,
        "checks": [
            {"what": "hard worst-case recall", "v1": show(worst(v1, recall)), "need": show(need_r),
             "v2": show(got_r), "passed": got_r >= need_r},
            {"what": "hard worst-case precision", "v1": show(worst(v1, precision)), "need": show(need_p),
             "v2": show(got_p), "passed": got_p >= need_p},
        ],
    }


def item_2(v1, v2):
    """M1 set: in each version, precision and recall each at least v1's - 0.02 in that version."""
    checks = []
    for version in VERSIONS:
        for name, rate in (("precision", precision), ("recall", recall)):
            need, got = rate(v1[version]) - M1_DROP, rate(v2[version])
            checks.append({"what": f"M1 {version} {name}", "v1": show(rate(v1[version])), "need": show(need),
                           "v2": show(got), "passed": got >= need})
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def item_3(v2_by_set):
    """0 invalid answers in every version of both sets; v1-format breaks are reported, not judged."""
    checks = [{"what": f"{name} {version} invalid", "v2": c[version]["invalid"],
               "style_a_breaks": c[version]["style_a_breaks"], "passed": c[version]["invalid"] == 0}
              for name, c in v2_by_set.items() for version in VERSIONS]
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def judge(v1, v2):
    """v1 and v2 map "hard" and "m1" to {version: counts}. Returns the verdict for each D-041 item."""
    items = {"1": item_1(v1["hard"], v2["hard"]), "2": item_2(v1["m1"], v2["m1"]), "3": item_3(v2)}
    return {"items": items, "passed": all(i["passed"] for i in items.values())}


def load_answers(folder, name):
    return {version: load_jsonl(Path(folder) / f"{name}.{version}.jsonl") for version in VERSIONS}


def traps(pairs, v1_answers, v2_answers):
    """Per trap of the hard set: pairs, and right labels for v1 and v2 in each version."""
    table = {}
    for model, answers in (("v1", v1_answers), ("v2", v2_answers)):
        for version in VERSIONS:
            for trap, m in score_by_trap(pairs, answers[version]).items():
                row = table.setdefault(trap, {"pairs": m["pairs"]})
                row[f"{model} {version}"] = round(m["accuracy"] * m["pairs"])
    return table


def compare(sets=SETS, v1_files=V1, v2_files=V2, state_path=None):
    """Load both test sets through the frozen check, score every saved answer file, apply the rule."""
    pairs = {name: load_frozen(path, state_path) for name, path in sets.items()}
    answers = {model: {name: load_answers(*files[name]) for name in sets}
               for model, files in (("v1", v1_files), ("v2", v2_files))}
    c = {model: {name: {v: counts(pairs[name], answers[model][name][v]) for v in VERSIONS} for name in sets}
         for model in answers}
    verdict = judge(c["v1"], c["v2"])
    return {"counts": c, "verdict": verdict, "traps": traps(pairs["hard"], answers["v1"]["hard"],
                                                            answers["v2"]["hard"])}


def report(result):
    """Plain-text report: PASS or MISS for each item and overall, then the counts."""
    word = {True: "PASS", False: "MISS"}
    v = result["verdict"]
    lines = [f"Overall: {word[v['passed']]}"]
    for number, item in v["items"].items():
        lines.append(f"Item {number}: {word[item['passed']]}")
        for check in item["checks"]:
            detail = ", ".join(f"{k} {check[k]}" for k in check if k not in ("what", "passed"))
            lines.append(f"  {word[check['passed']]}  {check['what']}: {detail}")
    for name in result["counts"]["v1"]:
        lines.append(f"\n{name} set: model version caught/said/real precision recall unclear invalid "
                     "format_errors style_a_breaks")
        for model in ("v1", "v2"):
            for version in VERSIONS:
                k = result["counts"][model][name][version]
                lines.append(f"  {model} {version:9} {k['caught']}/{k['said']}/{k['real']} "
                             f"{float(precision(k)):.3f} {float(recall(k)):.3f} {k['unclear']} {k['invalid']} "
                             f"{k['format_errors']} {k['style_a_breaks']}")
    lines.append("\nhard set per trap, right labels (v1 as_is swapped no_dates | v2 as_is swapped no_dates):")
    for trap, row in result["traps"].items():
        v1 = " ".join(str(row[f"v1 {x}"]) for x in VERSIONS)
        v2 = " ".join(str(row[f"v2 {x}"]) for x in VERSIONS)
        lines.append(f"  {trap}: of {row['pairs']}: {v1} | {v2}")
    return "\n".join(lines)


def main(argv):
    save = None
    if argv[1:2] == ["--save"] and len(argv) == 3:
        save = Path(argv[2])
    elif len(argv) != 1:
        sys.exit("usage: python scripts/compare_v2.py [--save <file.json>]")
    try:
        result = compare()
    except FrozenFileError as error:
        sys.exit(f"STOP: {error}")
    print(report(result))
    if save:
        with open(save, "x", encoding="utf-8") as f:  # "x": never replace a saved comparison
            f.write(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main(sys.argv)
