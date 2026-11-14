"""M3 Step 2: the baseline report for v1's and v2's adapters on test set 2 and validation set 2.

It reads saved answer files only and runs no model. The answers are written by
`scripts/eval_trained.py --model v1|v2 --test <set> --out results/m3/baseline/<model>/<set>`.
Each set is loaded through the frozen check (`judge.frozen.load_frozen`) and scored with
`judge.score.score` and `judge.score.score_by_trap`, the same scorer M2 used; nothing here changes
how an answer is scored. Answers that break v1's exact format are those `judge.reward.read_answer`
rejects.

The M2 numbers it compares with come from `results/m2/v2/compare.json` (counts saved in M2 Step 6),
so the M2 test file is never opened.

What M3's bar will be (a record only; the rule itself is coded in Step 6): against v1's adapter on
test set 2, worst-case recall at least v1's + 0.03 (9 of 300 contradictions), and worst-case precision
no more than 0.01 below v1's. "Worst case" is the lowest of the three versions. Every rate is printed
as counts (caught of said, caught of real), never as a reduced fraction (lesson A6).

Run from the repo folder, after the four runs:
    uv run python scripts/m3_baseline_report.py
    uv run python scripts/m3_baseline_report.py --save results/m3/baseline/report
(--save writes <name>.txt and <name>.json and never replaces a file.)
"""

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

from judge.frozen import FrozenFileError, load_frozen
from judge.reward import read_answer
from judge.score import VERSIONS, load_jsonl, score, score_by_trap

SETS = {"test": "data/m3/test-hard-2.jsonl", "valid": "data/m3/valid-hard-2.jsonl"}
ROOT = "results/m3/baseline"
NAMES = {"v1": "qwen3-4b-trained-v1", "v2": "qwen3-4b-grpo-v2"}  # the names eval_trained.py gives
M2_COMPARE = "results/m2/v2/compare.json"
RECALL_GAIN = Fraction(3, 100)
PRECISION_DROP = Fraction(1, 100)


def counts(pairs, answers):
    """Exact counts for one version of one set (as in scripts/compare_v2.py)."""
    _, rows = score(pairs, answers)
    return {
        "pairs": len(rows),
        "caught": sum(r["label"] == "contradiction" and r["gold_label"] == "contradiction" for r in rows),
        "said": sum(r["label"] == "contradiction" for r in rows),
        "real": sum(r["gold_label"] == "contradiction" for r in rows),
        "right": sum(r["label"] == r["gold_label"] for r in rows),
        "unclear": sum(r["label"] == "unclear" for r in rows),
        "invalid": sum(bool(r["problem"]) for r in rows),
        "format_breaks": sum(read_answer(r["raw"])[0] is None for r in rows),
    }


def precision(c):
    return Fraction(c["caught"], c["said"]) if c["said"] else Fraction(0)


def recall(c):
    return Fraction(c["caught"], c["real"]) if c["real"] else Fraction(0)


def rate(top, bottom):
    """Counts as "top/bottom (0.123)", never reduced."""
    return f"{top}/{bottom} ({top / bottom:.3f})" if bottom else f"{top}/{bottom} (-)"


def worst(by_version, measure):
    """The version with the lowest value of `measure`, and that value."""
    version = min(VERSIONS, key=lambda v: measure(by_version[v]))
    return version, measure(by_version[version])


def bar(v1_test):
    """What M3's bar will be, from v1's counts on test set 2, as pair counts."""
    r_version, r_worst = worst(v1_test, recall)
    p_version, p_worst = worst(v1_test, precision)
    real = v1_test[r_version]["real"]
    need_recall = r_worst + RECALL_GAIN
    need_caught = math.ceil(need_recall * real)
    need_precision = p_worst - PRECISION_DROP
    # The most "contradiction" answers allowed at exactly need_caught caught: caught / said >= need_precision.
    max_said = math.floor(need_caught / need_precision)
    return {
        "recall": {"v1_version": r_version, "v1_caught": v1_test[r_version]["caught"], "real": real,
                   "need_caught": need_caught, "more_than_v1": need_caught - v1_test[r_version]["caught"]},
        "precision": {"v1_version": p_version, "v1_caught": v1_test[p_version]["caught"],
                      "v1_said": v1_test[p_version]["said"],
                      "need": f"{need_precision.numerator}/{need_precision.denominator}",
                      "need_decimal": float(need_precision),
                      "at_need_caught_max_said": max_said, "at_need_caught_max_false_alarms": max_said - need_caught},
    }


def traps(pairs, answers):
    """Per trap: pairs, then right labels and `unclear` answers for each model and version."""
    table = {}
    for model, by_version in answers.items():
        for version in VERSIONS:
            for trap, m in score_by_trap(pairs, by_version[version]).items():
                row = table.setdefault(trap, {"pairs": m["pairs"]})
                row[f"{model} {version}"] = round(m["accuracy"] * m["pairs"])
                row[f"{model} {version} unclear"] = m["unclear"]
    return table


def paired(pairs, a1, a2):
    """Per version: pairs v1 gets wrong and v2 right, and the reverse (all pairs, and contradictions only)."""
    out = {}
    for version in VERSIONS:
        r1 = {r["id"]: r for r in score(pairs, a1[version])[1]}
        r2 = {r["id"]: r for r in score(pairs, a2[version])[1]}
        right1 = {i for i, r in r1.items() if r["label"] == r["gold_label"]}
        right2 = {i for i, r in r2.items() if r["label"] == r["gold_label"]}
        real = {i for i, r in r1.items() if r["gold_label"] == "contradiction"}
        out[version] = {
            "v1_wrong_v2_right": sorted(right2 - right1),
            "v1_right_v2_wrong": sorted(right1 - right2),
            "contradictions_v1_misses_v2_catches": sorted((right2 - right1) & real),
            "contradictions_v2_misses_v1_catches": sorted((right1 - right2) & real),
        }
    return out


def shares(trap_table, models=("v1", "v2")):
    """Per trap and model: right labels over all three versions, as (right, of)."""
    return {trap: {m: (sum(row[f"{m} {v}"] for v in VERSIONS), 3 * row["pairs"]) for m in models}
            for trap, row in trap_table.items()}


def load_answers(folder, name):
    return {version: load_jsonl(Path(folder) / f"{name}.{version}.jsonl") for version in VERSIONS}


def build(sets=SETS, root=ROOT, m2_compare=M2_COMPARE, state_path=None):
    """Load both sets through the frozen check, score every saved answer file, and gather the report."""
    pairs = {name: load_frozen(path, state_path) for name, path in sets.items()}
    answers = {model: {name: load_answers(Path(root) / model / name, NAMES[model]) for name in sets}
               for model in NAMES}
    c = {model: {name: {v: counts(pairs[name], answers[model][name][v]) for v in VERSIONS} for name in sets}
         for model in NAMES}
    test_traps = traps(pairs["test"], {m: answers[m]["test"] for m in NAMES})
    m2 = json.loads(Path(m2_compare).read_text(encoding="utf-8"))
    return {
        "counts": c,
        "bar": bar(c["v1"]["test"]),
        "traps": test_traps,
        "paired": paired(pairs["test"], answers["v1"]["test"], answers["v2"]["test"]),
        "m2": {"counts": {m: m2["counts"][m]["hard"] for m in NAMES}, "traps": m2["traps"]},
        "shares": {"m3": shares(test_traps), "m2": shares(m2["traps"])},
    }


def report(result):
    """Plain-text report, counts first."""
    lines = []
    for name in result["counts"]["v1"]:
        lines.append(f"{name} set 2: model version  caught/said (precision)  caught/real (recall)  "
                     "unclear invalid format_breaks")
        for model in result["counts"]:
            for version in VERSIONS:
                k = result["counts"][model][name][version]
                lines.append(f"  {model} {version:9} {rate(k['caught'], k['said']):>17} {rate(k['caught'], k['real']):>17}"
                             f"  {k['unclear']} {k['invalid']} {k['format_breaks']}")
        lines.append("")
    b = result["bar"]
    r, p = b["recall"], b["precision"]
    lines.append("M3 bar (record only; coded in Step 6), from v1 on test set 2:")
    lines.append(f"  worst-case recall: v1 {r['v1_caught']}/{r['real']} ({r['v1_version']}); "
                 f"need at least {r['need_caught']}/{r['real']} ({r['more_than_v1']} more caught)")
    lines.append(f"  worst-case precision: v1 {p['v1_caught']}/{p['v1_said']} ({p['v1_version']}); "
                 f"need at least {p['need']} ({p['need_decimal']:.4f}); at {r['need_caught']} caught that allows "
                 f"at most {p['at_need_caught_max_said']} said, {p['at_need_caught_max_false_alarms']} false alarms")
    lines.append("")
    lines.append("test set 2 per trap, right labels (v1 as_is swapped no_dates | v2 as_is swapped no_dates)"
                 " [unclear, all 3 versions: v1 | v2]:")
    for trap, row in result["traps"].items():
        v1 = " ".join(str(row[f"v1 {x}"]) for x in VERSIONS)
        v2 = " ".join(str(row[f"v2 {x}"]) for x in VERSIONS)
        u1 = sum(row[f"v1 {x} unclear"] for x in VERSIONS)
        u2 = sum(row[f"v2 {x} unclear"] for x in VERSIONS)
        lines.append(f"  {trap}: of {row['pairs']}: {v1} | {v2} [{u1} | {u2}]")
    lines.append("")
    lines.append("test set 2 pair by pair: v1 wrong and v2 right / v1 right and v2 wrong "
                 "(contradictions only in brackets):")
    for version, d in result["paired"].items():
        lines.append(f"  {version:9} {len(d['v1_wrong_v2_right'])} / {len(d['v1_right_v2_wrong'])} "
                     f"[{len(d['contradictions_v1_misses_v2_catches'])} / {len(d['contradictions_v2_misses_v1_catches'])}]")
    lines.append("")
    lines.append("per-trap share right over all 3 versions, M2 hard set -> M3 test set 2 (v1; v2):")
    for trap in result["shares"]["m3"]:
        new, old = result["shares"]["m3"][trap], result["shares"]["m2"].get(trap)
        cells = []
        for m in ("v1", "v2"):
            o = f"{old[m][0]}/{old[m][1]} ({old[m][0] / old[m][1]:.3f})" if old else "-"
            cells.append(f"{o} -> {new[m][0]}/{new[m][1]} ({new[m][0] / new[m][1]:.3f})")
        lines.append(f"  {trap}: {'; '.join(cells)}")
    return "\n".join(lines)


def main(argv):
    save = None
    if argv[1:2] == ["--save"] and len(argv) == 3:
        save = Path(argv[2])
    elif len(argv) != 1:
        sys.exit("usage: python scripts/m3_baseline_report.py [--save <path without extension>]")
    try:
        result = build()
    except FrozenFileError as error:
        sys.exit(f"STOP: {error}")
    text = report(result)
    print(text)
    if save:
        with open(save.with_suffix(".txt"), "x", encoding="utf-8") as f:  # "x": never replace a saved report
            f.write(text + "\n")
        with open(save.with_suffix(".json"), "x", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv)
