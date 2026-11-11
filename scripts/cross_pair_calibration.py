"""Rerun the test that chose the cross-pair check's limit (data/m3/README.md).

Scores every pair of M2's training and validation pairs against every other one with
`check_dataset.cross_pair_scores`, then prints, for a limit, how many links are flagged and how many of
the repeated ideas found by hand (LESSONS 2026-10-27) are caught. Reads only those two M2 files.

    uv run python scripts/cross_pair_calibration.py            # the checker's limit
    uv run python scripts/cross_pair_calibration.py 0.40 0.48  # other limits
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FILES = ("data/m2/train-hard.jsonl", "data/m2/valid-hard.jsonl")
# Repeated ideas still in the final M2 files, found by searching for the ideas LESSONS 2026-10-27 names.
HAND_FOUND = {
    "vegetarian eats meat": ("va2-078", "tr2-0180", "tr2-0385", "tr2-0850", "tr2-0972"),
    "never owned a car, drives one": ("tr2-0448", "tr2-0473", "tr2-0813"),
    "only child, has a brother or sister": ("va2-013", "va2-068", "tr2-0420", "tr2-0530", "tr2-0891"),
    "bakery nights, works days": ("va2-079", "tr2-0525"),
}


def main():
    pairs = [json.loads(line) for name in FILES for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = list(enumerate(pairs, start=1))
    scores = {frozenset((pairs[line_no - 1]["id"], other)): s
              for s, line_no, _, _, other, _ in check_dataset.cross_pair_scores(rows, [], limit=0.0)}
    limits = [float(x) for x in sys.argv[1:]] or [check_dataset.CROSS_PAIR_LIMIT]
    for limit in limits:
        flagged = [link for link, s in scores.items() if s > limit]
        involved = {pair_id for link in flagged for pair_id in link}
        print(f"limit {limit}: {len(flagged)} links flagged, {len(involved)} of {len(pairs)} pairs involved")
        for idea, ids in HAND_FOUND.items():
            links = [frozenset(link) for link in itertools.combinations(ids, 2)]
            caught = [link for link in links if scores.get(link, 0.0) > limit]
            members = {pair_id for link in caught for pair_id in link}
            best = max(scores.get(link, 0.0) for link in links)
            print(f"  {idea}: {len(caught)} of {len(links)} links, {len(members)} of {len(ids)} pairs, best score {best:.2f}")


if __name__ == "__main__":
    main()
