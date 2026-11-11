# Run from the session scratch folder that holds w1..w8 (writer files are not in git). Kept as a record of how the files were built.
"""Check the merged plan for repeated ideas: within itself, and against every pair in the non-frozen files of
data/m1 and data/m2. Uses the checker's own word handling (idea_words) and the TF-IDF cosine of
cross_pair_scores, on 'topic + clash' for plan lines and on both memories for old pairs."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')) + '/'

import sys, json, glob, math
from collections import Counter
sys.path.insert(0, ROOT + 'scripts')
import check_dataset as cd
R = ROOT
FROZEN = {R + 'data/m1/test.jsonl', R + 'data/m2/test-hard.jsonl'}
OLD = [p for p in sorted(glob.glob(R + 'data/m1/*.jsonl') + glob.glob(R + 'data/m2/*.jsonl')) if p not in FROZEN]
plan = {}
for w in range(1, 9):
    for line in open(f'w{w}/plan.jsonl', encoding='utf-8'):
        if line.strip():
            row = json.loads(line); row['writer'] = f'w{w}'; plan[row['plan_id']] = row
slots = {s['plan_id']: s for s in json.load(open('slots.json'))}
missing = set(slots) - set(plan); extra = set(plan) - set(slots)
print('plan lines', len(plan), 'missing', sorted(missing)[:10], 'extra', sorted(extra)[:10])
bad_writer = [p for p in plan if p in slots and slots[p]['writer'] != plan[p]['writer']]
print('wrong writer', bad_writer[:10])
norm = lambda s: ' '.join(s.lower().split())
tc = Counter(norm(r['topic']) for r in plan.values())
print('repeated topics', [t for t, c in tc.items() if c > 1])
def as_pair(pid, text):
    return {"id": pid, "person": "Zz Zz", "a": {"text": text}, "b": {"text": ""}}
rows = [(i, as_pair(pid, r['topic'] + ' ' + r['clash'])) for i, (pid, r) in enumerate(sorted(plan.items()), 1)]
old = []
seen_ids = set()
for path in OLD:
    ps = []
    for _, p in cd.read_pairs(path):
        key = (path.split('/')[-2], p['id'])
        if p['id'] in seen_ids and 'rl-pool' in path: continue
        ps.append(p)
    old.append((path, ps, False))
limit = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
found = cd.cross_pair_scores(rows, old, limit=limit)
oldpairs = {(path, p['id']): p for path, ps, _ in old for p in ps}
seen = set()
for s, line_no, pid, path, oid, _ in found:
    key = (pid, oid)
    if key in seen: continue
    seen.add(key)
    r = plan[pid]
    if path is None:
        o = plan[oid]; other = f"PLAN {oid} [{o['writer']} {slots[oid]['hard']}] {o['topic']} :: {o['clash']}"
    else:
        o = oldpairs[(path, oid)]; other = f"OLD {path.split('data/')[1]} {oid}: {o['a']['text']} | {o['b']['text']}"
    print(f"{s:.2f} {pid} [{r['writer']} {slots[pid]['hard']}] {r['topic']} :: {r['clash']}\n      {other}")
print('hits', len(seen))
