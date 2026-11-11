# Run from the session scratch folder that holds w1..w8 (writer files are not in git). Kept as a record of how the files were built.
"""Merge the writers' pairs into data/m3/test-hard-2.jsonl and data/m3/valid-hard-2.jsonl.
Renames two people whose name words clash with ordinary words in other pairs. Orders each file one pair at a time
with the rule built in (no label more than 3 times in a row), Python random, seed 20261111; ids follow the order."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')) + '/'

import json, random, re
R = ROOT
plan = {json.loads(l)['plan_id']: json.loads(l) for l in open(R + 'data/m3/situation-plan.jsonl')}
RENAME = {"p161": ("Chumani Two Bulls", "Chumani Bettelyoun"), "p677": (None, "Faraz")}  # p677: first name only
pairs = []
for w in range(1, 9):
    for l in open(f'w{w}/pairs.jsonl', encoding='utf-8'):
        if l.strip(): pairs.append(json.loads(l))
assert len(pairs) == 800 and {p['id'] for p in pairs} == set(plan)
for p in pairs:
    if p['id'] == 'p161':
        assert p['person'] == 'Chumani Two Bulls'; p['person'] = 'Chumani Bettelyoun'
    if p['id'] == 'p677':
        old = p['person'].split()[0]; assert old == 'Danish'
        p['person'] = 'Faraz ' + ' '.join(p['person'].split()[1:])
        for k in ('a', 'b'):
            p[k]['text'] = re.sub(r'(?<!\w)Danish(?!\w)', 'Faraz', p[k]['text'])
        p['reason'] = re.sub(r'(?<!\w)Danish(?!\w)', 'Faraz', p['reason'])
KEYS = ["id", "person", "a", "b", "label", "type", "hard", "reason", "source", "label_by"]
def order(items, seed):
    rng = random.Random(seed)
    for attempt in range(1000):
        left = items[:]; rng.shuffle(left); out = []
        while left:
            run = [x['label'] for x in out[-3:]]
            blocked = run[0] if len(run) == 3 and len(set(run)) == 1 else None
            choices = [x for x in left if x['label'] != blocked]
            if not choices: break
            pick = rng.choice(choices); left.remove(pick); out.append(pick)
        if not left: return out
    raise SystemExit('could not order')
for name, setname, prefix in (('test-hard-2.jsonl', 'test', 'h2'), ('valid-hard-2.jsonl', 'valid', 'v2')):
    items = sorted([p for p in pairs if plan[p['id']]['set'] == setname], key=lambda p: p['id'])
    out = order(items, 20261111)
    with open(R + 'data/m3/' + name, 'w', encoding='utf-8') as f:
        for i, p in enumerate(out, 1):
            row = {k: p[k] for k in KEYS}; row['id'] = f'{prefix}-{i:03d}'
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with open(f'idmap-{setname}.json', 'w') as f:
        json.dump({f'{prefix}-{i:03d}': p['id'] for i, p in enumerate(out, 1)}, f)
    print(name, len(out))
