# Run from the session scratch folder that holds w1..w8 (writer files are not in git). Kept as a record of how the files were built.
"""Run the checker's own walls on all pairs written so far and list every problem (not only the first).
Name wall: check_names per pair per old file. Near copies: check_near_copies. Cross-pair: cross_pair_scores.
Frozen files' pairs are reported by id only."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')) + '/'

import sys, json, glob
sys.path.insert(0, ROOT + 'scripts')
import check_dataset as cd
R = ROOT
FROZEN = [R + 'data/m1/test.jsonl', R + 'data/m2/test-hard.jsonl']
OTHER = [p for p in sorted(glob.glob(R + 'data/m1/*.jsonl') + glob.glob(R + 'data/m2/*.jsonl')) if p not in FROZEN]
class Hit(Exception): pass
def boom(line_no, msg): raise Hit(msg)
cd.fail = boom
mine = []
for w in range(1, 9):
    for l in open(f'w{w}/pairs.jsonl', encoding='utf-8'):
        if l.strip(): p = json.loads(l); p['_w'] = f'w{w}'; mine.append(p)
rows = list(enumerate(mine, 1))
old = [(p, [x for _, x in cd.read_pairs(p)], p in FROZEN) for p in FROZEN + OTHER]
names_hits = []
for path, ps, frozen in old:
    for n, p in rows:
        try: cd.check_names([(n, p)], ps, path)
        except Hit as e: names_hits.append((p['id'], p['_w'], str(e).split(' belongs')[0]))
# names of my own people inside other pairs of mine (test vs valid wall and general)
own = {}
for p in mine:
    for t in p['person'].split(): own[t] = p['id']
import re
for p in mine:
    for txt in (p['a']['text'], p['b']['text'], p['reason']):
        for wd in re.findall(r"[^\W\d_]+", txt):
            if wd in own and own[wd] != p['id'] and wd[:1].isupper():
                names_hits.append((p['id'], p['_w'], f"name {wd!r} of {own[wd]}"))
near = []
for path, ps, frozen in old:
    for n, p in rows:
        try: cd.check_near_copies([(n, p)], ps, path)
        except Hit as e: near.append((p['id'], p['_w'], str(e).replace(R, '')))
print('NAME HITS', len(names_hits))
for h in sorted(set(names_hits)): print('  ', *h)
print('NEAR COPIES', len(near))
for h in near: print('  ', *h)
found = cd.cross_pair_scores(rows, old)
print('CROSS-PAIR over limit', len(found))
P = {p['id']: p for p in mine}
for s, n, pid, path, oid, frozen in found:
    me = P[pid]
    if path is None:
        o = P[oid]; other = f"{oid} [{o['_w']}] {o['a']['text']} | {o['b']['text']}"
    elif frozen:
        other = f"{oid} of {path.replace(R,'')} (frozen, text not shown)"
    else:
        o = next(x for x in dict((q, v) for q, v, _ in [(a, b, c) for a, b, c in old])[path] if x['id'] == oid)
        other = f"{oid} of {path.replace(R,'')}: {o['a']['text']} | {o['b']['text']}"
    print(f"  {s:.2f} {pid} [{me['_w']} {me['hard']}] {me['a']['text']} | {me['b']['text']}\n        -> {other}")
