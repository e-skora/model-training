# Run from the session scratch folder that holds w1..w8 (writer files are not in git). Kept as a record of how the files were built.
"""Screen candidate names: well-formed, every name word unique across all candidates kept, and no name word of any
person in data/m1 or data/m2 (checked with the checker's own name wall, check_names)."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')) + '/'

import sys, json, re, glob
sys.path.insert(0, ROOT + 'scripts')
import check_dataset as cd
R = ROOT
OLD = sorted(glob.glob(R + 'data/m1/*.jsonl') + glob.glob(R + 'data/m2/*.jsonl'))
old = [(p, [pair for _, pair in cd.read_pairs(p)]) for p in OLD]
class Hit(Exception): pass
def boom(line_no, message): raise Hit(message)
cd.fail = boom
def blocked(name):
    first = name.split()[0]
    fake = [(1, {"person": name, "a": {"text": f"{first} reads."}, "b": {"text": f"{first} reads."}, "reason": "x"})]
    for path, pairs in old:
        try: cd.check_names(fake, pairs, path)
        except Hit: return True
    return False
cands = []
for w in range(1, 9):
    for line in open(f'w{w}/names.txt', encoding='utf-8'):
        if '|' not in line: continue
        name, culture = [x.strip() for x in line.split('|', 1)]
        cands.append((w, name, culture))
seen, kept, why = set(), [], {}
for w, name, culture in cands:
    parts = name.split()
    if len(parts) < 2 or not all(p[:1].isupper() for p in parts) or re.search(r'mara|lindqvist', name, re.I):
        why[name] = 'form'; continue
    if any(p in seen for p in parts):
        why[name] = 'repeat'; continue
    if blocked(name):
        why[name] = 'wall'; continue
    seen.update(parts); kept.append({"writer": f"w{w}", "person": name, "culture": culture})
from collections import Counter
print('candidates', len(cands), 'kept', len(kept), Counter(why.values()))
print(Counter(k['writer'] for k in kept))
json.dump(kept, open('names_kept.json', 'w'), ensure_ascii=False, indent=0)
