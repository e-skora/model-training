# Run from the session scratch folder that holds w1..w8 (writer files are not in git). Kept as a record of how the files were built.
"""Build 800+ names: screen every name word of every candidate (names.txt, names2.txt) with the checker's name wall,
then pair free first names with free surnames from the same writer's list and culture where possible. Seed 20261111."""
import random, json, re
import screen_names as s   # loads the walls (prints its own summary first)
rng = random.Random(20261111)
firsts, surs = {}, {}   # writer -> list of (word, culture)
for w in range(1, 9):
    for fn in ('names.txt', 'names2.txt'):
        for line in open(f'w{w}/{fn}', encoding='utf-8'):
            if '|' not in line: continue
            name, culture = [x.strip() for x in line.split('|', 1)]
            parts = name.split()
            if len(parts) < 2 or not all(p[:1].isupper() for p in parts): continue
            firsts.setdefault(w, []).append((parts[0], culture))
            surs.setdefault(w, []).append((' '.join(parts[1:]), culture))
cache = {}
def free(word):
    if word not in cache: cache[word] = not re.search(r'mara|lindqvist', word, re.I) and not s.blocked(word + ' ' + 'Qqqzx' if ' ' not in word else 'Qqqzx ' + word)
    return cache[word]
used, out = set(), []
for w in range(1, 9):
    fs = [(f, c) for f, c in firsts[w] if free(f)]
    ss = [(x, c) for x, c in surs[w] if all(free(p) for p in x.split())]
    rng.shuffle(fs); rng.shuffle(ss)
    for f, c in fs:
        if f in used: continue
        # same culture surname first, else any surname from this writer's region
        pick = next((t for t in ss if t[1] == c and not set(t[0].split()) & used), None) or \
               next((t for t in ss if not set(t[0].split()) & used), None)
        if not pick: break
        ss.remove(pick); used.add(f); used.update(pick[0].split())
        out.append({"writer": f"w{w}", "person": f + ' ' + pick[0], "culture": c if pick[1] == c else f"{c} / {pick[1]}"})
from collections import Counter
print('names built', len(out), Counter(o['writer'] for o in out))
print('mixed-culture names', sum('/' in o['culture'] for o in out))
json.dump(out, open('names_pool.json', 'w'), ensure_ascii=False, indent=0)
