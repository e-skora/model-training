# Run from the session scratch folder that holds w1..w8 (writer files are not in git). Kept as a record of how the files were built.
"""Build the 800 plan slots (set, trap, label, type, same-day, change word, dates, writer). Seed 20261111."""
import json, random, math
from datetime import date, timedelta
SEED = 20261111
# trap: (label, type, test n, test same-day, test change, valid n, valid same-day, valid change)
TRAPS = [
 ("explain_away","contradiction",None,80,40,0,27,13,0),
 ("paraphrase","contradiction",None,60,30,2,20,10,1),
 ("small_difference","contradiction",None,60,30,0,20,10,0),
 ("change_word_contradiction","contradiction",None,50,30,50,17,10,17),
 ("far_apart_no_change","contradiction",None,50,0,0,16,0,0),
 ("indirect_update","non_contradiction","update",50,0,0,17,0,0),
 ("update_with_overlap","non_contradiction","update",50,0,46,16,0,15),
 ("looks_like_clash","non_contradiction","compatible_detail",60,30,4,20,10,2),
 ("near_identical","non_contradiction","compatible_detail",40,20,8,13,6,3),
 ("rule5_bait","non_contradiction","merely_related",50,26,6,17,9,2),
 ("near_identical","non_contradiction","merely_related",50,26,6,17,9,2),
]
rng = random.Random(SEED)
START, END = date(2023,1,1), date(2026,9,30)
def gap(hard, typ):
    if hard == "far_apart_no_change": lo, hi = 120, 540
    elif typ == "update": lo, hi = 7, 900
    elif typ is None: lo, hi = 3, 56
    else: lo, hi = 3, 90
    return round(math.exp(rng.uniform(math.log(lo), math.log(hi))))
slots = []
for hard, label, typ, tn, ts, tc, vn, vs, vc in TRAPS:
    for st, n, s, c in (("test", tn, ts, tc), ("valid", vn, vs, vc)):
        same = [True]*s + [False]*(n-s); rng.shuffle(same)
        chg = [True]*c + [False]*(n-c); rng.shuffle(chg)
        for i in range(n):
            g = 0 if same[i] else gap(hard, typ)
            a = START + timedelta(days=rng.randrange((END-START).days - g + 1))
            slots.append(dict(set=st, hard=hard, label=label, type=typ, same_day=same[i], change_word=chg[i],
                              a_date=a.isoformat(), b_date=(a+timedelta(days=g)).isoformat()))
assert len(slots) == 800
# deal to 8 writers: shuffle, then sort by (set, trap), deal round-robin so each writer gets every trap
rng.shuffle(slots)
slots.sort(key=lambda s: (s["set"], s["hard"], s["type"] or ""))
for i, s in enumerate(slots):
    s["writer"] = f"w{i % 8 + 1}"
# plan ids in a fresh shuffled order, so the id says nothing about the trap
order = list(range(800)); rng.shuffle(order)
for pid, i in enumerate(order, 1):
    slots[i]["plan_id"] = f"p{pid:03d}"
slots.sort(key=lambda s: s["plan_id"])
with open("slots.json", "w") as f:
    json.dump(slots, f, indent=0)
from collections import Counter
print(Counter((s["writer"], s["set"]) for s in slots))
print(Counter(s["writer"] for s in slots))
for w in ("w1","w8"):
    print(w, Counter((s["hard"], s["type"] or s["label"]) for s in slots if s["writer"]==w))
