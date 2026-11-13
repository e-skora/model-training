# Run from the session scratch folder that holds w1..w8 (writer files are not in git). Kept as a record of how the files were built.
"""Phase B validator for one writer's pairs file against that writer's plan slice.
Usage: python3 validate.py <pairs.jsonl> <slice.jsonl>
Prints every problem, then PASS or FAIL. Capitalised words other than the person's first name are listed as notes."""
import sys, json, re
FIELDS = ["id", "person", "a", "b", "label", "type", "hard", "reason", "source", "label_by"]
CHANGE = re.compile(r"\b(now|moved|switched|since|stopped|later|replaced)\b", re.I)
SOFT_CHANGE = re.compile(r"\b(these days|no longer|used to|anymore|any more|nowadays|new|newly|took up|gave up|ever since)\b", re.I)
REASON_BAN = re.compile(r"\b(first|second|memory a|memory b|earlier memory|later memory|same day|\d{4}-\d{2}-\d{2})\b", re.I)
WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?|\d+")
pairs_path, slice_path = sys.argv[1], sys.argv[2]
plan = {}
for line in open(slice_path, encoding="utf-8"):
    if line.strip():
        s = json.loads(line); plan[s["plan_id"]] = s
problems, notes, seen = [], [], set()
name_words = {}
for n, line in enumerate(open(pairs_path, encoding="utf-8"), 1):
    if not line.strip():
        problems.append(f"line {n}: empty line"); continue
    try:
        p = json.loads(line)
    except json.JSONDecodeError as e:
        problems.append(f"line {n}: not JSON ({e})"); continue
    pid = p.get("id")
    err = lambda m: problems.append(f"line {n} ({pid}): {m}")
    if list(p) != FIELDS: err(f"fields must be exactly {FIELDS} in this order")
    if pid not in plan: err("id is not in your slice"); continue
    if pid in seen: err("id repeats")
    seen.add(pid); s = plan[pid]
    for k in ("person", "label", "type", "hard"):
        if p.get(k) != s.get(k): err(f"{k} is {p.get(k)!r}, slice says {s.get(k)!r}")
    for w, k in (("a", "a_date"), ("b", "b_date")):
        m = p.get(w) or {}
        if list(m) != ["text", "written_at"]: err(f"memory {w} must have exactly text, written_at")
        if m.get("written_at") != s[k]: err(f"memory {w} written_at {m.get('written_at')!r}, slice says {s[k]!r}")
    if p.get("source") != "synthetic" or p.get("label_by") != "claude-code": err("source/label_by wrong")
    ta, tb = (p.get("a") or {}).get("text", ""), (p.get("b") or {}).get("text", "")
    first = s["person"].split()[0]
    for w, t in (("a", ta), ("b", tb)):
        if not re.search(rf"(?<!\w){re.escape(first)}(?!\w)", t): err(f"memory {w} does not name {first}")
        nw = len(WORD.findall(t))
        if not 4 <= nw <= 25: err(f"memory {w} has {nw} words; need 4 to 25")
    has_change = bool(CHANGE.search(ta) or CHANGE.search(tb))
    if s["change_word"] and not has_change: err("slice says change_word true, but neither memory has one of the seven words")
    if not s["change_word"] and has_change: err(f"slice says change_word false, but a memory has {(CHANGE.search(ta) or CHANGE.search(tb)).group(0)!r}")
    if s["hard"] == "far_apart_no_change" and (SOFT_CHANGE.search(ta) or SOFT_CHANGE.search(tb)):
        err(f"far_apart_no_change has change wording {(SOFT_CHANGE.search(ta) or SOFT_CHANGE.search(tb)).group(0)!r}")
    r = p.get("reason", "")
    if not r.strip(): err("no reason")
    if REASON_BAN.search(r): err(f"reason uses {REASON_BAN.search(r).group(0)!r}")
    if re.search(r"\bmara\b|lindqvist", line, re.I): err("banned name Mara/Lindqvist")
    for part in s["person"].split():
        name_words.setdefault(part, set()).add(pid)
    caps = sorted({w for t in (ta, tb, r) for i, w in enumerate(WORD.findall(t))
                   if w[:1].isupper() and w.split("'")[0] not in s["person"].split()} - {"I"})
    if caps: notes.append(f"{pid}: capitalised words {caps}")
missing = sorted(set(plan) - seen)
if missing: problems.append(f"slice ids with no pair yet: {len(missing)} (fine while you write in batches)")
for p in problems: print("PROBLEM", p)
for x in notes: print("note", x)
real = [p for p in problems if not p.startswith("slice ids with no pair")]
print(f"{len(seen)} pairs checked; " + ("FAIL" if real else "PASS"))
