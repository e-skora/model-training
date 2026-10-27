# Brief: write hard memory pairs for the M2 training and validation sets

You write training data for a small model (the "judge") that reads two short memories about one person and says whether they contradict each other. The model will learn exactly what you write, including any lazy habit. Quality matters more than speed.

These pairs are for reinforcement learning (RL): the model answers each pair several times, and it only learns from pairs it sometimes gets wrong. So every pair is built around a **trap**: a way a careless reader could get it wrong by reading the surface instead of the meaning. Easy pairs teach nothing.

## What you may and may not read

- Read: this brief; `milestones/M1-contradiction-judge/SPEC.md` (labels and reading rules); `src/judge/prompt.py` (what the model sees).
- **Never open** `data/m2/test-hard.jsonl` (the frozen exam), anything in `results/`, `reviews/` or `data/m1/`, or another writer's file. The checker compares your file with those files by itself; you never need to look.
- Never read anything outside this repo and your own output folder, and never anything from a folder named Aeverum.
- Do not run git. Do not edit any repo file. Write only your own output file.

## Line format

One pair per line (JSONL), exactly these fields, in this order:

```json
{"id": "w00-001", "person": "Odile Varenne", "a": {"text": "Odile keeps her bike in the hallway.", "written_at": "2025-02-11"}, "b": {"text": "Odile keeps her bike in the shed she built last summer, no longer in the hallway.", "written_at": "2025-09-03"}, "label": "non_contradiction", "type": "update", "hard": "update_with_overlap", "reason": "The newer memory says Odile moved her bike from the hallway to a shed.", "source": "synthetic", "label_by": "claude-code"}
```

- `label`: `contradiction`, `non_contradiction` or `unclear`.
- `type`: `null` for `contradiction` and `unclear`. For `non_contradiction`: `update`, `compatible_detail` or `merely_related`.
- `hard`: the pair's trap, from the list below. It must fit the label and type.
- `source` is always `"synthetic"` and `label_by` is always `"claude-code"` (D-043).
- `a` is always the earlier memory, or the same day. `b` is never dated before `a`. Dates are between 2023-01-01 and 2026-09-30, written YYYY-MM-DD.
- Write the lines with a small Python script (`json.dumps(row, ensure_ascii=False)`), so every line parses and letters like é or ş stay real letters.

## The labels

- **contradiction:** both cannot be true.
- **non_contradiction, update:** the later memory replaces the earlier one, and the change shows in the words.
- **non_contradiction, compatible_detail:** both are true, and one adds detail to the **same** fact ("likes tea" / "drinks green tea every morning").
- **non_contradiction, merely_related:** a **different** fact on the same topic, no clash ("hikes" / "owns hiking boots"). Two separate facts are `merely_related`, not `compatible_detail`.
- **unclear:** a fact that is truly missing decides it, and neither reading is more natural. Rare, and with strict rules (below).

## Reading rules 1 to 5 (from the M1 SPEC)

1. Two memories that state a current favourite, choice or fact as true at the same moment, and disagree, are a `contradiction`.
2. If the second memory was written later and describes a change ("later", "now", "moved", "switched"), it is an `update`, not a contradiction.
3. If both can be true together, it is not a contradiction.
4. **Do not invent an escape.** If the two memories state different current facts about the same thing, it is a `contradiction`, even if you can imagine a way out ("maybe she has two dogs", "maybe the inside is yellow and the outside white"). `unclear` is only for pairs where a fact that is truly missing decides it, and neither reading is more natural (D-029).
5. **Different days, times or places with no exclusive word** ("only", "always", "every day", "never") can both be true, so they are not a contradiction: "works from home on Fridays" and "works from home on Mondays" is `merely_related` (D-029).

**D-044:** two memories written months apart that disagree, with nothing in their words showing a change, stay a `contradiction` (rule 1). A change has to be in the text, not only in the dates.

### One slot or many? (how rules 3 and 4 fit together)

Before you label a clash, ask whether the thing has room for **one** value or **many**:

- **One slot:** a favourite, *the* car, where someone lives, what floor the flat is on, how many children, the day a fixed weekly class meets, how a past event went, anything with "only", "never", "always", "her one ...". Two different values are a `contradiction`, and "maybe she has two" is an invented escape (rule 4).
- **Many slots:** activities, skills, languages, instruments, places someone sometimes goes, things someone sometimes eats. Two different values can both be true (rule 3, rule 5), unless an exclusive word closes the slot.

Keep away from the fuzzy middle. A `contradiction` should be clearly one slot; a `merely_related` near miss should be clearly many slots. If you cannot tell which it is, write a different pair.

## The traps

Each trap has a definition and one made-up example. The examples show the idea; never reuse their names, topics or wording.

| Trap (`hard`) | Label, type | What it is |
|---|---|---|
| `explain_away` | contradiction | You could invent a way both are true, but rule 4 says no. |
| `paraphrase` | contradiction | The clash is said in different words; seeing it takes one step of everyday sense. |
| `small_difference` | contradiction | One number, day, colour or place differs, in a one-slot fact. |
| `change_word_contradiction` | contradiction | A change word appears, but both memories describe the same moment, or the same past change told two different ways. |
| `far_apart_no_change` | contradiction | Written months apart, nothing in the words shows a change (rule 1, D-044). |
| `indirect_update` | update | A change told without any of the seven change words (now, moved, switched, since, stopped, later, replaced). |
| `update_with_overlap` | update | Wording close to a contradiction, plus a clear change. |
| `looks_like_clash` | compatible_detail | Units, counts, part and whole, nicknames, general and specific: looks different, says the same. |
| `near_identical` | compatible_detail | Almost the same sentence, and the second adds detail to the same fact. |
| `rule5_bait` | merely_related | Different days, times or places, no exclusive word (rule 5). |
| `near_identical` | merely_related | Almost the same sentence, but a different fact; no clash. |
| `true_unclear` | unclear | A truly missing fact decides it, and neither reading is more natural. |

Examples (all made up for this brief):

- `explain_away`: "Benedek lives alone." / "Benedek shares his flat with two old school friends." (Escape: "maybe he has two flats". No.)
- `paraphrase`: "Ilse never learned to ride a bike." / "Ilse cycles to the office along the canal each morning."
- `small_difference`: "Tomasz's flat is on the fourth floor." / "Tomasz's flat is on the fifth floor."
- `change_word_contradiction`: "Rohan switched from tea to coffee last spring." / "Rohan switched from coffee to tea last spring." (same day)
- `far_apart_no_change`: "Signe's favourite season is autumn." (2024-02-10) / "Signe's favourite season is spring." (2025-01-22)
- `indirect_update`: "Hiroko works at a flower shop." / "Hiroko left the flower shop in October and started at a bakery." (b dated after October)
- `update_with_overlap`: the line format example above.
- `looks_like_clash`: "Leilani runs about five kilometres each morning." / "Leilani's morning run is a little over three miles."
- `near_identical` (compatible_detail): "Anouk bakes bread on Sundays." / "Anouk bakes sourdough on Sunday mornings."
- `rule5_bait`: "Cyrus swims at the lake on Saturdays." / "Cyrus swims at the indoor pool on Tuesday evenings."
- `near_identical` (merely_related): "Ewa's kitchen door is painted blue." / "Ewa's garden shed door is painted blue."
- `true_unclear`: "Kaveh has no pets." / "Kaveh feeds the tortoise in the back garden every evening." (Whose tortoise is it? The text does not say, and both readings are natural.)

Hard cases inside the traps:
- **Contradictions must not read like updates.** In training, dates are hidden in a third of examples and the two memories are swapped in half. If only one memory has change wording ("Ravi now lives in Pune" / "Ravi lives in Goa"), it reads as an update. So in `change_word_contradiction`, the change word is in both memories, or it is about something else, or both tell the same past change two ways.
- **Updates must show their change in the words.** The newer memory itself must say it is the newer state ("these days ... instead of", "gave up", "took up", "no longer", "her new ...", "after the move", "as of March", "used to ... but"). In either order and without dates, a careful reader can still tell which is newer.
- **No `far_apart_no_change` pair may contain any change wording** at all, and none is same-day.
- **Change words in every label.** In each label and type of your file, at least 15% of pairs contain one of the seven change words (in updates, at least 15% contain none). In non-contradictions the change word is about something that does not clash ("Pim switched to oat milk in his morning coffee" / "Pim drinks his afternoon coffee black").

## Text alone decides

Every pair except a `true_unclear` one must have one right answer from its **text alone**: check each pair once with its dates and once as if the dates were gone, and once with the memories swapped. All readings must give the same label.

## Dates

- About **half** of each label and type is written on the same day: contradiction (counting all its traps together), compatible_detail, merely_related, unclear. `change_word_contradiction`, `explain_away`, `paraphrase`, `small_difference` and the non-contradiction traps each aim at about half.
- **Updates** always have b strictly later than a (gaps from 3 days to 30 months). Never same-day.
- **`far_apart_no_change`:** 4 to 18 months apart.
- Other contradictions written on different days: a few days to 8 weeks apart.
- **b is never dated before a.**
- **Time words must fit the dates.** If a memory says "in September", "last year", "since January" or "after the wedding", the dates must agree: an update that says "moved in September" has a dated before that September and b after it. A contradiction that says "since January" on both sides should be same-day, or the two Januaries differ and both can be true.

## `unclear` rules (D-029, D-030)

- `unclear` is only for a pair where **a fact that is truly missing decides it, and neither reading is more natural** (rule 4). Name that fact in the reason.
- Good missing facts are concrete facts about the world: whose thing it is, which copy or which language was used, whether a visit means living there, who else lives in the home.
- **Not `unclear`:**
  - an invented escape ("maybe two dogs", "maybe two jobs", "maybe the inside is a different colour"): that is a `contradiction`;
  - two activities on the same day with no times given, or different days or places with no exclusive word: that is `merely_related` (rule 5);
  - a silent change months apart: that is a `contradiction` (D-044);
  - an argument about what a word means ("is a scone a cake?");
  - anything rule 1, 2 or 3 clearly decides.
- A second judge will read every `unclear` pair blind. Only the ones it also calls `unclear` are kept (D-030). Write them so a careful stranger honestly cannot tell.
- **Vary how `unclear` reasons start.** No two of your `unclear` reasons may start with the same three words. Do not start with "It depends on" or "Whether".
- About half of your `unclear` pairs are same-day, and at least 1 of your 6 contains one of the seven change words.

## Reasons

One short sentence that stays true with the memories in either order and without dates. Never write "first", "second", "memory a/b", "earlier memory", a date, or "same day". Describe by content: "One says Pim bakes bread every Sunday, the other that Pim has never baked bread." For updates, "the newer memory says ..." is fine, because the words show which is newer.

## People and names

- **A fresh person in every pair.** Each pair is about one made-up person. Both memories name that person by the first name exactly as written first in `person` ("Pim's" is fine). Others may be mentioned in passing, but the fact is about the named person.
- **Each name word once.** In your file, no first name or surname is used twice, even in different people.
- **Names from your assigned cultures only**, mixing genders. No famous real people. Never Mara or Lindqvist.
- **The name wall.** The checker rejects any word, anywhere in your file, that is a name word of a person in the M1 files or the hard test set. It matches whole words and is case-sensitive, so it also catches capitalised ordinary words: place names, pet names, brand names, mountains, even a capitalised word at the start of a reason. Before each checker run, list every capitalised word in your new lines and keep them few. When the checker rejects a word, change it; do not try to find out more about the other files.

## Topics

Everyday topics only: home, work, food, hobbies, travel, pets, routines, belongings, tastes, places, sport, study, shopping, clubs. Nothing sensitive: no illness beyond everyday things like glasses or a cold, no religion, politics, crime, money troubles, sex or death. Lean on your assigned topic focus, but not only on it.

## No templates

- No sentence pattern more than 3 times in your file. "X's favourite colour is A" and "X's favourite food is B" count as the same pattern. The checker enforces part of this.
- Vary length (4 to 25 words), structure, tense and detail. Sometimes two facts in one memory; sometimes the clash is in a side clause.
- Not every contradiction is a one-word swap, and not every one-word swap is a contradiction.
- Do not write two pairs of one trap the same way. Do not reuse the examples in this brief.

## How to work

1. Write your file in **4 batches of about 26 pairs**, mixing traps and labels within each batch.
2. After each batch, reread every new pair against reading rules 1 to 5 and the checks above: right label? right type (same fact or different fact)? does the trap fit? does it hold without dates and swapped? do the dates fit the time words? Fix what is weak.
3. Run the checker from the repo folder after each batch, and fix until it prints PASS:
   ```
   cd /Users/frank/code/model-training && uv run python scripts/check_dataset.py <your file> --against data/m2/test-hard.jsonl --against data/m1/test.jsonl --distinct-from data/m1/train.jsonl --distinct-from data/m1/valid.jsonl
   ```
   It stops at the first problem. Its share checks only start once a label or type has 20 pairs.
4. Your counts per trap are in your assignment. Ids are `<prefix>-001` upward, in file order.

## What you hand back

When the checker prints PASS on your whole file, reply in under 150 words:
- the file path and its line count;
- pairs per trap, and how many are same-day per label;
- how many contain a change word per label;
- the ids of your `unclear` pairs;
- the ids of any pair you were unsure about, and why, in a few words;
- the checker's last line.
