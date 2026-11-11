# Brief: plan and write hard memory pairs for M3 (hard test set 2 and hard validation set 2)

You help write a test set for a small model (the "judge"). The judge reads two short memories about one made-up person and says whether they contradict each other. This set is the exam the next model is scored on, so every pair must have exactly one right answer, and every pair must be different from every other pair. Quality matters more than speed.

Every pair is built around a **trap**: a way a careless reader could get it wrong by reading the surface instead of the meaning.

You work in two phases. **Phase A:** plan your slots (a topic and a one-phrase idea for each), and suggest names. **Phase B** (only when you are told to start it): write the pairs from the final plan.

## What you may and may not read

- Read only this brief and the files named in your task (your own slot file and, in Phase B, your own plan slice).
- **Never open** anything under `data/` (except this brief), `results/`, `reviews/`, `runs/` or `prompts/`, another writer's files, or any folder named Aeverum. Others check your work against the old files; you never need to look.
- Never run any model. Do not run git. Do not edit any repo file. Write only your own output files, in the folder your task names.

## The labels

- **contradiction:** both cannot be true.
- **non_contradiction, update:** the later memory replaces the earlier one, and the change shows in the words.
- **non_contradiction, compatible_detail:** both are true, and one adds detail to the **same** fact ("likes tea" / "drinks green tea every morning").
- **non_contradiction, merely_related:** a **different** fact on the same topic, no clash ("hikes" / "owns hiking boots"). Two separate facts are `merely_related`, not `compatible_detail`.
- There are **no `unclear` pairs** in this set. Every pair must be decidable.

## Reading rules 1 to 5

1. Two memories that state a current favourite, choice or fact as true at the same moment, and disagree, are a `contradiction`.
2. If the second memory was written later and describes a change ("later", "now", "moved", "switched", or said another way), it is an `update`, not a contradiction.
3. If both can be true together, it is not a contradiction.
4. **Do not invent an escape.** If the two memories state different current facts about the same thing, it is a `contradiction`, even if you can imagine a way out ("maybe she has two dogs", "maybe the inside is yellow and the outside white").
5. **Different days, times or places with no exclusive word** ("only", "always", "every day", "never") can both be true, so they are not a contradiction: "works from home on Fridays" and "works from home on Mondays" is `merely_related`.

**Far apart (D-044):** two memories written months apart that disagree, with nothing in their words showing a change, stay a `contradiction` (rule 1). A change has to be in the text, not only in the dates.

### One slot or many?

Before you label a clash, ask whether the thing has room for **one** value or **many**:

- **One slot:** a favourite, *the* car, where someone lives, what floor the flat is on, how many children, the day a fixed weekly class meets, how a past event went, anything with "only", "never", "always", "her one ...". Two different values are a `contradiction`, and "maybe she has two" is an invented escape (rule 4).
- **Many slots:** activities, skills, languages, instruments, places someone sometimes goes, things someone sometimes eats. Two different values can both be true (rules 3 and 5), unless an exclusive word closes the slot.

Keep away from the fuzzy middle. A `contradiction` should be clearly one slot; a `merely_related` near miss should be clearly many slots. If you cannot tell which it is, choose a different idea.

## The traps

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

Examples (all made up for this brief; never reuse their names, topics or wording):

- `explain_away`: "Benedek lives alone." / "Benedek shares his flat with two old school friends." (Escape: "maybe he has two flats". No.)
- `paraphrase`: "Ilse never learned to ride a bike." / "Ilse cycles to the office along the canal each morning."
- `small_difference`: "Tomasz's flat is on the fourth floor." / "Tomasz's flat is on the fifth floor."
- `change_word_contradiction`: "Rohan switched from tea to coffee last spring." / "Rohan switched from coffee to tea last spring." (same day)
- `far_apart_no_change`: "Signe's favourite season is autumn." (2024-02-10) / "Signe's favourite season is spring." (2025-01-22)
- `indirect_update`: "Hiroko works at a flower shop." / "Hiroko left the flower shop in October and started at a bakery." (b dated after October)
- `update_with_overlap`: "Odile keeps her bike in the hallway." / "Odile keeps her bike in the shed she built last summer, no longer in the hallway."
- `looks_like_clash`: "Leilani runs about five kilometres each morning." / "Leilani's morning run is a little over three miles."
- `near_identical` (compatible_detail): "Anouk bakes bread on Sundays." / "Anouk bakes sourdough on Sunday mornings."
- `rule5_bait`: "Cyrus swims at the lake on Saturdays." / "Cyrus swims at the indoor pool on Tuesday evenings."
- `near_identical` (merely_related): "Ewa's kitchen door is painted blue." / "Ewa's garden shed door is painted blue."

Hard cases inside the traps:
- **Contradictions must not read like updates.** The judge also sees each pair with dates hidden and with the two memories swapped. If only one memory has change wording ("Ravi now lives in Pune" / "Ravi lives in Goa"), it reads as an update. So in `change_word_contradiction`, the change word is in both memories, or it is about something else, or both tell the same past change two ways.
- **Updates must show their change in the words.** The newer memory itself must say it is the newer state ("these days ... instead of", "gave up", "took up", "no longer", "her new ...", "after the move", "as of March", "used to ... but"). In either order and without dates, a careful reader can still tell which is newer.
- **No `far_apart_no_change` pair contains any change wording at all** (not only the seven words: no "these days", "no longer", "used to", "new", "anymore" either).

## Text alone decides

Every pair must have one right answer from its **text alone**: check it once with its dates, once as if the dates were gone, and once with the memories swapped. All readings must give the same label and type.

## Your slot file

Your task names a slot file. Each slot is one pair to plan and later write, with these fields fixed in advance by a program (seed 20261111), which you must follow exactly:

- `plan_id`: the pair's id (keep it).
- `set`: `test` or `valid` (you treat both the same way).
- `hard`, `label`, `type`: the trap, label and type.
- `same_day`: whether both memories are written on the same day.
- `change_word`: `true` means **at least one** memory contains one of the seven change words (now, moved, switched, since, stopped, later, replaced) as a whole word, used naturally. `false` means **neither** memory contains any of the seven, in any capitalisation. Words like "moving", "stops", "switch" or "nowhere" do not count either way, but do not lean on them to dodge the rule.
- `a_date`, `b_date`: the two `written_at` dates. Memory a is always the one dated first (or the same day). In updates, b is the newer state.

**Dates must fit the time words.** If a memory names a month, season, year or event ("in September", "last summer", "since January", "after the wedding"), it must agree with the dates. An update that says "moved in September" needs a dated before that September and b after it. A contradiction that says "since January" on both sides works only if both refer to the same January. If a time word does not fit your dates, leave it out or choose another.

## Phase A: plan your slots and suggest names

For **every** slot, write one line (JSONL) with exactly these fields:

```json
{"plan_id": "p123", "topic": "rooftop beehives", "clash": "keeps two hives on the roof / has never kept bees"}
```

- `topic`: 2 to 5 words, concrete and specific (not "food" but "breakfast porridge"; not "pets" but "elderly cat's diet").
- `clash`: one short phrase (under 15 words) saying what the clash is (contradictions), what the change is (updates), or what the two facts are and why they fit (compatible_detail, merely_related).
- Stay inside **your topic area** (in your task). Use its whole breadth.
- **Every line a different idea.** No two of your lines may share a topic, and no two may use the same clash idea, even on different topics. Do not use these ideas, which earlier sets already used many times: a vegetarian eating meat, never owned a car but drives one, an only child with a sister, a night-shift baker who works days, a fear of dogs and cuddling puppies, two instruments taken as a clash, a nickname and a full name.
- **Fit the trap.** The idea must be one that works for the slot's trap, label, type, `same_day` and `change_word`, so a pair can be written from it without bending.
- Avoid the most obvious everyday ideas; think of the specific lives of people from many places.

Also write **a list of 150 candidate names** for people, from your assigned cultures only, one per line as `First Surname | culture` (for example `Tavita Leiataua | Samoan`). Rules:
- Real-sounding names that people of that culture actually use; mixed genders; no famous people; never Mara or Lindqvist.
- **Every name word once:** no first name or surname appears twice in your list.
- Prefer names that are less common, because thousands of names are already taken by earlier sets. A program screens them, and only names that pass are used. Every pair gets a name from some writer's list, chosen by the program, so your pairs will have people from many cultures.

Then reply in under 100 words: the two file paths and their line counts. Stop there and wait.

## Phase B: write the pairs

You will get your plan slice: your slots, each with its final `topic`, `clash`, `person` and `culture`. Write one pair per slot. One line (JSONL) per pair, exactly these fields, in this order:

```json
{"id": "p123", "person": "Tavita Leiataua", "a": {"text": "...", "written_at": "2025-02-11"}, "b": {"text": "...", "written_at": "2025-09-03"}, "label": "non_contradiction", "type": "update", "hard": "update_with_overlap", "reason": "...", "source": "synthetic", "label_by": "claude-code"}
```

- `id` is the slot's `plan_id`. `person`, the dates, `label`, `type` and `hard` are copied from your slice exactly. `source` is always `"synthetic"` and `label_by` always `"claude-code"`.
- Write the lines with a small Python script (`json.dumps(row, ensure_ascii=False)`), so every line parses and letters like é or ş stay real letters.
- **Follow the plan.** The pair must be about the slot's topic and carry its clash idea. If an idea turns out not to work, change it as little as possible and list the id in your reply.
- **Name the person.** Both memories name the person by the first name exactly as in `person` ("Tavita's" is fine). Others may be mentioned in passing, but the fact is about the named person. Use the culture to make details natural, without clichés.

### Reasons

One short sentence that stays true with the memories in either order and without dates. Never write "first", "second", "memory a/b", "earlier memory", a date, or "same day". Describe by content: "One says Pim bakes bread every Sunday, the other that Pim has never baked bread." For updates, "the newer memory says ..." is fine, because the words show which is newer. Vary how reasons start.

### No templates, and few capitalised words

- No sentence pattern more than 3 times in your file. "X's favourite colour is A" and "X's favourite food is B" count as the same pattern.
- Vary length (4 to 25 words per memory), structure, tense and detail. Sometimes two facts in one memory; sometimes the clash is in a side clause.
- Not every contradiction is a one-word swap, and not every one-word swap is a contradiction.
- **Keep capitalised words few** besides the person's first name: place, pet, shop and brand names are checked against thousands of names from earlier sets and may be rejected. Prefer "the river market" to a named market. Never use another person's full name.
- Everyday topics only. Nothing sensitive: no illness beyond everyday things like glasses or a cold, no religion, politics, crime, money troubles, sex or death.

### How to work in Phase B

1. Write in the batches your task gives (about 50 pairs each).
2. After each batch, reread every new pair against reading rules 1 to 5: right label? right type (same fact or different fact)? does the trap fit? does it hold without dates and swapped? do the dates fit the time words? does the change-word rule hold? Fix what is weak.
3. Run the validator your task names on your file and fix until it prints PASS.
4. Reply in under 150 words: the file path and line count; the ids of any pair you changed from the plan, or were unsure about, and why, in a few words; and the validator's last line. Then wait for the next batch.
