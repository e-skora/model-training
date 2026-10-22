# Brief: write 200 made-up memory pairs for the M1 training set

You write training data for a small model that judges whether two short "memories" about one person contradict each other. Quality matters more than speed: the model learns exactly what you write, including any lazy patterns.

Repo (read-only for you, except where said): `/Users/frank/code/model-training`. Read `milestones/M1-contradiction-judge/SPEC.md` (labels and reading rules) and `src/judge/prompt.py` (the prompt the model sees). Do NOT open `data/m1/test.jsonl` (the frozen test set; you must not copy its style or content). Do not read anything outside this repo and your scratch folder, and never anything from folders named Aeverum. Do not run git. Do not edit any repo file.

Your output: ONE file, path given in your assignment, JSONL, one pair per line, appended batch by batch.

## Line format (exactly these fields)

```json
{"id": "g1-001", "person": "Pim Oduya", "a": {"text": "Pim ...", "written_at": "2025-03-04"}, "b": {"text": "Pim ...", "written_at": "2025-03-04"}, "label": "contradiction", "type": null, "reason": "One short sentence."}
```

- `label`: `contradiction`, `non_contradiction` or `unclear`.
- `type`: `null` for `contradiction` and `unclear`; for `non_contradiction` one of `update`, `compatible_detail`, `merely_related`.
- `a` is ALWAYS the earlier memory (or same day). `b` is never dated before `a`. Dates between 2023-01-01 and 2026-09-30, format YYYY-MM-DD.
- Write the JSON with a script or carefully by hand; it must parse. Keep non-English letters as real letters (UTF-8), not escapes.

## The labels and reading rules

1. Two memories that state a current favourite, choice or fact as true at the same moment, and disagree, are a **contradiction**.
2. If one memory was written later and describes a change, it is an **update**, not a contradiction.
3. If both can be true together, it is not a contradiction.

- **contradiction**: both cannot be true. Includes "silent changes": memories months apart that state a current fact differently with NO change wording ("Ana's car is a grey Fiat" ... a year later "Ana's car is a blue Toyota") stay **contradiction** under rule 1. Also: a past event cannot be updated ("At the 3 May party Ana sang first" vs later "At the 3 May party Ana sang last" is a contradiction even if months apart).
- **update** (non_contradiction): the later memory replaces the earlier one, and the change is clear FROM THE WORDING ALONE. This is essential: in training, dates are hidden in a third of examples and the two memories are shown in reversed order in half. So the newer memory must itself signal that it is the newer state ("moved", "now", "switched", "no longer", "new", "these days ... instead of", "gave up", "took up", "used to ... but", "after the renovation", "as of March"). b is always dated strictly later than a (gaps from 3 days to 30 months).
- **compatible_detail** (non_contradiction): both true together, and one adds detail to the SAME fact ("likes tea" / "drinks green tea every morning").
- **merely_related** (non_contradiction): a DIFFERENT fact on the same topic, no clash ("hikes" / "owns hiking boots"). If the two memories are two separate facts, it is merely_related, not compatible_detail.
- **unclear**: whether they clash depends on a fact the text does not give (scope or time), so a careful reader honestly cannot tell. Examples of the idea (do not copy): "is allergic to cats" / "has a cat" (mild allergy? still owns it?); "works at the library on weekdays" / "works at a café on weekday mornings" (part-time both, or a clash?); "drives a blue car" / "drives a red car to work" (a work car?). NEVER use unclear for silent changes (those are contradiction by rule 1), and never for pairs where rule 1, 2 or 3 clearly decides. Reason form: say which missing fact decides it.

## Reasons

One short sentence that is still true when the two memories are shown in either order and without dates. So never write "first", "second", "memory A/B", "the earlier memory", dates, or "same day". Describe by content: "One says Pim bakes bread every Sunday, the other that Pim has never baked bread." For updates, "the newer memory says ..." is fine because the wording shows which is newer.

## Hard cases, by name (counts are for your 200)

- **Change words in every label.** The change words are: now, moved, switched, since, stopped, later, replaced. At least 20 of your contradictions contain one (for example both memories say "now" on the same day and disagree; the change word is about something else; both describe the same past change differently). At least 5 compatible_detail and at least 5 merely_related pairs contain one with no clash ("Pim switched to oat milk in his morning coffee" / "Pim drinks his afternoon coffee black"). Unclear pairs may contain them too.
- **Indirect updates.** At least 7 of your 34 updates use NONE of those seven words, yet the wording still shows the change ("Pim's new flat ...", "These days Pim cycles instead of driving", "Pim gave up meat after the trip", "As of January Pim teaches only mornings").
- **Near-misses that are not contradictions:** compatible details that match numbers, units, ranges or counts ("gets up before seven" / "gets up around 6:30"; "has two dogs" / "has a dog named Bingo"; "1.85 metres" / "a little over six feet"); some give the detail first. Merely related pairs that differ by one or two words but do not clash ("works from home on Fridays" / "works from home on Mondays"). At least 5 of each.
- **Contradictions that look mild:** small differences that still clash (one weekly class on Tuesday vs on Thursday; "has never learned to swim" vs "swims laps every Saturday").
- **Silent changes and past events:** about 10 contradictions months apart with no change wording, and about 5 about a fixed past event.

## Dates

About half of each label (contradiction, compatible_detail, merely_related, unclear) has both memories on the same day; the rest on different days. Updates are never same-day. Vary the gaps.

## Variety (the checker enforces part of this)

- No sentence pattern more than 3 times in your 200 pairs. "X's favourite colour is A" / "X's favourite food is B" count as the same pattern. Vary length (4 to 25 words), structure, tense, detail; sometimes two facts in one memory; sometimes the clash is in a side clause.
- Not every contradiction is a one-word swap; not every one-word swap is a contradiction.
- Everyday topics only: home, work, food, hobbies, travel, pets, routines, belongings, tastes, places, sport, study, shopping, clubs. Nothing sensitive (no illness beyond everyday things like glasses or a cold, no religion, politics, crime, money troubles, sex, death).
- Each pair is about ONE person. Both memories name that person by their first name, exactly as written first in `person` (possessive "Pim's" is fine). A memory may mention others in passing, but its fact is about the named person.
- Every person is made up. Full names unique across your 200. Names from your assigned cultures only, mixing genders. Never use any word from `banned-names.txt` (in the same folder as this brief) as a first name or surname or anywhere in text (the checker matches whole words, case-sensitive, so also avoid place names like "Lima" if listed), and never Mara or Lindqvist. No famous real people.

## Process

1. Write batch 1 (50 pairs: about 22-23 contradiction, 8-9 update, 8 compatible_detail, 8 merely_related, 2-3 unclear), mixed order, append to your file.
2. Reread every pair of the batch against the reading rules: is the label right? would a careful reader agree? is the type right (same fact vs different fact)? is each update clear from wording alone, in either order? do reasons avoid order and dates? Fix problems.
3. Run the checker from the repo folder:
   `cd /Users/frank/code/model-training && uv run python scripts/check_dataset.py <your file> --against data/m1/test.jsonl`
   It stops at the first problem; fix and rerun until it prints PASS. (Its share checks only apply once a label has 20 pairs.)
4. Repeat for batches 2, 3 and 4. Totals for your 200 must be exactly: 90 contradiction, 34 update, 33 compatible_detail, 33 merely_related, 10 unclear. Ids `<prefix>-001` to `<prefix>-200`.
5. Final: the checker passes on the whole file. Report back in under 150 words: label counts, how many contradictions with change words, indirect updates, same-day counts per label, and anything you were unsure about (pair ids).
