# Blind judging brief (used by Claude in Cowork for every dataset review)

How it is used:
- Claude in Cowork copies the dataset into its own workspace.
- It strips everything except `id`, `person`, `a` and `b`.
- It shuffles the pairs with a fixed seed and splits them into parts of about 110.
- It gives each part to a separate agent with this brief, then compares the answers with the labels.
- Disputed `unclear` pairs are dropped, not relabelled (D-030, D-045).

---

You judge pairs of memories about one made-up person. You are NOT told the intended answers. Judge each pair yourself, carefully, one at a time.

## Labels
- "contradiction": the two memories cannot both be true. type = null.
- "non_contradiction": both can be true. type is one of:
  - "update": the later memory replaces the earlier one, and the change is shown ("She moved to Bergen" after "She lives in Oslo").
  - "compatible_detail": both are true together, and one adds detail to the same fact ("She likes tea" / "She drinks green tea in the morning").
  - "merely_related": a different fact on the same topic, with no clash ("She hikes" / "She owns hiking boots").
- "unclear": ONLY if a truly missing fact decides it and neither reading is more natural. Some pairs in a set may truly be unclear; most are not.

## Reading rules
1. Two memories that state a current favourite, choice or fact as true at the same moment, and disagree, are a contradiction. This holds even if they were written months apart, when neither describes a change (D-044).
2. If the later memory describes a change ("later", "now", "moved", "switched", or said another way), it is an update, not a contradiction.
3. If both can be true together, it is not a contradiction.
4. Do not invent an escape. If the two memories state different current facts about the same thing, it is a contradiction, even if you can imagine a way out ("maybe she has two dogs"). "unclear" is only for pairs where a truly missing fact decides it.
5. Different days, times or places with no exclusive word ("only", "always", "every day", "never") can both be true, so they are not a contradiction ("works from home on Fridays" / "works from home on Mondays" is merely_related).

## Also judge each pair with the dates hidden
Imagine both "written_at" dates were removed and the order of the two memories might be swapped. Would your label change?

## Output
Write one JSON object per input line, in input order, to the output file named in your task, with exactly these fields:
{"id": "...", "label": "...", "type": null or "...", "label_without_dates": "...", "sure": true or false, "note": "one short sentence; required when sure is false or label_without_dates differs"}

Do not look for, open or guess any other file. Do not write code to guess labels from wording; read each pair.
