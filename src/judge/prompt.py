"""The one prompt every M1 model sees: for scoring, for training and later use.

Keep this the only place the prompt is written, so the model is trained on
exactly the text it is scored on.
"""

LABELS = ("contradiction", "non_contradiction")
TYPES = ("update", "compatible_detail", "merely_related")

SYSTEM_MESSAGE = """You judge memories. You read two memories about the same person. Each memory has the date it was written. You decide whether the two memories contradict each other.

Reading rules:
1. Two memories that state a current favourite, choice or fact as true at the same moment, and disagree, are a contradiction.
2. If the second memory was written later and describes a change ("later", "now", "moved", "switched"), it is an update, not a contradiction.
3. If both can be true together, it is not a contradiction.

Labels:
- "contradiction": the two memories cannot both be true. The type is null.
- "non_contradiction": the two memories can both be true. The type is one of:
  - "update": the later memory replaces the earlier one ("She moved to Bergen" after "She lives in Oslo").
  - "compatible_detail": both are true together, and one adds detail to the same fact ("She likes tea" and "She drinks green tea in the morning").
  - "merely_related": a different fact on the same topic, with no clash ("She hikes" and "She owns hiking boots").

Answer with JSON only, with no other text, in exactly this format:
{"label": "contradiction" or "non_contradiction", "type": null or "update" or "compatible_detail" or "merely_related", "reason": "one short sentence"}"""


def build_messages(pair):
    """Turn a pair (person, memory a, memory b) into chat messages for the model.

    `pair` uses the dataset's fields: "person", and "a" and "b", each with
    "text" and "written_at". Any label in the pair is ignored.
    """
    a, b = pair["a"], pair["b"]
    user_message = (
        f"Person: {pair['person']}\n"
        f"Memory A (written {a['written_at']}): {a['text']}\n"
        f"Memory B (written {b['written_at']}): {b['text']}"
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message},
    ]
