"""The one prompt every M1 model sees: for scoring, for training and later use.

Keep this the only place the prompt is written, so the model is trained on
exactly the text it is scored on. `prompt_sha256` fingerprints it (D-027).
"""

import hashlib
import json

LABELS = ("contradiction", "non_contradiction", "unclear")
TYPES = ("update", "compatible_detail", "merely_related")

SYSTEM_MESSAGE = """You judge memories. You read two memories about the same person. Each memory may show the date it was written, but dates can be missing. The order of the two memories means nothing: either one may be the older. You decide whether the two memories contradict each other.

Reading rules:
1. Two memories that state a current favourite, choice or fact as true at the same moment, and disagree, are a contradiction.
2. If one memory was written later and describes a change ("later", "now", "moved", "switched"), it is an update, not a contradiction. When dates are missing, judge an update from the wording alone.
3. If both can be true together, it is not a contradiction.

Labels:
- "contradiction": the two memories cannot both be true. The type is null.
- "non_contradiction": the two memories can both be true. The type is one of:
  - "update": the later memory replaces the earlier one ("She moved to Bergen" after "She lives in Oslo").
  - "compatible_detail": both are true together, and one adds detail to the same fact ("She likes tea" and "She drinks green tea in the morning").
  - "merely_related": a different fact on the same topic, with no clash ("She hikes" and "She owns hiking boots").
- "unclear": the text alone cannot settle it, because whether they clash depends on facts the memories do not give, such as the scope or the time. The type is null. Do not use it when rule 1 already decides.

Answer with JSON only, with no other text, in exactly this format:
{"label": "contradiction" or "non_contradiction" or "unclear", "type": null or "update" or "compatible_detail" or "merely_related", "reason": "one short sentence"}"""

# How each memory is written in the user message, with and without its date.
MEMORY_LINE = "Memory {name} (written {written_at}): {text}"
MEMORY_LINE_NO_DATE = "Memory {name}: {text}"
USER_TEMPLATE = "Person: {person}\n{memory_a}\n{memory_b}"


def memory_line(name, memory, dates):
    if dates and memory.get("written_at"):
        return MEMORY_LINE.format(name=name, written_at=memory["written_at"], text=memory["text"])
    return MEMORY_LINE_NO_DATE.format(name=name, text=memory["text"])


def build_messages(pair, dates=True):
    """Turn a pair (person, memory a, memory b) into chat messages for the model.

    `pair` uses the dataset's fields: "person", and "a" and "b", each with
    "text" and "written_at". Any label in the pair is ignored. With
    `dates=False`, or when a memory has no "written_at", its date is left out.
    """
    user_message = USER_TEMPLATE.format(
        person=pair["person"],
        memory_a=memory_line("A", pair["a"], dates),
        memory_b=memory_line("B", pair["b"], dates),
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message},
    ]


def prompt_sha256():
    """SHA-256 of the prompt template: the system message and the user-message templates (D-027)."""
    template = {
        "system": SYSTEM_MESSAGE,
        "user": USER_TEMPLATE,
        "memory": MEMORY_LINE,
        "memory_no_date": MEMORY_LINE_NO_DATE,
    }
    return hashlib.sha256(json.dumps(template, sort_keys=True).encode("utf-8")).hexdigest()
