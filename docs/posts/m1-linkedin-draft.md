# LinkedIn post draft (M1). Elias edits before posting.

I'm learning to fine-tune open AI models hands-on, and I'm publishing every model I make. Here is the first one.

The job: read two short memories about a person and say whether they contradict each other. "Sam's favourite season is autumn" and "Sam's favourite season is spring" clash. "Sam moved to Bergen" written after "Sam lives in Oslo" is an update, not a clash. A memory app needs this to keep its facts straight.

What I did:
- Wrote a 200-pair test before any training, and locked it.
- Tested three untrained open models. A bigger 8B model scored well, until I swapped the order of the two memories. Then it raised false alarms on most updates.
- Fine-tuned a smaller 4B model (Qwen3-4B) on about 1,500 made-up examples, using Unsloth on rented GPUs from Modal. Training took about 30 minutes and cost about $1.

Result on the locked test: when it says "contradiction" it is right 94% to 98% of the time, and it catches 94% to 96% of real contradictions, whatever the order and with or without dates. It runs on my laptop in under a second per answer.

What I learned: the test set matters more than the training. Two independent checks caught problems in my data before they could teach the model bad habits.

Next: improving the same model with reinforcement learning.

Model, data and code: [links]
