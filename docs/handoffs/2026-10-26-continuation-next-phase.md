# Continuation: explore the next phase

**Where:** the Cowork project "Fine Tuning and RL training", **new chat**. The folder `~/code/model-training` must be connected.

Copy everything below the line into that chat.

---

This is a new session of the "Fine Tuning and RL training" project. M1 is done and published. **The job of this session is to explore what the next phase should be.** Explore means: lay out the options, angles and tradeoffs. Do not pick one or start building unless I ask.

**Open the session the usual way.**
1. Read, in this order: `STATE.md`, `ROADMAP.md`, `DECISIONS.md` (especially D-001, D-003, D-015, D-023 to D-034), `LESSONS.md` (especially the 2026-10-23 "why it works" note and the 2026-10-26 close-out lessons), `milestones/M1-contradiction-judge/SPEC.md` and `results/m1/gguf-v1/summary.md`.
2. Tell me in one to three plain sentences where things stand.

**Where things stand (check it against the files; the files win):**
- **M1 is done.** A Qwen3-4B contradiction judge was fine-tuned with QLoRA (Unsloth on Modal) for under $2 of GPU time in total (`runs/RUNLOG.md`). It runs on my Mac in Ollama as `m1-judge:v1`, and it is public on Hugging Face (`eskora`) and on GitHub (`e-skora/model-training`).
- **Its scores on the frozen test set:**
  - When it says "contradiction", it is right 94% to 98% of the time.
  - It catches 94% to 96% of real contradictions.
  - It gave 0 broken answers.
  - It is weakest without dates, and it sometimes dodges with `unclear`.
- **Aeverum** (my memory product) may use it as its night-pass judge. Aeverum's own step-12 benchmark decides that. Its step-12 preparation notes are listed in `STATE.md` under "Open items".

**My standing plan** (D-001, D-003): learn real training hands-on, to start a company or get hired as a forward-deployed engineer. I wanted to finish fine-tuning quickly so I can move on to reinforcement learning (RL). The later steps I named were Together AI (managed fine-tuning, the way companies buy it) and then Tinker by Thinking Machines (RL practice). D-015 proposed M2 as RL (GRPO) on the same judge with Unsloth on Modal; that is a default, not my decision.

**What to explore (add angles I haven't thought of):**
- **RL on the same judge with Unsloth on Modal (GRPO):**
  - what the reward would be
  - what it could fix (recall without dates, `unclear` dodges)
  - cost and time
  - what I would learn
- **RL on a new task instead,** one where reasoning matters more.
- **Tinker for RL, now instead of later:** what it is today, its pricing and access, and how it differs from doing RL myself. Check its current docs before saying anything.
- **Together AI managed fine-tuning:** the "how companies buy it" experience, compared with what I did by hand. Check its current docs and pricing.
- **Doing M1's job again for another use case** that a company would pay for, as practice for forward-deployed work.
- **What Aeverum still needs,** and whether a phase should serve it, for example the `unclear` scoring or a v2 judge.
- **For each option:**
  - what I'd learn
  - what I'd have to show at the end
  - rough cost and time
  - how it fits my goals

**How to work with me** (also in the project instructions):
- Plain language a child could follow. Define new terms.
- Concise, with no hype and no summary at the end.
- Check any tool's current pricing or screens against its docs, and say so if you couldn't.
- Ask only what is mine to decide, in one batch in chat, never in pop-up boxes.
- No em dashes in anything public.
- Nothing important lives only in chat. When we settle the phase, write it into `ROADMAP.md`, `DECISIONS.md` and a new milestone `SPEC.md`.
