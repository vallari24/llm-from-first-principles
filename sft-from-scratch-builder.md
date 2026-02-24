# The Builder's Guide to Fine-Tuning: What SFT Actually Does (And Doesn't)

*When to fine-tune, what to expect, and the edge cases that will bite you — no code required.*

> **Want the implementation details?** This post covers intuitions and decisions. For the full code walkthrough with loss masking, training loops, and architecture diagrams, see the companion post: [SFT from Scratch: The Complete Implementation Guide](sft-from-scratch-engineer.md).

---

**Table of Contents**

1. [Why Your Base Model Isn't a Product](#why-your-base-model-isnt-a-product)
2. [The Three Stages: Format, Quality, Polish](#the-three-stages-format-quality-polish)
3. [What SFT Actually Changes (And What It Doesn't)](#what-sft-actually-changes-and-what-it-doesnt)
4. [How Much Data Do You Need?](#how-much-data-do-you-need)
5. [The Loss Mask: The One Concept You Need to Know](#the-loss-mask-the-one-concept-you-need-to-know)
6. [Before and After: What Changes](#before-and-after-what-changes)
7. [The Cost: Catastrophic Forgetting](#the-cost-catastrophic-forgetting)
8. [When to Fine-Tune (Decision Framework)](#when-to-fine-tune-decision-framework)
9. [The Edge Cases That Will Bite You](#the-edge-cases-that-will-bite-you)
10. [SFT in 2026: What the Industry Learned](#sft-in-2026-what-the-industry-learned)
11. [Your Minimal SFT Checklist](#your-minimal-sft-checklist)

---

## Why Your Base Model Isn't a Product

You downloaded a model. Or you're calling an API. It generates text. But try using it as a product and you'll notice something immediately: it doesn't follow instructions. It doesn't have a persona. It doesn't know when to stop.

Ask a base model "What is love?" and it doesn't answer your question — it continues the text as if it were the next line in a book:

```
You:    "What is love?"
Model:  "What is love? KING RICHARD: I tiny our ymy sumood veng..."
```

A base model is an autocomplete engine. It learned *language* — which words follow which — but not *behavior*. It has no concept of "someone asked me a question and I should answer it." It has no persona, no format, no stopping point.

This is the gap between a language model and a product. Closing it is called **post-training**, and it happens in three stages.

---

## The Three Stages: Format, Quality, Polish

```
Pre-training          →    SFT          →    RLHF / DPO
learns language            learns format      learns quality
(internet text)            (demonstrations)   (preferences)
```

Here's the insight that makes the whole pipeline click: each stage requires *less* human effort but produces *more* targeted improvement. It's a funnel from expensive general data to cheap specific feedback.

**Stage 1: SFT (Supervised Fine-Tuning)** — Teach the model *how to respond*. Humans write ideal responses from scratch: "Given this prompt, here's what a perfect response looks like." The model learns the format — that when someone asks a question, it should produce an answer, not continue the text. Think of it as teaching customer service etiquette to someone who already has domain expertise.

**Stage 2: Reward Model** — Teach the model what "good" means. The model generates multiple responses, and humans *rank* them. "Response A is better than Response B." This is dramatically cheaper than Stage 1 because comparing is easier than creating. A neural network learns to score any response — a machine proxy for human judgment.

**Stage 3: RL (Reinforcement Learning)** — Optimize for quality automatically. No new human data needed. The model generates responses, the reward model scores them, and the model updates toward higher scores. A safety constraint prevents the model from "gaming" the reward model with weird outputs.

| Stage | Who creates the data | Task for humans | Cost |
|---|---|---|---|
| SFT | Humans write responses | "Write the ideal answer" | High — creating is hard |
| Reward Model | Humans rank model outputs | "Which is better, A or B?" | Medium — comparing is easy |
| RL (PPO) | No humans needed | Automated loop | Low — compute only |

DPO (Direct Preference Optimization) collapses Stages 2 and 3 into one step — it uses the same comparison data but trains the model directly, skipping the reward model and RL entirely. Simpler pipeline, same core idea: learn from human preferences.

**This post focuses on Stage 1: SFT.** It's the foundation — without it, the model can't respond to instructions at all, and the later stages have nothing to refine.

---

## What SFT Actually Changes (And What It Doesn't)

This is the most important thing to understand about SFT: **it teaches format, not knowledge.**

The model already knows things. During pre-training, it read billions of words — Wikipedia, textbooks, code, conversations. It has the knowledge. What it lacks is the *manner* of responding.

Think of it this way: you've hired a brilliant researcher who has read every book in the library. They know the answers. But when a customer walks up and asks a question, the researcher starts reciting a random passage from a related textbook instead of answering the question directly. SFT is the customer service training — it teaches the researcher to listen to the question, formulate a response, and deliver it in a helpful format.

**What SFT changes:**
- The model learns to recognize "someone is asking me something" and respond accordingly
- It adopts a consistent persona and tone (whatever the training data demonstrates)
- It learns when to stop talking (the end-of-turn signal)
- It follows the structured conversation format — user turn, then assistant turn

**What SFT doesn't change:**
- The model's knowledge — it knows the same facts before and after
- The model architecture — it's the exact same neural network, same layers, same size
- The quality of reasoning — a mediocre answer and a brilliant answer get treated identically during training

The architecture point surprises people. SFT doesn't add a "chat module" or a "response generator." The model is the exact same transformer — same attention layers, same everything. The only change is that a few special tokens are added to the vocabulary (markers for "user turn starts here," "assistant turn starts here," "turn is over") and the weights get slightly adjusted through training.

The weights don't get replaced. They get *nudged*. SFT uses a lower learning rate than pre-training — small adjustments that teach the new behavior without destroying existing knowledge.

---

## How Much Data Do You Need?

Less than you think, if it's good.

The [InstructGPT](https://arxiv.org/abs/2203.02155) result that surprised everyone: **13,000 examples was enough** to teach a 175-billion-parameter model to follow instructions. That's a tiny dataset for a model that size. But SFT isn't teaching new knowledge — the model already learned that from pre-training on the internet. SFT is teaching a *behavior*: "when someone asks you something, respond helpfully in this format."

Behaviors are learned from fewer examples than knowledge.

| Goal | Dataset size | Why |
|---|---|---|
| Teach the chat format | ~100-500 examples | The model just needs the structural pattern |
| Teach a specific persona or domain | ~1K-5K examples | More variety needed to generalize |
| Production-quality instruction-following | ~5K-50K (filtered) | Diversity across task types matters |

**Quality matters more than quantity.** The ~40 labelers who created OpenAI's [InstructGPT](https://arxiv.org/abs/2203.02155) dataset were carefully screened for writing quality, sensitivity to bias, and agreement with each other. Their instructions had three priorities, in order: be **helpful**, be **truthful**, be **harmless**. These ~40 freelancers' writing style defined the behavior of a model used by hundreds of millions of people.

One high-quality example is worth more than a hundred noisy ones. The model will imitate your data — if your examples are mediocre, the SFT model will be mediocre. This is not something more data fixes.

**The 2025-2026 shift: pruning.** The industry learned that more data can actually hurt. The modern approach: generate a large synthetic dataset using a stronger model, then **throw away 50-95% of it**, keeping only the examples the model finds hardest. Meta, DeepSeek, and others all converged on this: generate a lot, filter ruthlessly, keep only what pushes the model's boundaries.

---

## The Loss Mask: The One Concept You Need to Know

If there's one technical concept worth understanding about SFT, it's the **loss mask**. It's the single mechanism that makes SFT different from pre-training.

Here's the intuition: during training, the model reads the entire conversation — both the user's message and the assistant's response. But it's **only graded on the assistant's response.**

Think of a student taking a test. They read the question (not graded) and write an answer (graded). The reading is necessary — without it they can't answer — but only the answer counts.

![Loss mask visualization — gray tokens are masked, blue tokens are trained](diagrams/sft_loss_mask.png)

In the diagram above, the gray tokens (the user's message) are "masked" — the model reads them but receives no training signal from them. The blue tokens (the assistant's response) are what the model is actually learning to produce.

The training loop, in plain English:

```
for each training example:
    1. Feed the whole conversation to the model
    2. The model makes predictions at every position
    3. Ignore predictions for the user's tokens (mask them)
    4. Grade only the assistant's token predictions
    5. Update the model to get better at producing assistant responses
```

**Why does this matter?** Without masking, half the training signal would go toward teaching the model to write user messages — which is useless, since users write their own messages. Masking focuses 100% of the learning on the only part that matters: generating good responses.

![With vs without loss masking](diagrams/sft_masking_comparison.png)

Without masking, the model sometimes starts generating user messages in its responses — it learned to produce the entire conversation, not just the assistant's part. With masking, convergence is cleaner and the model stays in "response mode."

---

## Before and After: What Changes

The behavioral shift is immediate and visible:

```
┌──────────────────────┬──────────────────────────────────────────────────┐
│ Prompt               │ Base model (autocomplete)  → SFT model (chatbot)│
├──────────────────────┼──────────────────────────────────────────────────┤
│ "Write a greeting"   │ "Write a greeting do       → "Good morrow to    │
│                      │  sther and LUCHENTT..."       thee, noble friend!"│
├──────────────────────┼──────────────────────────────────────────────────┤
│ "Who are you"        │ "Who are you, fares be     → "A humble player    │
│                      │  a that the kin..."           upon this stage."   │
├──────────────────────┼──────────────────────────────────────────────────┤
│ "Speak of love"      │ "Speak of love be dond     → "Love is a smoke    │
│                      │  and KING wose..."            raised with sighs." │
└──────────────────────┴──────────────────────────────────────────────────┘
```

The base model treats every prompt as text to continue — it keeps writing as if it's the next line of a book. The SFT model recognizes the conversation format and switches into response mode. Same model, same size — the only difference is fine-tuning with loss masking.

The more important result is **generalization**. The model above was trained on just 40 examples. But when given prompts it had never seen:

```
"Tell a joke"         (not in training data)  →  "A fool with wit is wiser than a sage!"
"Describe the moon"   (not in training data)  →  "The moon doth rise upon the sleeping world."
```

It didn't memorize 40 question-answer pairs — it learned the *pattern*: when prompted, switch from reading to responding. That's what you're buying with SFT: the format, the behavior, the pattern. Not the answers — the model already had those.

---

## The Cost: Catastrophic Forgetting

SFT isn't free. When you fine-tune a model to follow instructions, some of its original knowledge gets slightly degraded. This is called **catastrophic forgetting** — the model's weights can't fully serve both the old purpose (generating coherent text) and the new purpose (following instructions) without some compromise.

How bad is it in practice?

| Factor | Naive approach | Production 2026 |
|---|---|---|
| Parameters updated | 100% of the model | ~1% (using LoRA adapters) |
| Training passes over data | ~25 | 1-2 |
| Original data mixed in during training | No | Yes |
| Forgetting | +37% worse | ~1-3% worse |

Production models keep forgetting to **1-3%** using techniques like:

- **LoRA** — only update ~1% of the model's parameters, leaving the rest frozen. The original knowledge is literally unchanged.
- **Data mixing** — include some original training data in each SFT batch, so the model "rehearses" old knowledge while learning new behavior.
- **Lightweight SFT, heavy RL** — do the minimum SFT needed to teach the format, then use reinforcement learning (which has built-in constraints against forgetting) for quality improvement.

**What this means for you:** if you're fine-tuning a model and notice it's getting worse at its original tasks, you're probably training too aggressively. Use LoRA, train for fewer epochs, and mix in original data. These are standard practices — your ML team will know how to apply them.

---

## When to Fine-Tune (Decision Framework)

Not every problem needs SFT. Here's a decision framework:

**Use SFT when you need:**

- **Consistent persona** — the model should always respond in a specific voice, style, or character. Prompting works for simple cases, but SFT locks it in reliably.
- **Specific response format** — structured JSON outputs, consistent length, particular formatting that prompting can't reliably achieve.
- **Domain-specific behavior** — the model should handle medical questions differently from casual chat, or should always use industry-specific terminology.
- **Tool-use patterns** — teaching the model to call your APIs, search your knowledge base, or execute code instead of guessing.

**Don't use SFT when:**

- **You need better quality** — SFT teaches format, not quality. If the model follows instructions but gives mediocre answers, you need RLHF/DPO, not more SFT.
- **You need up-to-date knowledge** — SFT can't teach the model new facts efficiently. Use RAG (retrieval-augmented generation) to inject current information at runtime.
- **You need one-off customization** — if you just need the model to behave differently for one use case, try prompting first. A good system prompt is cheaper and more flexible than fine-tuning.
- **You need factual accuracy** — SFT doesn't fix hallucination. It can teach the model to say "I don't know" in some cases, but for reliable facts, you need retrieval or verification systems.

**The prompting-first rule:** always try prompting and system messages before fine-tuning. If that works — stop. SFT is more expensive, less flexible, and introduces forgetting. Only fine-tune when prompting demonstrably can't achieve what you need.

### Failure modes as product symptoms

| What you're seeing | What's likely wrong | What to do |
|---|---|---|
| Model ignores your instructions | Format isn't learned | More SFT data with diverse prompts |
| Model follows format but answers are mediocre | SFT working, quality isn't its job | Move to RLHF/DPO for quality |
| Model sounds different on different turns | Inconsistent training data | Audit data for contradictions in persona |
| Model makes things up confidently | Hallucination — SFT doesn't fix this | Add retrieval (RAG), or tool-use training |
| Model forgot how to do things it used to do well | Catastrophic forgetting | Use LoRA, reduce training epochs, mix in original data |
| Model sometimes writes the user's message | Loss masking is broken | Technical bug — have your ML team check the mask |

---

## The Edge Cases That Will Bite You

### Hallucination: your model will confidently make things up

This is the most important edge case for product builders. Ask the model about a fictional person — "Who is Orson Kovacs?" — and it won't say "I don't know." It'll generate a confident, plausible-sounding biography:

```
"Orson Kovacs was a Hungarian-born physicist who contributed
 to early quantum mechanics research at the University of..."
```

Every word is invented. The model has never seen this name. But it's seen thousands of similar patterns — `"[Name] was a [nationality] [profession] who..."` — and SFT taught it to respond helpfully and confidently. So it does.

**Why SFT can't fix this:** SFT trains the model to produce responses that match the training examples, token by token. Nothing in the training process penalizes factual incorrectness — it only rewards predicting the right next word. If your training data doesn't include "I don't know" responses, the model never learns to produce them.

**What works instead:**

- **Retrieval (RAG)** — give the model access to a knowledge base at runtime. Instead of generating facts from memory, it looks them up. This is the most reliable approach for factual accuracy.
- **Refusal training** — include "I don't know" examples in your SFT data. The [Llama 3](https://arxiv.org/abs/2407.21783) team automated this: they probed the model with thousands of questions, used a stronger model to check correctness, and generated refusal examples for questions the model consistently got wrong. No human writes a single "I don't know" response.
- **Tool use** — teach the model to search before answering. Instead of inventing a biography, it learns to emit a search query, wait for results, then summarize real information. This is taught through SFT with special tokens that trigger system actions.

### Knowledge of self: how does your model identify itself?

A base model doesn't know its own name. It doesn't know who made it, what it can do, or what today's date is. This is a product decision you need to make:

- **Bake it into training** — include examples like `"What's your name?" → "I'm [YourProduct], made by [YourCompany]."` Consistent but inflexible — rename the product and you need to retrain.
- **System message at runtime** — prepend a hidden instruction every conversation: `"You are [YourProduct]. Today's date is Feb 23, 2026."` Flexible and cheap, but the model might ignore it if the instructions conflict with its training.
- **Both (what production models do)** — SFT teaches the model to *follow* system messages; the system message provides the specifics at runtime. Best of both worlds.

### Jagged intelligence: brilliant at hard things, broken at easy things

Here's the edge case that will confuse your users most: **the same model that writes better legal briefs than most lawyers cannot count the letters in "strawberry."**

This isn't a bug. It's a structural property of how these models process text.

The model processes text in chunks called "tokens." The word "strawberry" becomes something like `["str", "awber", "ry"]` — three tokens. The model literally cannot see individual characters. When asked to count letters, it's like asking someone to count brushstrokes in a painting they can only see from across the room.

**Why this matters for your product:**

- **Don't promise character-level tasks** — spelling, counting, anagrams, reversing strings. The model will get these wrong in ways that look careless to users.
- **Don't promise precise arithmetic on large numbers** — "1847293" becomes multiple tokens with no digit boundaries. Small numbers are usually fine; large or unusual numbers aren't.
- **Do use tool delegation** — the fix is teaching the model to call a code interpreter for math and string operations, or a search engine for facts. This is standard SFT practice. The model doesn't need to count — it needs to write a one-line program that counts.

The broader pattern: know your model's structural blind spots and design your product around them. Don't fight the architecture — delegate to tools where the architecture is weak.

### The thinking problem: why chain-of-thought costs you money

Every token the model generates costs the same amount of compute. There's no "think harder" mode — the model gets one pass through its layers per token, whether it's generating "the" or solving a differential equation. The only way to get more compute is to generate more tokens.

This is why "chain of thought" works — when the model writes out intermediate reasoning steps, each step gives it another pass through its neural network. The thinking tokens aren't explanations for your benefit. They're *computation*.

But this has a product cost: the model can't judge how much thinking a question needs. "What's the capital of France?" doesn't need 500 reasoning tokens, but the model might generate them anyway. You're paying for tokens — and some of those tokens are wasted on overthinking easy questions.

**What you can do:** set token budgets, use models with adaptive thinking (where available), and monitor your average tokens-per-response to catch inefficiency.

---

## SFT in 2026: What the Industry Learned

The core mechanism hasn't changed — SFT is still "fine-tune on conversations with loss masking." What changed dramatically is the data pipeline around it.

In 2022, OpenAI hired ~40 contractors to write 13K responses by hand. In 2026, the pipeline looks like this:

```
Human seed examples (~hundreds)
       ↓
Synthetic expansion via a stronger model (~millions)
       ↓
Quality filtering via model-as-judge (~thousands survive)
       ↓
Lightweight SFT (1-2 training passes)
       ↓
RL for quality improvement (the new heavy lifting)
```

**Humans moved from data creators to system designers.** They write the principles and the seed examples. Models generate the actual training data. Other models judge its quality. The surviving examples — typically 1-5% of what was generated — are what the model actually trains on.

Key findings across labs:

- **NVIDIA ([Nemotron](https://arxiv.org/abs/2406.11704))**: ~98% synthetic data in their SFT mix. Only ~20K human-annotated examples out of 18M+ total.
- **DeepSeek ([R1](https://arxiv.org/abs/2501.12948))**: Synthetic reasoning traces actually **outperform human-expert** ones on downstream tasks.
- **Meta ([Llama 3](https://arxiv.org/abs/2407.21783))**: Pruned 50-95% of SFT examples, keeping only the hardest. Fewer examples = better results.
- **Anthropic (Claude)**: Uses "[Constitutional AI](https://arxiv.org/abs/2212.08073)" — the model critiques and revises its own outputs against written principles. Humans write the constitution, not the training data.

### What the labs agree on

| Principle | Why it matters for you |
|---|---|
| Quality > quantity | Don't collect more data — collect better data, or filter harder |
| SFT should be lightweight | Just enough to teach format. Push quality to RLHF/DPO |
| Pruning beats hoarding | The model learns more from 5K hard examples than 100K easy ones |
| Synthetic data works | A stronger model can generate training data for a weaker one |
| Humans design, models execute | Your team's job is the pipeline design, not writing thousands of examples |

### The lab philosophies

| Lab | SFT approach | Key innovation |
|---|---|---|
| **Meta** | Less is more — minimal SFT, heavy RL | Model-as-judge pruning |
| **DeepSeek** | Distillation at scale | Stronger model teaches weaker model to reason step-by-step |
| **Anthropic** | Self-improvement | Model critiques its own outputs against a constitution |
| **NVIDIA** | Massive synthetic pipelines | 18M+ examples with automated quality filtering |
| **OpenAI** | Reasoning-first | Deep chain-of-thought integration (details sparse) |

The takeaway: if you're building on top of a foundation model, you don't need thousands of human-written examples. You need a small set of excellent seed examples, a strong model to expand them, and a quality filter to keep only what matters.

---

## Your Minimal SFT Checklist

If you're planning an SFT project, here's the sequence to hand to your ML team:

1. **Start with a pre-trained model** that generates coherent text in your domain. Don't try to teach knowledge through SFT — start with a model that already knows what it needs to know.

2. **Write 50-100 examples by hand** that demonstrate exactly the behavior you want. These set the tone, format, and persona. Every example should be one you'd be proud to ship as a product response.

3. **Expand to 500-5K examples** using a stronger model to generate more examples in the same style. Or collect them carefully from domain experts.

4. **Filter aggressively.** Remove duplicates, low-quality responses, anything that contradicts your persona, anything off-format. If in doubt, cut it.

5. **Train for 2-3 passes** over the data at a conservative learning rate. Use LoRA to minimize forgetting.

6. **Evaluate on unseen prompts** — not loss numbers. Generate responses to 20+ prompts the model hasn't seen and read them. Are they on-format? On-persona? Coherent?

7. **Pick the best checkpoint** based on manual evaluation, not training metrics. Lower loss doesn't always mean a better model.

8. **Check for forgetting** — make sure the model can still do the things it could do before fine-tuning. If degradation exceeds ~5%, reduce training epochs or switch to LoRA.

This gets you a model that follows instructions in your format. Quality improvement — making responses *good*, not just *formatted* — is the job of RLHF or DPO, covered in a follow-up post.

---

**References:**
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, Ouyang et al. 2022)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Meta 2024)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (DeepSeek 2025)
- [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704) (NVIDIA 2024)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (Anthropic 2022)

*This is part of a series on building LLMs from scratch. For the full implementation with code, see [SFT from Scratch: The Complete Implementation Guide](sft-from-scratch-engineer.md). [Previous: Building a Language Model from Scratch](building-gpt-from-scratch.md).*
