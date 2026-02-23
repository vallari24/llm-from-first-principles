# The Mental Model for AI

Everyone's building with AI. Few know what a token is.

---

Every product team is shipping AI features. Founders are choosing between models at dinner. PMs are debating context windows in product reviews. Developers are wiring up LLM APIs over the weekend.

But most builders don't have a mental model for what's actually happening under the hood. They're making architectural decisions — which model, how much context, what temperature — based on vibes and vendor marketing.

You don't need to build a transformer to make good decisions. But you do need the right mental models. This post gives you those.

If you're a PM writing an AI feature spec, a founder choosing between Claude and GPT, or a developer integrating an LLM into your product — this is for you. No code, no math, just the intuitions that lead to better decisions.

By the end of this post, you'll understand how these models actually process text, why they sometimes make things up, what drives the cost of every API call, and when to use an agent vs. a simple workflow. These aren't academic concepts — they're the decisions you'll face in your next sprint planning.

---

## Tokens: The Model Doesn't Read Words

An LLM doesn't see words. It sees **tokens** — chunks of text, each mapped to a number. Think of it like **Scrabble tiles**: your sentence gets snapped into pieces, and each piece has a number on the back. That's what the model actually works with.

Common words like "the" or "hello" fit into a single tile. Rare or long words get split across multiple tiles. "Understanding" might be three tiles: "under" + "stand" + "ing." A typo like "hellooo" could be five or six tiles because the tokenizer has never seen it before.

Here's the thing that trips people up: **different providers use different tile sets.** The same sentence produces different token counts depending on the model.

```
"Tokenization is surprisingly tricky"

  OpenAI (GPT-4o):     5 tokens
  Anthropic (Claude):   6 tokens
  Google (Gemini):      5 tokens

Same text. Different tile sets. Different counts.
```

A useful rule of thumb: **1 token ≈ ¾ of a word.** So 128k tokens is roughly a 200-page book. A 200k token window? About 300 pages. These are approximations, but they're close enough for product decisions.

**So what for builders:** Tokens are the unit of everything — cost, capacity, speed. Your API bill is measured in tokens. Your context window is measured in tokens. And different providers tokenize differently, so "128k tokens" from OpenAI and "200k tokens" from Anthropic aren't as directly comparable as they look. When you're estimating costs, use each provider's tokenizer to count — don't assume token counts transfer across models. Prompt design and model selection are engineering decisions, not vibes.

---

## The Context Window: How Much the Model Can See

The context window is the total amount of text the model can work with at once. Think of it as **a desk, not a filing cabinet.** Everything the model needs has to fit on the desk — your message, the conversation history, the system instructions, and the response it's generating. There's no filing cabinet to pull from later. If it's not on the desk, it doesn't exist.

Here's what's competing for space on that desk:

```
┌──────────────────────────────────────┐
│           THE DESK                   │
│                                      │
│  ┌────────────────────┐              │
│  │ System instructions │  "Always    │
│  │                     │  respond    │
│  │                     │  in French" │
│  └────────────────────┘              │
│  ┌────────────────────┐              │
│  │ Conversation        │  All prior  │
│  │ history             │  messages   │
│  └────────────────────┘              │
│  ┌────────────────────┐              │
│  │ Your latest         │  What you   │
│  │ message             │  just asked │
│  └────────────────────┘              │
│  ┌────────────────────┐              │
│  │ The model's         │  What it's  │
│  │ response            │  generating │
│  └────────────────────┘              │
│                                      │
│  Everything must fit. No exceptions. │
└──────────────────────────────────────┘
```

Different models have different desk sizes:

- **GPT-4o** — ~128k tokens (~200 pages)
- **Claude** — ~200k tokens (~300 pages)
- **Gemini** — ~1M+ tokens (~1,500 pages)

Bigger desk = more you can spread out. But a bigger desk has a problem of its own.

### The middle gets blurry

Imagine reading a 300-page document in one sitting. You'd remember the introduction clearly. You'd remember the last few pages. But page 147? Blurry at best.

LLMs have the same problem. Research shows they pay the most attention to the **beginning** and the **end** of the context, and significantly less to the middle. It's a U-shaped pattern:

```
Attention across the context window:

  START ████████░░░░░░░░░░░░░░░░░░░░████████ END
        ↑ strong                    strong ↑
                 ↓ the middle ↓
                    blurry
```

This isn't a bug that bigger windows fix. It actually gets *worse* as the window grows. More pages on the desk means the middle pages get even less attention.

In a product context, this means the model remembers your first few messages and your most recent messages well — but that important instruction you gave 20 messages ago? It might effectively "forget" it, even though it's right there in the window.

**So what for builders:** Design your product to manage context deliberately. Put critical instructions at the start (system prompt) or the end (latest message) — not buried in the middle. Summarize old conversation history instead of keeping every message. Keep system prompts lean. The developers who get the best results aren't the ones with the biggest context windows — they're the ones who keep their context focused and relevant.

---

## How LLMs Generate Text: One Word at a Time

Here's the part that surprises most people: the model has no plan for the full sentence before it starts writing. It generates **one token at a time,** each time using everything that came before to predict what comes next.

Think of it like **phone autocomplete on steroids.** Your phone predicts the next word based on what you've typed. An LLM does the same thing — but instead of looking at your last few words, it looks at the entire context window. Every token on the desk.

For each position, the model produces a probability for every possible next token — like rolling a **weighted die.** Common continuations get big faces on the die. Unlikely ones get tiny faces.

```
Prompt: "The capital of France is"

The model's weighted die:

  ┌──────────┬──────────┬──────────┬──────────┐
  │          │          │          │          │
  │  Paris   │  the     │  known   │  Lyon    │
  │  92%     │  3%      │  2%      │  0.1%    │
  │          │          │          │          │
  └──────────┴──────────┴──────────┴──────────┘
  (big face)  (small)    (small)    (tiny)

  → Rolls → "Paris"

Then: "The capital of France is Paris"
  → Rolls again → "."

Then: "The capital of France is Paris."
  → Rolls again → [DONE]
```

This is why LLM responses **stream in** word by word — it's not buffering; it's literally generating one token, then using that to generate the next, then the next. And it means the model can't "go back and fix" what it already wrote — once a token is generated, it's committed. The model has to work forward from whatever it's said so far, which is why long outputs sometimes start strong and drift off course.

### Temperature: how adventurous the dice are

The **temperature** setting controls how the model rolls that die. It's the difference between "always pick the most likely word" and "get creative."

```
Prompt: "The sunset looked"

Low temperature (T=0.2):        High temperature (T=1.0):

  "The sunset looked              "The sunset looked
   beautiful over the              like a wound bleeding
   ocean as the sky                tangerine and violet
   turned golden."                 across a trembling sky."

  → Safe, predictable             → Creative, surprising
  → Great for data extraction     → Great for storytelling
```

Low temperature sharpens the die — the big faces get even bigger, small faces disappear. The model almost always picks the most likely token. High temperature flattens the die — unlikely tokens get a real chance, leading to more surprising (and sometimes unhinged) outputs.

**So what for builders:** Temperature is a product decision, not a default you leave alone. Set it low (0.0–0.3) for structured extraction, classification, or anything where consistency matters. Set it higher (0.7–1.0) for creative features, brainstorming, or content generation. If your AI feature sometimes gives wildly different answers for the same input and that's a problem, your temperature is probably too high.

---

## What the Model Knows vs. What It Sees

There are two completely different sources of information an LLM works from, and confusing them causes most product design mistakes.

Think of it like an **employee with training and a briefing document:**

```
┌─────────────────────────┬─────────────────────────┐
│    WHAT IT LEARNED       │    WHAT YOU GAVE IT      │
│    (Training)            │    (Context)             │
│                          │                          │
│  • Vast knowledge        │  • Your specific data    │
│  • Frozen at a cutoff    │  • Current and live      │
│    date                  │  • Limited to the desk   │
│  • Can't be updated      │  • Changes every call    │
│    per-call              │                          │
│                          │                          │
│  Like: 4 years of       │  Like: a briefing doc    │
│  university education    │  handed over before      │
│                          │  a meeting               │
└─────────────────────────┴─────────────────────────┘
```

Training gave the model broad knowledge — grammar, reasoning, facts about the world up to its cutoff date. But that knowledge is frozen. The model doesn't know what happened yesterday unless you tell it.

Context is everything on the desk right now — your message, the system prompt, any documents you included. It's current but limited to what fits.

**Hallucination** happens when the model optimizes for "sounds right" instead of "is right." It's not lying — it's doing exactly what it was trained to do: produce fluent, plausible text. Sometimes that text happens to be wrong. Ask it for a citation and it might invent a paper that doesn't exist — complete with a plausible author name, journal, and year. It's not malicious; it's autocomplete doing what autocomplete does.

**RAG (Retrieval-Augmented Generation)** in one line: find relevant documents first, put them on the desk, *then* ask the question. Instead of relying on the model's frozen training, you give it current, specific information to work from. Your customer support bot doesn't need to "know" your return policy from training — it needs the policy document in its context window, right there on the desk.

**So what for builders:** Don't trust raw LLM output for facts — build verification into your product. RAG isn't optional for accuracy-critical features; it's table stakes. And design for graceful handling of "I don't know" — a model that admits uncertainty is more useful than one that confidently makes things up.

---

## Why AI Costs What It Costs

LLM pricing works like a **metered utility — a water meter.** You pay for what flows through the pipes, measured in tokens.

But here's the key: **output water costs 3–5x more than input water.** That's because input tokens are processed in parallel (the whole prompt at once), while output tokens are generated one at a time, sequentially. The pump works harder on output.

```
           INPUT                          OUTPUT
      (your prompt)                   (model's reply)

    ═══════════╗                    ╔═══════════
    thin pipe  ║   ┌──────────┐    ║  thick pipe
    cheap      ║───│  MODEL   │────║  expensive
    $3/M tokens║   └──────────┘    ║  $15/M tokens
    ═══════════╝                    ╚═══════════
```

A few things that catch builders off guard:

**Conversations get expensive.** Every message in a conversation resends the full history. Message 1 sends 100 tokens. Message 2 sends 200. Message 10 might send 5,000 — the history snowballs. Long conversations cost way more than short ones.

**Prompt caching keeps the pipes warm.** If you send the same system prompt on every call (you probably do), providers can cache it and skip reprocessing. You get up to 90% off the cached portion. Keeping your system prompt consistent across calls isn't just tidy — it's a cost optimization.

**Thinking models charge for scratch work.** Some models (Claude with extended thinking, OpenAI's o1/o3) do internal reasoning before responding. Those "thinking tokens" are output tokens you pay for but never see. A simple question might generate 50 visible tokens but 500 thinking tokens behind the scenes.

**So what for builders:** Model your costs per-feature, not per-month. A chatbot and a document summarizer have completely different cost profiles even on the same model. Consistent system prompts enable caching discounts. Output tokens are where the money goes — design for concise responses where possible. And always check the full usage breakdown in your API responses; the bill is often bigger than you'd expect.

---

## Agents vs. Workflows

This is the most overhyped distinction in AI right now, and getting it right will save you months of engineering.

Think of it as **classical music vs. jazz.**

### Workflows: classical music

A workflow is a composed score. Every note is planned. The LLM fills in specific parts, but your code controls the flow.

```
WORKFLOW (Classical Score):

  Step 1: Extract entities     → LLM call
  Step 2: Look up in database  → Your code
  Step 3: Format response      → LLM call
  Step 4: Return result        → Your code

  Every step is predetermined.
  Your code is the conductor.
```

Use workflows when you know the steps upfront. Data extraction, report generation, form processing, classification pipelines. If you could draw a flowchart without any "it depends" branches, it's a workflow.

### Agents: jazz

An agent improvises. You give it tools and a goal, and it figures out the path. It decides what to do next, observes the result, and adapts.

```
AGENT (Jazz Improvisation):

  ┌──→ Observe (what just happened?)
  │        ↓
  │    Think (what should I try next?)
  │        ↓
  │    Act (call a tool, generate code)
  │        ↓
  └─── Done? ─── No → loop back
                  Yes → return result

  The LLM decides the steps.
  You set the boundaries.
```

Use agents when you *don't* know the steps upfront. Debugging, research, open-ended problem solving — tasks where the next step depends on what you find.

### Most real products are in between

The dirty secret: most production AI is workflows with a thin agent layer on top. Pure agents are expensive, unpredictable, and hard to debug. Pure workflows are rigid. The sweet spot is usually a workflow that uses an LLM at specific decision points.

A "workflow with LLM routing" might use a model to decide which of three fixed paths to take, but each path is predetermined code. A "constrained agent" might loop autonomously but only within a specific set of allowed tools and with a hard iteration limit. Pure workflows and pure agents are the extremes — most shipped products sit somewhere in the middle.

```
WORKFLOW ←─────────────────────────────→ AGENT
Fixed steps       Some LLM decisions     Fully autonomous
Predictable       at branch points       LLM-driven loop
Cheap                                    Expensive
```

**So what for builders:** Default to workflows. They're cheaper, more reliable, and easier to debug. Only reach for agents when the task genuinely requires exploration — when you can't draw the flowchart ahead of time. If someone pitches you an "AI agent" for a task with known steps, they're over-engineering it. Most shipped AI is workflows, not agents, despite what the marketing says. And if you're building an agent, set hard limits — max iterations, timeouts, cost caps. An agent without guardrails is a credit card bill without a limit.

---

## Cheat Sheet: Key Decisions Every AI Builder Should Understand

Save this for your next architecture review.

1. **Pick the right context window for the job.** Bigger isn't always better — larger windows cost more and suffer worse "lost in the middle" effects. Match the window to your actual data needs.

2. **Put critical instructions at the start or end.** The model pays the most attention to the beginning and end of the context. Don't bury important instructions in the middle of a long prompt.

3. **Set temperature per feature, not per app.** Use low temperature (0–0.3) for extraction and classification. Use higher temperature (0.7–1.0) for creative generation. One setting doesn't fit all features.

4. **Model your costs per feature.** A chatbot (long conversations, lots of output) has a completely different cost profile than a classifier (short input, one-word output). Budget accordingly.

5. **Default to workflows, not agents.** If you can draw the flowchart, build a workflow. Only use agents when the task requires genuine exploration. Most production AI is workflows.

6. **Use RAG for anything accuracy-critical.** Don't rely on the model's training data for current or specific facts. Retrieve the relevant documents, put them in the context, then ask.

7. **Manage context like a resource.** Summarize old conversation history. Keep system prompts lean. Start fresh sessions for new tasks. Context is finite and expensive — treat it that way.

8. **Keep system prompts consistent for caching.** Prompt caching can cut input costs by up to 90%, but only if the same prefix is sent every time. Don't dynamically change your system prompt unless you have to.

9. **Design for concise output.** Output tokens cost 3–5x more than input tokens. If your feature can work with a short response, instruct the model to be brief. Your bill will thank you.

10. **Build verification, not trust.** LLMs optimize for plausible, not correct. Any feature where accuracy matters needs a verification layer — whether that's RAG, tool use, or human review.

---

## Where to Go Deeper

These are simplifications. But they're the *right* simplifications — the ones that lead to good decisions even when you don't know the full technical details.

When you're ready to go deeper:

- **"Attention Is All You Need"** — the 2017 paper that started it all. It's more readable than you'd expect.
- **Andrej Karpathy's "Let's build GPT from scratch"** — a 2-hour video where you build a working language model from zero. The best way to make these mental models concrete.
- **Anthropic's "Building Effective Agents"** — the definitive guide to when to use agents vs. workflows, from the team that builds Claude.

These mental models will get you through 90% of the AI product decisions you'll face. For the other 10%, you'll know exactly what questions to ask — and that's the real advantage.

*Subscribe to [Vallari's Newsletter](https://substack.com/@vallarimehta) for the deep-dive series where we crack open the hood and build a transformer from scratch — with code.*