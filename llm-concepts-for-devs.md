# LLM Concepts Every Developer Should Know

Practical concepts for developers building with LLMs — understanding tokens, context windows, and agents vs. workflows.

---

## Tokens: The Currency of LLMs

A neural network can't read text — it only sees numbers. **Tokens** are the numbers. The process of converting text to tokens is called **encoding**; converting back is **decoding**.

A tokenizer splits text into chunks and maps each chunk to a number. The set of all chunks it knows is its **vocabulary**. "Hello World" might become `[15339, 2375]` with one tokenizer, or `[9906, 1917]` with another.

### How tokenizers are trained (BPE)

Most LLM tokenizers use **Byte-Pair Encoding (BPE)**:

1. Start with every individual byte as its own token
2. Find the most frequently occurring pair of adjacent tokens in the training data
3. Merge that pair into a new single token
4. Repeat until you hit your target vocabulary size

After thousands of merges, common words like `" the"` and `"ing"` become single tokens, while rare words get split into pieces.

```
"understanding" with different vocabulary sizes:

  1k vocab:   u·n·d·e·r·s·t·a·n·d·i·n·g  → 13 tokens
  50k vocab:  under·stand·ing              → 3 tokens
  200k vocab: understand·ing               → 2 tokens
```

Bigger vocabulary = fewer tokens per text = more text fits in the context window. But bigger vocabulary = larger embedding table = more parameters to train. Real LLMs land around 32k–100k tokens as the sweet spot.

### Input tokens vs. output tokens

When you call an LLM API, two meters are running:

- **Input tokens** — everything you send: your prompt, the system prompt, conversation history, tool definitions. All of it gets tokenized and counted.
- **Output tokens** — everything the model generates in response.

These are billed at **different rates**. Output tokens typically cost 2–5× more than input tokens because generation is sequential — the model produces one token at a time, while input tokens are processed in parallel in a single forward pass.

```
Example (Claude Sonnet):
  Input:  $3 per million tokens
  Output: $15 per million tokens

Your API call:
  System prompt + user message = 500 input tokens
  Model reply                 = 200 output tokens

  Cost = (500 × $3/1M) + (200 × $15/1M)
       = $0.0015 + $0.003
       = $0.0045
```

### Why "Hello World" costs more tokens than you'd expect

You send `"Hello World"` to Claude. The API reports 11 input tokens. Your text is 2 words — where did the rest come from?

**The system prompt.** Every API call includes a system prompt — instructions telling the model how to behave. Even if you don't set one explicitly, the provider may inject a default. Tool definitions, if you're using function calling, also add tokens. The input token count reflects *everything* the model sees, not just what you typed.

This is why checking the `usage` object in the API response matters — it tells you the real count.

### Why the same text has different token counts across providers

Send `"Hello World"` to OpenAI — 3 tokens. To Google — 4 tokens. Different number for the same text.

Every provider trains **their own tokenizer** with their own vocabulary. They all use BPE variants, but the specific merge rules differ because they were trained on different data:

| Provider | Tokenizer |
|----------|-----------|
| OpenAI | tiktoken |
| Anthropic | Proprietary BPE |
| Google | SentencePiece |
| Meta (Llama) | SentencePiece BPE |

Think of it like compression — ZIP and GZIP use similar ideas but produce different file sizes. Same input, different codebooks, different token counts. **Don't assume token counts transfer across providers.** Use each provider's tokenizer or counting API to estimate costs accurately.

### Prompt caching

If you're sending the same system prompt on every API call (which you usually are), **prompt caching** lets the provider skip re-processing those tokens. The first call pays full price to create the cache; subsequent calls with the same prefix get a steep discount — often 90% off for the cached portion.

This matters when your system prompt + tool definitions are thousands of tokens, which they often are in agent-style applications.

### Thinking tokens

Some models (Claude with extended thinking, OpenAI's o1/o3) produce **thinking tokens** — internal reasoning the model does before its visible response. These are output tokens you get billed for, even if you never see them directly. A simple question might generate 50 visible output tokens but 500 thinking tokens behind the scenes. Always check the full usage breakdown in your API response.

### The practical takeaway

Three things to watch:

1. **System prompts and tools inflate input tokens.** Your "short prompt" might be 80% hidden context.
2. **Output tokens cost more.** If you can get the same answer in fewer tokens (structured output, concise instructions), do it.
3. **Different providers tokenize differently.** Don't assume token counts are portable across providers.

---

## The Context Window: The Most Important Constraint

The context window is the **total number of tokens the model can see at once** — input tokens plus output tokens combined. Every model has a hard limit:

```
GPT-4o:          128k tokens
Claude Sonnet:   200k tokens
Gemini 2.0:      1M+ tokens
```

This is a hard wall. Exceed it and the API returns an error. Hit it mid-generation and the response gets cut off incomplete.

### What fills the context window

It's not just your message. Everything counts:

```
┌─────────────────────────────────┐
│         CONTEXT WINDOW          │
│                                 │
│  System prompt         ~500     │
│  Tool definitions      ~2,000   │
│  Conversation history  ~varies  │  ← INPUT TOKENS
│  Your latest message   ~100     │
│                                 │
│  ─────────────────────────────  │
│  Model's response      ~varies  │  ← OUTPUT TOKENS
│                                 │
└─────────────────────────────────┘
```

In a coding agent like Claude Code, tool definitions and MCP server schemas can eat 10k–20k tokens before you've said a word. Conversation history grows with every exchange. That 200k window fills up faster than you'd think.

### The "Lost in the Middle" problem

Bigger context windows don't mean the model uses all the information equally.

Research from Stanford ([Liu et al., 2023](https://cs.stanford.edu/~nfliu/papers/lost-in-the-middle.arxiv2023.pdf)) showed that LLMs exhibit a **U-shaped attention pattern**: they attend most to information at the **beginning** and **end** of the context, and significantly less to information **in the middle**.

```
Attention strength across the context window:

HIGH ████████░░░░░░░░░░░░░░░░░░░░░░░░████████ HIGH
     ↑ beginning                          end ↑
                   ↓ middle ↓
LOW         ░░░░░░░░░░░░░░░░░░░░              LOW
```

In a long conversation, the model remembers your first messages and your most recent messages well, but can effectively "forget" important instructions or context from 20 messages ago — even though those tokens are right there in the window.

This isn't a bug that bigger windows fix. Models with larger windows still show this pattern — it gets *more* pronounced as the window grows.

### Why the U-shape? It's two different effects

The attention mechanism *can* put high weight anywhere — nothing in the math prevents attending to the middle. The U-shape comes from how position encodings and training interact.

**Why the end gets high attention: positional decay.** Modern models use rotary position embeddings (RoPE), where the attention score between two tokens **decays with distance**. The token being generated is always at the end of the sequence. So nearby tokens — recent context — get a natural positional boost.

**Why the beginning gets high attention: learned bias.** During pretraining, the model sees billions of examples where the first tokens carry structural information — document titles, topic sentences, system prompts. The model learns through training: "the beginning usually tells me what this whole thing is about." This isn't from the attention formula — it's a bias baked into the model's weights.

**Why the middle loses:** it's far from the current token (no positional boost) AND doesn't have a learned structural role (no training bias). The model *can* attend to position 50,000 if the content there produces a strong enough score — and sometimes it does. But statistically, nothing pushes the model to favor the middle, so it doesn't.

```
BEGINNING: high attention  ←  learned bias ("start = important structure")
MIDDLE:    low attention   ←  far from current token + nothing special learned
END:       high attention  ←  positional proximity (RoPE decay)
```

### Why this matters for AI coding agents

In a long coding session:

- Your initial instructions ("always use TypeScript", "don't modify tests") drift out of the high-attention zone
- Tool outputs (file contents, error logs) pile up in the middle
- The model starts repeating mistakes or forgetting constraints you set earlier

This is why experienced users of coding agents get better results by **keeping context lean** rather than maximizing it.

### Practical strategies

**Clear vs. compact:**
- **Clear** (e.g. `/clear` in Claude Code) — wipes the conversation, starts fresh. Use when switching tasks entirely.
- **Compact** (e.g. `/compact` in Claude Code) — summarizes the conversation into fewer tokens while preserving key context. Use when continuing the same task but history is getting long.

**Prevent bloat at the source:**
- Keep system prompts concise — every token there is a token you lose for actual work
- Be selective with MCP servers — each one adds tool schemas to your context
- Point agents to specific functions or line ranges instead of pasting entire files
- Start new conversations for new tasks rather than continuing a 50-message thread

**Work with the U-shape:**
- Put critical instructions in the system prompt (beginning — high attention)
- Repeat important constraints in your latest message (end — high attention)
- If the model "forgets" something, re-state it — moving it to the end refreshes it

### The fundamental tradeoff

More context = more information available = better potential answers. But more context also = higher cost, slower responses, and worse attention to any single piece of information.

The developers who get the best results aren't the ones with the biggest context windows — they're the ones who keep their context **focused and relevant**.

---

## Agents vs. Workflows: Know Which You're Building

When OpenAI launched AgentKit, they called it a tool for "building agents." But look at their examples and you see predetermined steps with fixed logic — not agents. This distinction matters because agents and workflows solve fundamentally different problems, and confusing them makes it harder to reason about trade-offs.

### The definitions (from Anthropic's "Building Effective Agents")

Anthropic's framework has become the standard way to talk about this:

- **Workflows** — LLM calls orchestrated through **predefined code paths.** Your code decides what happens and in what order.
- **Agents** — LLMs that **dynamically direct their own processes.** The model decides what to do next based on what it observes.

The key question: **who decides when to stop — your code or the LLM?**

In a workflow, your code runs step 1, then step 2, then step 3, then returns. The path is determined before the LLM ever runs. In an agent, the LLM runs in a loop — observe, decide, act — and the loop keeps going until the LLM decides it's done (or hits a safety limit you set).

### Workflows: classical music

A workflow is like a classical score — every note planned, rehearsed, optimized for a known outcome.

```
User input
    ↓
Step 1: Extract entities (LLM call)
    ↓
Step 2: Look up in database (code)
    ↓
Step 3: Format response (LLM call)
    ↓
Return result
```

The path is fixed. The LLM fills in specific steps, but your code controls the flow. If step 2 fails, your code decides what to do — not the LLM.

**Use workflows when:**
- You know the solution path upfront
- The steps are predictable and repeatable
- You need consistent, optimized execution
- You want reliability and auditability

Examples: data extraction pipelines, structured report generation, form processing, any task where you'd draw a flowchart and it wouldn't have "it depends" branches.

### Agents: jazz

An agent is like jazz — improvisation, reacting to what's happening, finding a path through unknown territory.

```
User input
    ↓
┌─→ Observe (tool results, environment)
│   ↓
│   Think (what should I do next?)
│   ↓
│   Act (call a tool, write code, ask a question)
│   ↓
└── Done? → No: loop back
         → Yes: return result
```

The LLM decides which tools to call, in what order, and when it's finished. You don't write the steps — you give it tools and a goal, and it figures out the path.

**Use agents when:**
- You don't know the solution path upfront
- The problem requires exploration or adaptation
- Different inputs need fundamentally different approaches
- The task needs improvisation — reacting to unexpected results

Examples: coding agents (Claude Code), research tasks, debugging (where the next step depends on what you find), open-ended problem solving.

### It's a spectrum, not a binary

Most real systems live somewhere in between:

```
WORKFLOW ←───────────────────────────→ AGENT

Fixed steps        Some LLM          Fully autonomous
No LLM decisions   decisions at       LLM-driven loop
                   branch points
```

A "workflow with LLM routing" might use a model to decide which of 3 fixed paths to take, but each path is predetermined. A "constrained agent" might loop autonomously but only within a specific set of allowed tools. Pure workflows and pure agents are the extremes — most production systems are somewhere in the middle.

### Why the distinction matters

If you call a workflow an "agent," you'll under-invest in the deterministic engineering that makes workflows reliable — error handling, retries, fallbacks, monitoring. You'll expect the LLM to handle things your code should handle.

If you call an agent a "workflow," you'll over-constrain it, writing rigid step sequences when the whole point is to let the model adapt. You'll fight the architecture instead of leveraging it.

Name the pattern correctly and the trade-offs become clear:

| | Workflow | Agent |
|---|---|---|
| **Control** | Your code | The LLM |
| **Path** | Predetermined | Discovered at runtime |
| **Reliability** | High (predictable) | Variable (depends on model) |
| **Flexibility** | Low (new cases need new code) | High (adapts to novel inputs) |
| **Cost** | Predictable | Variable (more LLM calls) |
| **Debugging** | Follow the code | Read the trace |

### Three properties of a real agent

Per Anthropic's framework, a true agent:

1. **Plans and operates independently** — it decides its own next step, potentially coming back to the human for clarification
2. **Gets ground truth from the environment** — it calls tools, runs code, reads files, and uses those results to inform its next decision
3. **Has a stopping condition** — it decides when it's done, or hits a developer-set limit (max iterations, timeout)

If your system doesn't have these — if the steps are predetermined and the LLM just fills in blanks — it's a workflow. And that's fine. Workflows are simpler, cheaper, and more reliable for the problems they're designed for.

The goal isn't to build agents everywhere. It's to pick the right pattern for the problem.

---

*Compiled from [Andrej Karpathy's "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY), [AI Hero's token explainer](https://www.youtube.com/watch?v=nKSk_TiR8YA), [AI Hero's context window deep dive](https://www.youtube.com/watch?v=-uW5-TaVXu4), [AI Hero's agents vs workflows](https://www.youtube.com/watch?v=AtYtuVTZCQU), and [Anthropic's "Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents).*
