# Fine-Tuning in Production: How Companies Actually Do It

*Technique taxonomy, model selection, and 24 case studies with verified blog post references — across SFT, DPO, GRPO, distillation, and more.*

> **For the mechanical details** of how SFT works under the hood — loss masking, chat templates, training loops — see the companion posts: [The Engineer's Guide](sft-from-scratch-engineer.md) and [The Builder's Guide](sft-from-scratch-builder.md).

---

**Table of Contents**

1. [Why This Post Exists](#why-this-post-exists)
2. [The Complete Fine-Tuning Taxonomy](#the-complete-fine-tuning-taxonomy)
3. [Should You Fine-Tune At All?](#should-you-fine-tune-at-all)
4. [Picking Your Base Model](#picking-your-base-model)
5. [Company Case Studies by Pattern](#company-case-studies-by-pattern)
6. [The Patterns Matrix — When to Apply What](#the-patterns-matrix--when-to-apply-what)
7. [Best Practices from Reliable Sources](#best-practices-from-reliable-sources)
8. [The Cost Reality](#the-cost-reality)
9. [What Can Go Wrong](#what-can-go-wrong)
10. [Conclusion + Quick Reference](#conclusion--quick-reference)

---

## Why This Post Exists

Most fine-tuning guides explain techniques. This one shows what happens when 24 companies ship them.

There's no shortage of tutorials on *how* to fine-tune a model. What's missing is the part that comes after: what technique did they pick, why, what went wrong, and what did they actually measure? This post fills that gap with verified case studies — every claim traced back to a public blog post, engineering writeup, or paper.

A note on methodology: many "fine-tuning case studies" floating around are actually companies using RAG, prompting, or off-the-shelf APIs. Accolade uses RAG. Agmatix uses prompt engineering. Adyen uses RAG for fraud detection. Wayfair uses RAG for product search. These are good approaches — but they're not fine-tuning, and citing them as such muddies the decision for anyone trying to figure out what technique to apply. This post includes only companies that actually modified model weights.

The techniques covered here span the full post-training spectrum: SFT, continued pre-training, DPO, KTO, GRPO, reinforcement fine-tuning (RFT), RLHF, distillation, and custom architectures. If you want to understand how any of these work at the implementation level, the companion posts linked above walk through the code. This post is about the decisions and outcomes.

---

## The Complete Fine-Tuning Taxonomy

Fine-tuning isn't one thing. It's a two-dimensional space: *what you optimize* (the training objective) crossed with *how you optimize efficiently* (the parameter strategy). Every production deployment sits somewhere on this grid.

### Axis 1: What to Optimize (Training Objective)

| Method | What It Does | Data Needed | Company Examples |
|--------|-------------|-------------|-----------------|
| **SFT** | Imitate demonstrations | Prompt-completion pairs | Shopify, Instacart, Uber, Stitch Fix, Grammarly |
| **Continued Pre-Training** | Inject domain knowledge | Large domain corpus | Bloomberg (finance), Replit (code) |
| **DPO** | Preference alignment without RL | Chosen/rejected pairs | Netflix, Spotify, Writer |
| **KTO** | Preference from binary feedback | Thumbs up/down signals | Contextual AI |
| **ORPO** | Combined SFT + preference in one pass | Chosen/rejected pairs | Emerging — few production deployments |
| **GRPO** | RL with verifiable rewards, no critic | Verifiable reward signal | Netflix (recommendations) |
| **RFT** (OpenAI) | Managed RL with custom grader | Prompts + grader function | DraftWise (legal) |
| **RLHF** (PPO) | Full RL with reward model | Preference rankings + reward model | Spotify (partially) |
| **RLAIF / Constitutional AI** | RL from AI feedback | AI-generated preferences | Rarely used by app companies |
| **Distillation** | Student mimics teacher | Teacher model outputs | Instacart, Walmart, Checkr |

**SFT** is the workhorse — nearly every company starts here. You collect input-output pairs that demonstrate the behavior you want, and the model learns to imitate them. The companion posts cover this in detail, including loss masking and chat template mechanics.

**Continued pre-training** differs from SFT in that you're not teaching behavior — you're teaching *knowledge*. Bloomberg fed 363B tokens of financial data into a model before any instruction tuning. Replit fed code. The goal is to saturate the model with domain-specific patterns before teaching it to follow instructions.

**DPO, KTO, and ORPO** are all preference optimization methods that skip the reward model. DPO needs explicit chosen/rejected pairs. KTO works with simpler binary feedback (just thumbs up or thumbs down). ORPO combines the SFT and preference steps into a single training pass. In practice, DPO dominates production use — KTO and ORPO are newer and less battle-tested.

**GRPO** (Group Relative Policy Optimization, from the DeepSeek R1 paper) is RL without a critic model. Instead of training a separate value function, it estimates baselines from a group of sampled outputs. Netflix uses it for recommendations where you can compute a verifiable reward signal.

**RFT** is OpenAI's managed reinforcement fine-tuning. You provide prompts and a grading function, OpenAI handles the RL loop. DraftWise used this for legal search quality.

**Distillation** is a cost play: run a large expensive model (the teacher) on your data, then train a small cheap model (the student) to reproduce those outputs. Instacart, Walmart, and Checkr all use this pattern — generate labels with GPT-4, train a 7-8B model to match them, serve the small model at a fraction of the cost.

### Axis 2: How to Optimize Efficiently (Parameter Strategy)

| Method | Trainable Params | Memory Savings | When to Use |
|--------|-----------------|----------------|-------------|
| **Full Fine-Tuning** | 100% | None | Deep adaptation, large budget |
| **LoRA** | ~0.1-1% | ~10x | Default choice for most companies |
| **QLoRA** | ~0.1-1% (4-bit base) | ~20x | GPU-constrained |
| **DoRA** | ~0.1-1% (decomposed) | ~10x | Better accuracy than LoRA |
| **Adapters** | Small bottleneck layers | ~5-10x | Multi-task serving |
| **Prompt Tuning** | Tiny (input embeddings only) | Maximum | Lightest adaptation |

**LoRA** is the industry default. Instead of updating all model weights, you freeze the base model and add small trainable matrices (rank decompositions) to specific layers. This cuts GPU memory by roughly 10x and makes it practical to fine-tune large models on consumer hardware. Anyscale's benchmarks show LoRA matches full fine-tuning quality in 90%+ of cases.

**QLoRA** adds quantization on top — the base model is stored in 4-bit precision, and LoRA adapters are trained in 16-bit. This roughly halves memory again, making 70B models trainable on a single GPU.

**Full fine-tuning** still has its place when you need deep adaptation: Shopify fine-tunes fully because they need the model to deeply understand multimodal product catalogs at 40M inferences per day. Bloomberg did full pre-training because they needed the model to *think* in finance, not just answer finance questions.

### The Two-Axis Grid: Where Companies Actually Land

```
                    Full FT        LoRA/QLoRA       Prompt Tuning
              ┌──────────────┬──────────────────┬────────────────┐
  SFT         │ Shopify      │ Instacart, Uber  │                │
              │ Bloomberg    │ Walmart, Amdocs  │                │
              │ Stitch Fix   │ AT&T, Checkr     │                │
              ├──────────────┼──────────────────┼────────────────┤
  DPO/KTO     │              │ Netflix, Spotify │                │
              │              │ Writer           │                │
              ├──────────────┼──────────────────┼────────────────┤
  GRPO/RFT/RL │              │ Netflix (GRPO)   │                │
              │              │ DraftWise (RFT)  │                │
              ├──────────────┼──────────────────┼────────────────┤
  Distillation│              │ Instacart        │                │
              │              │ Walmart          │                │
              ├──────────────┼──────────────────┼────────────────┤
  Cont. Pre-  │ Bloomberg    │                  │                │
  Training    │ Replit       │                  │                │
              └──────────────┴──────────────────┴────────────────┘
```

Two things jump out. First, the LoRA/QLoRA column dominates. Full fine-tuning is reserved for companies with unusual requirements (multimodal at massive scale, or building from scratch). Prompt tuning is largely absent — in practice, teams either prompt-engineer (no training) or LoRA (real training), with little middle ground.

Second, the bottom half of the grid is sparse. DPO, GRPO, and RL are second-stage techniques: companies add them only after SFT hits a quality ceiling. The typical progression is SFT first, then preference optimization if needed. Netflix is the clearest example — they evolved from SFT to SFT→GRPO→DPO as they needed more nuanced quality.

---

## Should You Fine-Tune At All?

Before picking a technique, ask whether you should be fine-tuning at all. The industry data suggests most companies shouldn't.

**a16z's 2025 enterprise AI report** found that most enterprises are not seeing ROI on fine-tuning. The investment in data collection, training infrastructure, and ongoing maintenance exceeds the benefit for most use cases. Prompt engineering and RAG handle the majority of production applications.

**Menlo Ventures' 2025 survey** confirmed the pattern: prompt design remains the dominant approach, with fine-tuning occupying a niche for high-volume, specialized use cases.

This doesn't mean fine-tuning is wrong — it means it's for specific situations, not a default strategy.

### The Decision Tree

```
Start here: Can you solve this with prompting + RAG?
  │
  ├── Yes → Stop. Ship it. Iterate on prompts.
  │         (Honeycomb built their query assistant for $30/month in API costs)
  │
  └── No, I've hit a wall → What kind of wall?
        │
        ├── Format/behavior problem (model doesn't follow instructions,
        │   wrong output structure, inconsistent style)
        │     → SFT is your answer. Fine-tuning is for form, not facts.
        │       (Anyscale: "fine-tuning is for form, not facts")
        │
        ├── Knowledge problem (model doesn't know about my domain)
        │     → RAG first. Add a retrieval layer.
        │       Fine-tuning doesn't reliably inject new facts.
        │
        ├── Cost problem (API costs unsustainable at my volume)
        │     → Self-host + fine-tune a small model.
        │       (Shopify: 40M inferences/day made API costs prohibitive)
        │       (Checkr: 5x cost reduction vs GPT-4)
        │
        └── Quality problem (format is fine, but outputs aren't good enough)
              → Add DPO/RL stage after SFT.
                (Netflix: SFT wasn't enough, added GRPO then DPO)
                (Spotify: SFT + DPO + RLHF for 4x engagement)
```

### Companies That Thought They Needed Fine-Tuning But Didn't

It's worth calling out companies often cited as fine-tuning examples that actually use other approaches:

| Company | What they actually do | Why it's not fine-tuning |
|---------|----------------------|------------------------|
| Accolade | RAG over medical knowledge base | No model weights modified |
| Adyen | RAG for fraud documentation | Retrieval, not training |
| Agmatix | Prompt engineering for agriculture | Prompt design, no training |
| Wayfair | RAG for product catalog search | Embedding + retrieval pipeline |
| Honeycomb | Prompting for query generation | $30/month, pure prompt engineering |
| Accenture | Various client deployments | Mostly RAG + prompt engineering |

These companies made the right choice — they identified that their problem didn't require weight modification and used simpler, cheaper approaches. The lesson: **the best fine-tuning decision is often not to fine-tune.**

---

## Picking Your Base Model

Once you've decided to fine-tune, the next decision is which model to start from. This breaks down into two sub-decisions: base vs. instruct, and model size vs. hosting strategy.

### Base vs. Instruct

**Instruct models** (most common choice): These are models that have already been through post-training — they follow instructions, respond in conversational format, and have safety guardrails. Most companies start here because it means less SFT work.

- Instacart: LLaMA 3 8B-instruct
- Amdocs: LLaMA 3.1 8B-instruct
- AT&T: Mistral 7B (instruct variant via NeMo)

**Base models** (rare, for deep adaptation): These are raw pre-trained models with no instruction tuning. You'd choose these when you need fundamental behavioral changes that instruct tuning would interfere with, or when you're building from scratch.

- Bloomberg: Trained from scratch — no existing model had sufficient financial knowledge
- Replit: Custom 3B/7B from scratch — code completion has different distribution than chat
- Shopify: Multimodal fine-tuning on Qwen2VL 7B — needed custom vision-language behavior

The rule of thumb: **start with instruct unless you have a specific reason not to.** Base models require more SFT data, more training compute, and more careful evaluation to reach a usable state.

### The Volume × Complexity Matrix

The right model size and hosting strategy depends on two factors: how many inferences you need (volume) and how complex the task is (does it require reasoning, multimodal understanding, or domain expertise).

```
                         Low Volume              High Volume
                   ┌─────────────────────┬─────────────────────┐
                   │                     │                     │
   Simple Task     │  API Fine-Tuning    │  Small Open Model   │
                   │  (managed)          │  (self-hosted)      │
                   │                     │                     │
                   │  Harvey (legal)     │  Instacart (8B)     │
                   │  Stitch Fix (GPT-3) │  Checkr (8B)        │
                   │  DraftWise (o-ser.) │  Convirza (8B)      │
                   │                     │                     │
                   ├─────────────────────┼─────────────────────┤
                   │                     │                     │
   Complex Task    │  Large Open Model   │  Custom Model       │
                   │  (self-hosted)      │  (built or heavily  │
                   │                     │   adapted)          │
                   │  Uber (Mixtral)     │  Shopify (7B multi) │
                   │  Walmart (70B)      │  Bloomberg (50B)    │
                   │                     │  Airbnb (12B)       │
                   │                     │                     │
                   └─────────────────────┴─────────────────────┘
```

**Low volume + simple** → Use API fine-tuning (OpenAI, Anthropic). You don't need to manage infrastructure. Harvey fine-tunes GPT models through OpenAI's API for legal work. DraftWise uses OpenAI's reinforcement fine-tuning. The cost per inference is higher, but volume is low enough that it doesn't matter.

**High volume + simple** → Self-host a small open model (7B-8B). This is the sweet spot for cost-driven fine-tuning. Instacart, Checkr, and Convirza all landed here: train a Llama-3 8B with LoRA, serve it on your own GPUs, and cut costs 5-10x compared to API calls.

**Low volume + complex** → Self-host a larger open model. When tasks require multi-step reasoning, long context, or cross-domain knowledge, small models struggle. Uber uses Mixtral for complex engineering tasks. Walmart uses larger models for product catalog understanding.

**High volume + complex** → Custom model territory. Shopify built a multimodal pipeline handling 40M inferences per day. Bloomberg trained a 50B model from scratch. Airbnb trained custom 12B models for speech recognition and paraphrasing. These are expensive projects — you're here because nothing else works at your scale and complexity.

### Model Selection by Use Case

| Use Case | Recommended Starting Point | Why | Companies |
|----------|---------------------------|-----|-----------|
| Text classification | 7-8B instruct + LoRA | Small model handles classification well | Instacart, AT&T |
| Structured extraction | 7-8B instruct + LoRA | Schema following is a format task | Amdocs, Checkr |
| Code generation | Code-specialized model | Code-specific pre-training matters | Replit |
| Conversational AI | 7-8B instruct + LoRA (or API) | Instruct base handles conversation | Shell, AT&T |
| Content generation | API fine-tuning (or 7-8B) | Quality matters more than cost | Stitch Fix, Writer |
| Recommendations | Depends on architecture | Novel application of LLMs | Netflix, Spotify |
| Multimodal | Vision-language model | Need multimodal architecture | Shopify |
| Finance/legal | Domain pre-training or API | Domain knowledge critical | Bloomberg, Harvey |

---

## Company Case Studies by Pattern

Each case study is backed by a verified public blog post or paper. For each company: what they built, how it works in production, what the real numbers are, and what you can learn from it.

---

### Pattern A: "API Costs Forced Our Hand" — Cost-Driven Fine-Tuning

The most common entry point for production fine-tuning. A company builds a working product on top of a large API model, discovers the per-inference cost is unsustainable at scale, and trains a small self-hosted model to replicate the behavior.

```
  Large API Model (expensive)
       │ generates training data (distillation)
       ▼
  Small Open Model + LoRA (cheap to serve)
       │
       ▼
  Self-hosted inference at 1/5th - 1/10th cost
```

#### Shopify — Multimodal Product Catalog at 40M Inferences/Day

**Problem:** Shopify's Global Catalogue needs to classify, tag, and enrich every product across its merchant ecosystem. That means understanding product images alongside text — category classification, attribute extraction, title standardization, description summarization, and review aggregation. At 10 million product updates per day flowing in from merchant uploads, APIs, and integrations, this produces 40 million LLM inference calls and 16 billion tokens inferred daily.

**Model evolution:** Shopify iterated through three vision-language models in sequence, each time improving accuracy while reducing cost:
1. LLaVA 1.5 7B — initial deployment
2. LLaMA 3.2 11B — higher accuracy but more GPU-hungry
3. **Qwen2VL 7B** — current production model, higher accuracy than the 11B *and* lower GPU requirements

**Training approach:** Full SFT (not LoRA). Multi-task, multi-entity — a single model handles classification, extraction, standardization, summarization, and review aggregation across products, variants, sellers, and reviews simultaneously. Rather than separate models per task, one model does everything.

**Data strategy:** A hybrid annotation pipeline:
1. Multiple LLM agents independently suggest labels for a product
2. An LLM arbitrator selects the best suggestion for training data
3. Human annotators resolve ambiguities on test sets
This produces high-quality datasets faster than pure human annotation.

**Key optimization — selective field extraction:** Instead of asking the model to output all fields for every request, they prompt for only the specific fields needed. This single change:
- Reduced median latency from **2 seconds → 500 milliseconds**
- Cut **GPU usage by 40%** (fewer tokens generated)
- Improved generalization

**Serving architecture:**
```
  10M+ product updates/day (merchants, APIs, integrations)
       │
       ▼
  Kafka-based Dataflow streaming pipeline
       │
       ▼
  Triton Inference Server (orchestrates GPU fleet)
       │ ── FP8 quantization (smaller memory footprint, larger batches)
       │ ── KV cache (reuses attention patterns across requests)
       │ ── Dynamic batching
       ▼
  40M inferences/day → 16B tokens/day
       │
       ▼
  Four-layer catalogue: Ingest → Understand → Match → Reconcile
       │
       └── Deduplication via locality-sensitive hashing +
           embedding-based clustering + discriminator cascade
```

**In production?** Yes. Powers Shopify Sidekick, Shop app chat, search, recommendations, and the merchant admin (real-time category suggestions as merchants create listings).

**What's next:** Consolidating per-entity models into a single multi-entity model, exploring graph-based reasoning across entity relationships, and continuous retraining via active learning.

**Key learning:** The model architecture matters less than the serving optimization. Shopify's biggest wins came from selective field extraction (4x latency improvement) and FP8 quantization, not from model architecture changes. At 40M inferences/day, serving efficiency dominates training efficiency.

**Source:** [shopify.engineering/leveraging-multimodal-llms](https://shopify.engineering/leveraging-multimodal-llms)

---

#### Checkr — 5x Cost Reduction for Background Check Processing

**Problem:** Checkr provides background checks for companies like Uber, DoorDash, and Instacart. They process criminal records, court documents, and regulatory filings — documents with complex, variable formats that require structured extraction.

**Model + technique:** Llama-3-8B fine-tuned via Airtrain platform, LoRA.

**Data strategy:** Classic distillation — used GPT-4 to generate high-quality labels on their document corpus, then trained the 8B model to match. The key insight: for narrow extraction tasks (extracting specific fields from known document types), a small model trained on a large model's outputs captures enough of the capability.

**Results:** 5x cost reduction and 30x speed improvement compared to GPT-4, with comparable accuracy on their document processing tasks.

**In production?** Yes, processing background checks at scale.

**Key learning:** Distillation has the best cost-performance ratio for narrow tasks. If your use case is specific enough (extracting structured data from a known document format), the 8B model matches GPT-4 at 1/5th the cost and 30x the speed.

**Source:** Airtrain case study — Checkr background check processing

---

#### Convirza — 10x Cost Reduction for Call Analytics

**Problem:** Convirza analyzes phone calls for marketing attribution — transcribing calls and extracting structured insights (lead quality, intent, outcome). API costs grew linearly with call volume.

**Model + technique:** Llama-3-8B fine-tuned via Predibase LoRA (LoRA Land platform).

**Data strategy:** Production call data labeled by the existing OpenAI pipeline, then used to train the smaller model. The existing API pipeline becomes the teacher.

**Results:** 10x cost reduction compared to OpenAI API, with comparable accuracy on call classification and extraction.

**In production?** Yes.

**Key learning:** Multi-LoRA serving (one base model, multiple task-specific adapters) lets you handle many call analytics tasks from a single GPU deployment. Predibase serves adapters for under $8 each — making it economical to have dozens of task-specific adapters sharing one base model.

**Source:** [predibase.com/blog/convirza-case-study](https://predibase.com/blog/convirza-case-study)

---

### Pattern B: "We Needed a Domain Specialist" — Domain-Specific Fine-Tuning

These companies fine-tune because their domain is specialized enough that general-purpose models perform poorly even with good prompting. The fix is either (a) pre-training on domain data, (b) SFT with domain-specific instruction pairs, or (c) both.

```
  General Pre-trained Model
       │ (optional: continued pre-training on domain corpus)
       ▼
  Domain SFT (domain-specific instruction pairs)
       │ (optional: DPO/preference tuning)
       ▼
  Domain-specific evaluation (NOT generic benchmarks)
```

#### Bloomberg — 50.6B Finance Model, 1.3M GPU Hours

**Problem:** Financial NLP requires understanding SEC filings, earnings calls, financial news, and market sentiment. General models underperform because financial text has its own vocabulary, document structures, and reasoning patterns (aspect-specific sentiment, numerical reasoning over financial tables, entity disambiguation across tickers and company names).

**Model:** BloombergGPT — 50.6B parameters. 70 transformer decoder layers, 40 attention heads, hidden dimension 7,680. Decoder-only causal LM based on BLOOM architecture with ALiBi positional encoding and a Unigram tokenizer (131,072 token vocabulary).

**Training data — the 51/49 split:**

| Source | Tokens | Share |
|--------|--------|-------|
| **Financial (FinPile)** | **363B** | **51.27%** |
| — Web (financial) | 298B | 42.01% |
| — News | 38B | 5.31% |
| — SEC Filings | 14B | 2.04% |
| — Press Releases | 9B | 1.21% |
| — Bloomberg Content | 5B | 0.70% |
| **General (public)** | **345B** | **48.73%** |
| — The Pile | 184B | 25.9% |
| — C4 | 138B | 19.48% |
| — Wikipedia | 24B | 3.35% |

Total: 569B tokens consumed during training across 139,200 steps.

**Infrastructure:** 512 NVIDIA A100 40GB GPUs (64 AWS p4d.24xlarge instances on SageMaker). ZeRO Stage 3, activation checkpointing, BF16 mixed precision. 102 TFLOPs average throughput, 32.5 seconds per step. Training duration: ~53 days. Total compute: 1.3 million GPU hours.

**Results — financial benchmarks (5-shot):**

| Task | BloombergGPT 50B | GPT-NeoX 20B | OPT 66B | BLOOM 176B |
|------|-------------------|--------------|---------|------------|
| ConvFinQA (numerical reasoning) | **43.41** | 30.06 | 27.88 | 36.31 |
| FiQA SA (sentiment) | **75.07** | 50.59 | 51.60 | 53.12 |
| FPB (sentiment) | **51.07** | 44.64 | 48.67 | 50.25 |
| Headline classification | **82.20** | 73.22 | 79.41 | 76.51 |
| Average (5 tasks) | **62.51** | 51.90 | 53.01 | 54.35 |

On internal Bloomberg sentiment analysis (equity news, transcripts, ESG), the margins are even larger — averaging **62.47 vs. 35.76** for the next-best model (OPT-66B). BloombergGPT nearly doubles competitors on internal financial tasks despite being smaller than OPT-66B and BLOOM-176B.

General benchmarks (MMLU, BIG-bench Hard, SuperGLUE) remain competitive — the financial specialization doesn't sacrifice general capability.

**In production?** Used internally at Bloomberg.

**Key learning:** The 51/49 financial/general data split is the design decision that matters most. Too much financial data would collapse general capabilities. Too little wouldn't specialize. Bloomberg proved that mixed-domain pre-training can beat models 3x its size on domain tasks while staying competitive on general benchmarks. But the price tag — $3M+ in compute, 53 days of 512 GPUs — means this only works for companies with permanent, large-scale NLP needs.

**Source:** [arxiv.org/abs/2303.17564](https://arxiv.org/abs/2303.17564) (BloombergGPT paper)

---

#### Harvey AI — Custom Legal Model with 0.2% Hallucination Rate

**Problem:** Legal work requires extreme precision. One hallucinated case citation can destroy trust and create malpractice liability. General-purpose models hallucinate legal citations at rates of 17-33% — unusable for professional legal work.

**What they built:** Not a simple fine-tune — a **custom-trained case law model** built in partnership with OpenAI using "novel mid-training and post-training techniques" that inject new knowledge and reasoning capabilities. Standard fine-tuning and RAG alone were deemed insufficient.

**Models in production:** GPT-4, GPT-4 Turbo, o1-preview, o1-mini, GPT-5 (referenced in recent posts). Custom embeddings trained with Voyage AI on 20B+ tokens of case law.

**Training data:** Equivalent of **10 billion tokens** of legal data, starting with Delaware case law and expanding to all U.S. case law. Harvey is also exploring NVIDIA accelerated computing on Azure to train open-source models.

**How the 97% preference was measured:** 10 of the largest law firms participated. Attorneys received **blind side-by-side comparisons** — output from the custom case law model vs. output from GPT-4 for the same legal question. Result: **97% of the time, lawyers preferred the custom model**. Reasons: longer and more complete answers, greater nuance, more relevant case law coverage, and critically — **every sentence supported by the case it cites** (the model doesn't fabricate cases).

**Results:**

| Metric | Value |
|--------|-------|
| Factual response improvement over base | **83% increase** |
| Hallucination rate | **~0.2%** (vs. 17-33% for other legal AI) |
| Lawyer preference over GPT-4 | **97%** |
| Irrelevant search result reduction (custom embeddings) | **25%** |

**Production scale:**

| Metric | Value |
|--------|-------|
| Top 100 US law firm adoption | **42%** |
| Enterprise clients | **500+** across **54 countries** |
| Annual revenue | **$100M** (achieved in 3 years) |
| Customer retention (13 months) | **70%+** |
| API request growth (Q1 2025) | **300%** |

**Architecture:**
```
  Legal query
       │
       ▼
  Custom embeddings (Voyage AI, 20B+ tokens of case law)
       │ ── reduces irrelevant results by 25%
       ▼
  RAG retrieval layer (case law, statutes, filings)
       │
       ▼
  Custom-trained case law model (mid-training + post-training)
       │ ── 0.2% hallucination rate
       │ ── every claim tied to citation
       ▼
  LLM orchestration + reasoning agents
       │ ── multi-step legal workflows
       │ ── per-customer customization
       ▼
  Research │ Drafting │ Review │ Agentic workflows
```

**Infrastructure:** Microsoft Azure OpenAI Service, Azure Kubernetes Service, Azure Database for PostgreSQL, Azure Blob Storage with BYOK encryption, multi-region deployment for data residency compliance.

**In production?** Yes, at massive scale. $300M Series E in June 2025 at ~$5B valuation.

**Key learning:** Harvey's success isn't about a novel training technique — it's about **the data and the feedback loop**. 10B tokens of curated legal text, blind lawyer evaluations to validate quality, and the 0.2% hallucination rate as a north star metric. In regulated domains, the hallucination rate is the metric that matters most — not MMLU, not perplexity, not general benchmarks.

**Source:** [openai.com/index/harvey](https://openai.com/index/harvey) and [harvey.ai/blog](https://harvey.ai/blog)

---

#### Grammarly — 3B Model Preferred 6.4:1 Over GPT-3 175B

**Problem:** Text editing requires fine-grained control over edit types — grammar correction, simplification, formality transfer, paraphrasing, coherence improvement, neutralization, and compression. General models can edit text, but they can't reliably produce the *specific type* of edit requested.

**Model:** CoEdIT, fine-tuned from FLAN-T5 (Google's instruction-tuned T5). Three sizes:

| Model | Base | Parameters |
|-------|------|-----------|
| CoEdIT-Large | flan-t5-large | **770M** |
| CoEdIT-XL | flan-t5-xl | **3B** |
| CoEdIT-XXL | flan-t5-xxl | **11B** |

Architecture: T5ForConditionalGeneration (encoder-decoder, seq2seq). The critical design choice: an encoder-decoder architecture, not a decoder-only model, because editing is fundamentally a seq2seq task (input text → edited text).

**Training data — the 82K examples:**
82,000 instruction-source-target triplets across 7 task types:

| Task | Source Datasets |
|------|----------------|
| Grammatical Error Correction | cLang-8, JFLEG |
| Text Simplification | TurkCorpus, ASSET, WikiAuto |
| Coherence | IteraTeR, DiscoFuse |
| Formality Transfer | GYAFC (Grammarly's Yahoo Answers Formality Corpus) |
| Neutralization | WNC (Wiki Neutrality Corpus) |
| Paraphrasing | MRPC (Microsoft Research Paraphrase Corpus) |
| Composite (multi-step) | Combinations of above |

Each example includes a natural language instruction specifying the edit type ("Make this more formal:", "Fix grammar:", "Simplify this sentence:"), so the model learns to map instruction type → editing behavior.

69,100 training + 1,710 validation examples are publicly available on HuggingFace (some withheld due to licensing). Code is open-source at github.com/vipulraheja/coedit.

**The GPT-3 comparison — human evaluation:**

In blind evaluation, human evaluators compared CoEdIT-XL (3B parameters) against GPT-3-Edit (175B parameters):

| | CoEdIT-XL (3B) | GPT-3-Edit (175B) | Tie |
|---|---|---|---|
| **Single task** | **64%** preferred | 10% preferred | 26% |
| **Composite tasks** | 38% preferred | 34% preferred | 28% |

On single editing tasks, evaluators preferred CoEdIT **6.4 to 1** over GPT-3 — despite being **~58x smaller**. On composite multi-step edits (a harder task), the gap narrows but CoEdIT-Composite still edges ahead.

**In production?** The CoEdIT research feeds into Grammarly's production editing pipeline. Models are publicly available on HuggingFace. Published at EMNLP 2023.

**Key learning:** This is the clearest evidence in this entire post for when fine-tuning provides outsized value. A 3B parameter model, fine-tuned on 82K task-specific examples, is preferred 6.4:1 over a model 58x its size — but only on the *specific task* it was trained for. Run CoEdIT on MMLU and it would lose badly. The lesson: **evaluate on your task, not on benchmarks.** Generic benchmarks completely fail to predict domain-specific performance.

**Source:** [grammarly.com/blog/engineering/coedit-text-editing](https://grammarly.com/blog/engineering/coedit-text-editing) and [arxiv.org/abs/2305.09857](https://arxiv.org/abs/2305.09857)

---

#### Replit — Code Repair Model That Matches GPT-4 on Real Errors

**Problem:** Code completion and code repair are different problems. Completion continues partial code. Repair *fixes* broken code — it intercepts compiler errors, linting warnings, and type errors, then generates diffs to resolve them. Replit's IDE generates "hundreds of millions" of LSP diagnostic events per day, but only ~10% of Python diagnostics have associated fixes. They needed a model to fill that gap.

**Code generation model (replit-code-v1-3b):**
- 2.7B parameters with AliBi positional embeddings and Flash Attention
- Trained on Stack Dedup v1.2: 175B tokens of permissively licensed source code in 350+ languages
- Trained for 3 epochs = 525B tokens total (~195 tokens per parameter)
- Infrastructure: 256 x A100-40GB GPUs on MosaicML platform
- HumanEval pass@1: 21.9%
- **In production:** Powers GhostWriter, Replit's code autocomplete

**Code repair model (the more interesting one):**
- **7B parameters**, fine-tuned from DeepSeek-Coder-Instruct-v1.5
- Trained on a single node with 8 H100 GPUs using MosaicML's LLM Foundry

**How it works in production:**
```
  IDE: user writes code
       │
       ▼
  LSP diagnostic event (compiler error, lint warning, type error)
       │ ── hundreds of millions of events/day across platform
       ▼
  Code Repair model (7B, DeepSeek-Coder fine-tune)
       │
       ├── Input: source code + diagnostic message + error line number
       │          (using angle-bracketed sentinel tokens)
       │
       └── Output: Numbered Line Diffs (NOT unified diffs —
                   they found line numbers were hallucinated
                   in unified diff format)
       │
       ▼
  Diff applied automatically to user's code
```

**Synthetic data pipeline for training:**
- Target: 100K examples of (code, diagnostic, fix) triples
- Scaling tested at 10K, 25K, 50K, 75K samples — found 100K optimal
- Generation: DSPy-based few-shot prompting against large code LLMs to generate diffs from real error states
- Filtering: LLM-based filtering increases correct-to-incorrect ratio, regex validates diffs, generated diffs tested for correct application
- Key discovery: Numbered Line Diffs work; Unified Diffs don't — models hallucinate line numbers in unified format

**Training details:** Decoupled AdamW optimizer, lr=1e-5 decaying to 0.01x, 100 warmup batches, BF16 mixed precision, 4 epochs, batch size 16, sequence packing ratio 6.0.

**Results:** The fine-tuned 7B model "matched or exceeded the performance of GPT-4 and Claude-3" on real-world error fixes — evaluated on both a 360-sample LeetCode benchmark (DebugBench, post-cutoff problems to avoid contamination) and a custom 389-sample benchmark testing the model in its actual inference setting.

**In production?** Yes — described as "the first low-latency code repair" model in production. Deployed in Replit's IDE, triggered automatically on diagnostic events.

**Key learning:** Two insights stand out. First, **diff format matters**: switching from unified diffs to numbered line diffs eliminated hallucination. Second, **synthetic data can match human-labeled quality** when you have a strong filtering pipeline — DSPy-generated examples, LLM-filtered, regex-validated, and diff-application-tested. The pipeline quality control matters as much as the model.

**Source:** [blog.replit.com/llm-training](https://blog.replit.com/llm-training) and [blog.replit.com/code-repair](https://blog.replit.com/code-repair)

---

#### Writer — Palmyra Family: 73% on CFA Level III (GPT-4 Scored 33%)

**Problem:** Enterprise content generation in healthcare and financial services requires both high quality and strict compliance. But Writer didn't just fine-tune an existing model — they built an entire model family spanning 128M to 70B+ parameters, with domain-specific variants for medicine and finance.

**Model family:**

| Model | Parameters | Context | Purpose |
|-------|-----------|---------|---------|
| Palmyra-Small | 128M | — | Lightweight tasks |
| Palmyra-Base | 5B | — | General enterprise |
| Palmyra-Large (X) | 20B | — | Enterprise API |
| Palmyra-Med-70B | 70B | 8K-32K | Healthcare |
| Palmyra-Fin-70B | 70B | 32K | Finance |
| Palmyra X5 | MoE (undisclosed) | **1M tokens** | Enterprise agents |
| Palmyra-mini | 1.5B (open-source) | — | On-device/private |

**DPO in production:** Writer uses DPO as a core post-training technique across the family. Palmyra-Med-70B is fine-tuned with a custom Medical Instruct dataset + DPO dataset. Palmyra-Fin-70B integrates a financial dataset with DPO fine-tuning. The DPO stage aligns outputs with domain-specific quality standards that SFT alone can't capture.

**Results — healthcare (Palmyra-Med-70B, zero-shot):**

| Benchmark | Palmyra-Med | Med-PaLM 2 (5-shot) | GPT-4 |
|-----------|-------------|---------------------|-------|
| MMLU Clinical Knowledge | **90.9%** | — | — |
| MMLU Medical Genetics | **94.0%** | — | — |
| Average (9 biomedical benchmarks) | **85.9%** | 84.1% | 82.8% |

Palmyra-Med beats Med-PaLM 2 by ~2 points — and Med-PaLM 2 requires 5-shot prompting while Palmyra achieves this zero-shot.

**Results — finance (Palmyra-Fin-70B):**

| Benchmark | Palmyra-Fin | GPT-4 |
|-----------|-------------|-------|
| CFA Level III exam | **73%** | **33%** |

First model reported to pass CFA Level III (historical passing threshold ~60%). GPT-4 scored 33% on the same exam.

**Results — general (Palmyra X5):**
- 1M-token context: processes full prompt in ~22 seconds
- Multi-turn function calls: ~300ms latency
- Cost: 3-4x less per token than GPT-4.1
- NVIDIA TensorRT-LLM optimizations: 23-30% reduction in time-to-first-token, ~60% increase in tokens/second

**In production?** Yes — Palmyra-Large powers Writer's Enterprise plan API. Med and Fin variants serve healthcare and financial customers. X5 is on AWS Bedrock and Writer AI Studio. Palmyra-mini (1.5B, open-source) launched September 2025 for on-device use.

**Key learning:** Domain-specific DPO produces measurable gains on domain benchmarks (CFA exam, medical licensing questions) that general-purpose DPO doesn't. Writer's approach — build the base model, then apply domain-specific SFT + DPO — gives them a family of models where each domain variant can be independently improved without affecting others.

**Source:** [writer.com/blog/palmyra](https://writer.com/blog/palmyra) and [developer.nvidia.com/blog](https://developer.nvidia.com/blog) (Writer Med & Fin release)

---

### Pattern C: "LoRA for Iteration Speed" — Parameter-Efficient Multi-Task

These companies use LoRA not just for memory savings, but as an operational strategy: fast iteration, multiple task-specific adapters, and lightweight deployment. The base model is shared; adapters are cheap to train and serve.

```
  Base Model (frozen, shared across tasks)
       │
       ├── LoRA Adapter A (classification)
       ├── LoRA Adapter B (extraction)
       ├── LoRA Adapter C (generation)
       └── LoRA Adapter D (summarization)
       │
       ▼
  Multi-LoRA serving: one GPU, multiple adapters
  (swap adapter per request, base model stays loaded)
```

#### Instacart — Head-Tail Architecture with 96.4% Precision

**Problem:** Instacart's Query Understanding (QU) system needs to interpret what millions of customers mean when they search. Legacy system: separate FastText classifiers for query classification, session-mined heuristics for query rewrites (~50% coverage), and no semantic role labeling. Each model had separate data pipelines, training, and serving. Poor generalization on tail queries — "vegan roast" returned nothing because there was no conversion history for the term.

**Model + technique:** Llama-3-8B-instruct + LoRA. But the interesting part is the **head-tail hybrid architecture**:

```
  User search query
       │
       ├── Head queries (frequent, cached)
       │     │
       │     ▼
       │   Offline Teacher Pipeline (large frontier LLM + RAG)
       │     │ ── RAG context: historical conversion data,
       │     │    product catalog embeddings with semantic
       │     │    similarity scores, session engagement data
       │     │
       │     ├── Cached results served directly (98% of queries)
       │     └── High-quality "curriculum" training dataset
       │              │
       │              ▼
       │         Fine-tune Student (Llama-3-8B + LoRA)
       │
       └── Tail queries (rare, real-time)
             │
             ▼
           Student model inference (2% of queries)
             │ ── latency: 300ms on H100 GPUs
             │    (down from 700ms on A100)
             ▼
           Post-processing guardrails:
             ├── Semantic similarity filtering
             ├── Catalog validation
             └── Hallucination filtering
```

**Three unified tasks (replacing separate ML systems):**
1. **Query Category Classification** — maps queries to hierarchical product taxonomy (Meat → Beef Ribs → Short Ribs)
2. **Query Rewrites** — substitutes, broader queries, synonyms
3. **Semantic Role Labeling (SRL)** — extracts product, brand, and attribute concepts for retrieval, ranking, and ad targeting

**Data strategy:** The teacher generates the curriculum — it's a large frontier LLM augmented with RAG context (historical conversion signals, product catalog embeddings, session engagement data). The teacher labels head queries offline with no latency constraints; the student learns from this curriculum to handle tail queries in real-time.

**Key optimization — adapter merging:** At inference time, LoRA adapter weights are merged directly into base model weights, eliminating adapter overhead. This produced a **30% latency cut** (700ms → 490ms). Combined with H100 GPU upgrade, final latency: **300ms**. They tested FP8 quantization (additional 10% latency cut) but chose to skip it — it caused a slight recall drop, and they prioritized quality.

**Results:**

| Metric | Value |
|--------|-------|
| SRL precision (fine-tuned student) | **96.4%** (vs. 95.4% for the frontier teacher) |
| SRL F1-score | 95.7% (vs. 95.8% teacher — essentially identical) |
| Query rewrite coverage | **50% → 95%+** of search traffic |
| Query rewrite precision | **90%+** across all three types |
| User scroll depth reduction | **6%** (users find items faster) |
| User complaints (tail queries) | **50% reduction** |
| Queries needing real-time inference | **Only 2%** (rest served from cache) |

The 96.4% precision is specifically on SRL — and notably, the fine-tuned 8B student achieves **higher precision than the much larger frontier teacher model** (96.4% vs. 95.4%), with essentially identical F1. The student outperformed the teacher on precision because fine-tuning on curated curriculum data produces more consistent outputs than few-shot prompting.

**Instacart's finding on method hierarchy:**
> Fine-tuning > Context-Engineering (RAG) > Prompting

**In production?** Yes, serving millions of cold-start queries weekly. Only 2% need real-time model inference; 98% are served from the offline teacher cache.

**What's next:** Expanding from single-query search to context-aware, multi-intent understanding — distinguishing "lasagna ingredients" (item search) from "quick lasagna recipe" (content discovery) from "lasagna delivery near me" (restaurant search).

**Key learning:** The head-tail architecture is the insight: you don't need real-time inference for every query. Cache the teacher's results for head queries (98% of traffic), fine-tune a fast student for the remaining 2%. The student ends up more precise than the teacher because curriculum learning on curated data beats few-shot prompting. And the 50% reduction in user complaints shows the impact — those tail queries were the ones users actually noticed.

**Source:** [instacart.com/company/tech-innovation/building-the-intent-engine](https://www.instacart.com/company/tech-innovation/building-the-intent-engine)

---

#### Uber — Michelangelo Platform for Self-Serve Fine-Tuning

**Problem:** Uber has multiple LLM use cases across the company — Uber Eats recommendations/search/item tagging, customer support chatbots, internal code tools, SQL query generation, user preference modeling — each with different requirements. Building separate models per team was unsustainable. The strategic motivation: "fine-tuned models can achieve similar performance to GPT-4 models while allowing for much more traffic at Uber's scale."

**Models fine-tuned:**
- Llama 2 (7B, 13B, 70B variants benchmarked)
- Llama 3 (450B mentioned as future scale target)
- Mixtral 8x7B (MoE, used for inference benchmarking)
- Both full fine-tuning and LoRA/QLoRA depending on use case

**Their finding on LoRA vs. full FT:** LoRA/QLoRA show "loss decreases much less than full parameter training" — they trade quality for efficiency and choose based on the use case. Full fine-tuning for quality-critical tasks, LoRA for iteration speed.

**Infrastructure — the Michelangelo platform:**

```
  ┌─────────────────────────────────────────────┐
  │  Layer 2: Federation                        │
  │  Multi-cluster resource management          │
  ├─────────────────────────────────────────────┤
  │  Layer 1: Orchestration                     │
  │  Kubernetes + KubeRay + Ray Train           │
  │    ├── TorchTrainer spawns Ray Actors       │
  │    ├── HuggingFace Transformers + Trainer   │
  │    └── DeepSpeed ZeRO (stages 1, 2, 3)     │
  ├─────────────────────────────────────────────┤
  │  Layer 0: Hardware                          │
  │  On-prem A100 (4 GPU/host, 600GB, 3TB SSD) │
  │  Cloud H100 (8 GPU/host, 1.8TB, 6TB SSD)   │
  │  Test scale: 32 GPUs across 8 hosts         │
  └─────────────────────────────────────────────┘

  Training pipeline:
  Data (HDFS/Terrablob/HuggingFace)
    → Tokenization
    → Distributed training (Ray + DeepSpeed)
    → Checkpoints (Terrablob)
    → Metrics (Comet server)
    → Serving (vLLM + Ray ActorPool)
```

**Software stack:** PyTorch, Ray Train, HuggingFace Transformers, DeepSpeed (ZeRO stages 1-3), NCCL, Flash Attention, Comet for metrics tracking.

**GPU optimization results:**

| Optimization | Impact |
|-------------|--------|
| DeepSpeed ZeRO-3 CPU offload | **34%+ GPU memory reduction** |
| Flash Attention | **50% GPU memory savings** |
| CPU offload | **3-4x batch size increase** |
| Flash Attention alone | **2x batch size** |
| Combined throughput | **2-3x increase** |
| H100 vs. A100 throughput | **3x higher** at equivalent batch sizes |

**Serving:** vLLM for inference. Ray ActorPool for distributed batch prediction.

**In production?** Yes for Uber Eats, customer support, code tools, RAG. Experimental: LLM Scorer (offline batch prediction), QLoRA (explored but inferior loss), Llama 3 450B (future target).

**Key learning:** The platform matters more than any individual model. Uber's investment in Michelangelo — making fine-tuning self-serve for non-ML teams — multiplied their impact across the organization. The key infrastructure decisions: hybrid A100 on-prem + H100 cloud under one platform, DeepSpeed ZeRO-3 + Flash Attention as the default memory optimization stack, and vLLM for serving everything. Network bandwidth is not their bottleneck; GPU memory is.

**Source:** [uber.com/blog/open-source-and-in-house-how-uber-optimizes-llm-training](https://www.uber.com/blog/open-source-and-in-house-how-uber-optimizes-llm-training)

---

#### Walmart — "Wallaby" Models for Product Catalogs

**Problem:** Walmart manages one of the world's largest product catalogs. Tasks include product classification, attribute extraction, description generation, and compliance checking across hundreds of categories. At Walmart's volume, API costs are prohibitive.

**Model + technique:** "Wallaby" model family, using LoRA + distillation. Larger models generate high-quality labels, smaller models are trained to reproduce them.

**Data strategy:** Walmart's internal product catalog data — millions of products with structured metadata, images, and descriptions. The distillation pipeline uses larger models to label edge cases and ambiguous product categories.

**Results:** Custom models that outperform general-purpose models on Walmart-specific product understanding tasks, served at a fraction of API costs.

**In production?** Yes.

**Key learning:** Retail product catalogs are a sweet spot for fine-tuning: the taxonomy is large but structured, the data is proprietary, and volume makes API costs prohibitive. LoRA adapters for different product categories keep the system modular — you can retrain one category's adapter without touching others.

**Source:** [tech.walmart.com](https://tech.walmart.com) (product catalogs blog)

---

#### Amdocs — 9-Point Accuracy Jump with LoRA on LLaMA 3.1

**Problem:** Amdocs, a telecom software company, needed to improve accuracy on telecom-specific NLP tasks — customer query classification, intent extraction, and knowledge retrieval from technical documentation.

**Model + technique:** LLaMA 3.1 8B-instruct + LoRA, fine-tuned with NVIDIA NeMo framework.

**Data strategy:** Telecom-specific instruction pairs derived from customer interactions and technical documentation.

**Results:** Accuracy improved from 0.74 to 0.83 on their telecom NLP benchmark — a 9-point jump from the simplest possible intervention (LoRA on a small instruct model).

**In production?** Yes.

**Key learning:** Sometimes the simplest approach delivers the best ROI. A 9-point accuracy improvement from LoRA on an 8B instruct model, with minimal infrastructure investment via NeMo, can be more valuable than a complex multi-stage pipeline that squeezes out 2 more points at 10x the engineering cost.

**Source:** [developer.nvidia.com/blog](https://developer.nvidia.com/blog) (Amdocs NeMo case study)

---

### Pattern D: "Format Wasn't Enough, We Needed Quality" — Multi-Stage Post-Training

These companies discovered that SFT alone hits a quality ceiling. The model follows the right format, but its outputs aren't *good enough* — it picks suboptimal recommendations, misses nuance, or lacks the judgment that human experts bring. The fix: add preference optimization (DPO, GRPO, or full RLHF) after SFT.

```
  Base Model
       │
  Stage 1: SFT (teach format and basic behavior)
       │
  Stage 2: DPO / GRPO / RFT (refine quality with preferences)
       │
  Stage 3: Safety alignment (optional, for user-facing applications)
       │
  Production deployment
```

#### Netflix — SFT→GRPO→DPO for Artwork Personalization (250M+ Users)

**Problem:** Netflix uses LLMs not just for "what should I watch?" but for **artwork personalization** — selecting which visual image of a title appeals most to each individual user, choosing from 4 to 40+ image options per title. They also apply LLMs to recommendation, search, and content understanding. Small improvements in recommendation relevance translate directly to subscriber retention across 250M+ subscribers.

**Models supported:** Qwen3, Gemma3, Qwen3 MoE, GPT-OSS (MoE), Llama 3.1 8B-Instruct. QwQ-32B used for reasoning augmentation (generating chain-of-thought explanations). No full-model materialization on a single device — all models require sharding (FSDP or Tensor Parallelism).

**The post-training pipeline (five supported techniques):**
1. **SFT** — negative log-likelihood loss with loss masking on assistant tokens only
2. **DPO** — chosen response = optimal artwork; rejected = randomly selected alternative
3. **RL (on-policy)** — inspired by DeepSeek-R1 and GRPO; integrated from open-source Verl library
4. **Knowledge Distillation**
5. **SFT with Reasoning** — QwQ-32B generates chain-of-thought explanations conditioned on known correct answers; filters responses not matching ground truth (<2% rejection rate)

**Key evolution:** The framework started SFT-only. With DeepSeek-R1 and GRPO in 2025, SFT became "table stakes rather than the finish line," driving the addition of RL support.

**Artwork personalization data and results:**

Training: 110,000 user-title-artwork tuples. Validation: 1,000. Test: 5,000 held-out user-title pairs. Artwork caption descriptions ~200 tokens each. Custom tokens: `<option>` and `</option>` delimiters for artwork choices.

| Method | Accuracy vs. Production | IPS Score vs. Production |
|--------|------------------------|-------------------------|
| Random guess | -74.96% | -4.59% |
| Zero-shot (8B) | -4.22% | -0.19% |
| SFT (8B) | -2.55% | +2.45% |
| DPO (8B) | +0.91% | +2.82% |
| **SFT + Reasoning (8B)** | **+1.41%** | **+5.21%** |

IPS (Inverse Propensity Score) weights correct predictions from 40-option sets 20x higher than 2-option sets. The best method achieves ~5% IPS improvement over Netflix's existing production system.

**DPO underperformance insight:** DPO underperformed SFT+Reasoning. Suspected cause: unclear distinction between negative examples — a user skipping artwork doesn't mean they'd dislike it, just that they preferred another. The rejected examples in DPO aren't truly "bad," making the preference signal noisy.

**Non-NLP use case — semantic IDs:** Netflix also trains internal models on member interaction event sequences (not natural language), using semantic IDs instead of video IDs. Result: **30% recall increase** in recommendation areas.

**Infrastructure and engineering:**

```
  ┌─────────────────────────────────────────────────┐
  │  Mako (Netflix internal ML compute platform)    │
  │  Provisions GPUs on AWS                         │
  ├─────────────────────────────────────────────────┤
  │  GPU Fleet: A100 + H200                         │
  │  Scales from single node to hundreds of GPUs    │
  ├─────────────────────────────────────────────────┤
  │  Software Stack:                                │
  │  PyTorch + FSDP │ Ray (orchestration)           │
  │  vLLM (inference) │ Verl (RL from Volcano Eng.) │
  │  HuggingFace (checkpoints, tokenizer)           │
  ├─────────────────────────────────────────────────┤
  │  Key Optimizations:                             │
  │  ├── Async sequence packing: 4.7x throughput    │
  │  ├── Vocab padding to multiples of 64           │
  │  │   (avoids 3x slower CUTLASS kernel fallback) │
  │  ├── Chunked cross-entropy for >128k vocabs     │
  │  └── Logit verifier gates new architectures     │
  └─────────────────────────────────────────────────┘
```

**Key engineering decisions:**
- On-the-fly async sequence packing: **4.7x improvement** in effective token throughput on skewed datasets
- Without vocabulary padding to multiples of 64, falling back from cuBLAS to CUTLASS kernel **triples** the language model head's execution time
- Chunked cross-entropy computation manages memory with large vocabularies (>128K tokens)
- **Logit verifier** gates new architecture support — given random inputs, the internal model must match the reference HuggingFace implementation exactly. AI coding agents automate model conversion, with the logit verifier as the acceptance criterion
- HuggingFace AutoTokenizer is the single source of truth (avoids training-serving skew)
- Hybrid single-controller + SPMD execution model for RL (SFT uses SPMD-only)

**In production?** The post-training framework is operational internally, serving multiple teams across recommendation, personalization, and search. The artwork personalization paper reports offline evaluation results — no A/B test or production deployment details published yet for the LLM-based approach (they have an existing production system that serves as the baseline).

**Key learning:** Netflix's biggest insight is that **SFT is table stakes, not the finish line.** Their evolution — SFT first, then adding RL when DeepSeek-R1 showed what was possible — mirrors what many companies will go through. The DPO underperformance finding is equally important: preference optimization only works when your negative examples are genuinely negative, not just "less preferred." In recommendation systems, that distinction is blurry.

**Source:** [netflixtechblog.com/scaling-llm-post-training-at-netflix](https://netflixtechblog.com/scaling-llm-post-training-at-netflix) and [arxiv.org/abs/2601.02764](https://arxiv.org/abs/2601.02764)

---

#### Spotify — DPO Preference Flywheel for AI Playlists

**Problem:** Spotify uses LLMs across multiple products — AI DJ (personalized commentary alongside songs), AI Playlist (natural language → playlist), contextualized recommendation explanations, podcast chapters, and search. The challenge in each: the model can output a list of songs (format), but the list needs to be songs *this specific user* will enjoy (quality).

**Models:** Llama 3.1 8B as primary production model (benchmarked 1B-8B, 8B won). Llama 3.1 405B used offline for synthetic data generation (too large for serving). 1B model for semantic ID work. LongT5 for podcast chapters. All served via vLLM with beam search decoding.

**The full adaptation pipeline (5 stages, from Dec 2024 blog):**
1. Extended pre-training on curated Spotify-specific datasets
2. Supervised instruction fine-tuning on 10 Spotify-specific tasks
3. RLHF (reinforcement learning from human feedback)
4. DPO (direct preference optimization)
5. Evaluation using MMLU as guardrail for general capability preservation

**Training data sources:** Internal Spotify examples, golden examples created by music domain experts, synthetic data from Llama 3.1 405B, zero-shot outputs from SOTA models, plus a small percentage of text-only instruction data mixed in to prevent catastrophic forgetting.

**DPO deep dive — the "Preference Tuning Flywheel" (from AI Playlist blog, Sep 2025):**

This is the most technically detailed preference optimization system published by any non-lab company:

```
  Stage 1: GENERATE
  Sample user queries from logs → produce diverse
  executable DSL plans (domain-specific language
  for playlist construction) → remove trivial variations
       │
       ▼
  Stage 2: SCORE
  Reward model estimates user preferences using
  engagement/retention signals (plays, skips, saves,
  refinements)
       │
       ▼
  Stage 3: SAMPLE
  Construct preference pairs with:
  ├── Margin constraints (only compare when one
  │   playlist is clearly better)
  └── Hard negatives (alternatives that come close
      but fall short)
  Pairs bucketed by difficulty (high/medium/low margin)
  to prevent overfitting
       │
       ▼
  Stage 4: FINE-TUNE
  DPO applied: increase probability of preferred
  responses while maintaining proximity to base model
```

**Key DPO design decisions:**
- Train on **final playlists** (not intermediate DSL plans) — aligning with actual user satisfaction
- User interactions (plays, skips, saves, refinements) treated as preference feedback
- Tool-pool sizing and caching provided gains **comparable to model improvements** — infrastructure optimization matters as much as training

**Verified metrics (with exact context):**

| Metric | Value | What it actually measures | Source |
|--------|-------|--------------------------|--------|
| Click-through improvement | **Up to 4x** | CTR on recs **with** LLM explanations vs. **without** (especially for niche content) | Dec 2024 |
| Domain task improvement | **14%** | Over baseline Llama across 10 Spotify-specific tasks | Dec 2024 |
| Listening time (A/B test) | **+4%** | AI Playlist with DPO vs. control | Sep 2025 |
| Tool call errors | **-70%** | Fewer erroneous DSL orchestration invocations after DPO | Sep 2025 |
| Episode recommendation | **1.96x** | Semantic ID system vs. baseline | Nov 2025 |
| Multi-task vs. single-task | **+22%** | Training on both search + recommendation vs. single task | Nov 2025 |
| Query expansion accuracy | **30.8% vs. 22.1%** | RSFT+DPO vs. no expansion on Natural Questions | Jul 2025 |
| Compute savings (query expansion) | **~70%** | AQE vs. generate-then-filter approach | Jul 2025 |

**The 4x engagement claim — what it actually is:** This is click-through rate on recommendations that include LLM-generated contextual explanations (e.g., "Dead Rabbitts' latest single is a metalcore adrenaline rush!") versus the same recommendations shown without explanations. It is NOT 4x overall Spotify engagement. The effect is especially strong for niche content where users lack context about unfamiliar artists.

**Products in production:**
- **AI DJ** — launched 2023, expanded to multiple markets. Real-time personalized commentary. Uses fine-tuned smaller Llama models.
- **AI Playlist** — beta 2024. Agentic LLM system: natural language prompt → DSL orchestration plan → tool calls to search/filter music → playlist. This is where DPO is applied.
- **Contextualized Recommendations** — LLM-generated narrative explanations for new releases, with human-in-the-loop (music editors review outputs).

**In production?** Yes — AI DJ, AI Playlist, and recommendation explanations are all in production. Semantic ID search and query expansion are research stage.

**Key learning:** Spotify's most interesting finding is that **infrastructure optimization (tool-pool sizing, caching) provided gains comparable to model improvements**. DPO on the model matters, but so does optimizing the tools the model calls. Also: the 4x engagement number comes from adding *explanations* to existing recommendations — it's about making recommendations understandable, not making better recommendations. Context matters as much as quality.

**Source:** [research.atspotify.com](https://research.atspotify.com) — Contextualized Recommendations (Dec 2024), Preference Optimization (Sep 2025), Semantic IDs (Nov 2025), Query Expansion (Jul 2025)

---

#### DraftWise — Three-Model Pipeline with Reinforcement Fine-Tuning

**Problem:** DraftWise helps lawyers draft contracts by searching through precedents and generating tailored language. Search quality requires understanding legal reasoning — semantic understanding of legal concepts, clause relevance, jurisdiction matching.

**Model + technique:** A **three-model pipeline** combining retrieval and reasoning:

```
  Lawyer asks: "Find a termination clause for a license
  agreement, friendly to my side"
       │
       ▼
  Model 1: Cohere Embed + Rerank
  Retrieves and ranks relevant contract examples
  with citations from precedent database
       │
       ▼
  Model 2: Fine-tuned o-series reasoning model (via RFT)
  Augments search results, reasons over them,
  identifies gaps, generates tailored contract language
  (trained to show its reasoning before executing)
       │
       ▼
  Model 3: Cohere Command
  Additional text generation as needed
       │
       ▼
  Output: drafted clause with citations
```

**How RFT works:** Unlike standard fine-tuning (train on fixed correct answers), RFT uses a **programmable grader that scores every candidate response**. The training algorithm shifts model weights so high-scoring outputs become more likely. DraftWise's grader evaluates whether the model's legal reasoning and output meet quality thresholds for the nuance of legal language.

**Infrastructure:** Azure AI Foundry, accessing fine-tuned o-series models (exact variant — o1, o3-mini, o4-mini — not disclosed) plus Cohere models through a unified SDK.

**Results:** 30% improvement in search result quality on internal benchmarks. 60% developer efficiency increase versus traditional methods. 300% API request growth in Q1 2025.

**In production?** Yes — powering Smart Draft (contract generation from examples) and Markup (intelligent contract review).

**Key learning:** RFT is the managed-service version of what Netflix built in-house. The three-model architecture (retrieve → reason → generate) is important: the reasoning model doesn't need to memorize law — it reasons over retrieved examples. This separates knowledge (retrieval) from judgment (RFT-trained reasoning), which is the right decomposition for legal work. The custom grader is the secret weapon — it encodes domain quality standards into the training loop.

**Source:** [microsoft.com/customers/story/draftwise](https://microsoft.com/customers/story/draftwise) and [cohere.com/customer-stories/draftwise](https://cohere.com/customer-stories/draftwise)

---

### Pattern E: "We Built from the Ground Up" — Custom Architectures

These companies built custom model architectures because their data or task structure doesn't fit the standard "text in, text out" paradigm. The models may use transformer components, but the overall architecture is purpose-built.

#### Stripe — Payments Foundation Model: 59% → 97% Card-Testing Detection Overnight

**Problem:** Payment fraud detection requires understanding transaction *sequences* — not just individual transactions, but patterns across time, merchants, geographies, and payment methods. Stripe's prior approach: individual ML models (XGBoost, logistic regression) trained on hand-selected discrete features (card BIN, ZIP code, merchant category code). Each product area (authorization, fraud, disputes) required its own task-specific model. A specialized card-testing model took **two years of engineering** to reach 80% attack reduction.

**What they built:** A **Payments Foundation Model (PFM)** — a transformer trained on structured financial transaction data, not natural language. Announced by Gautam Kedia (Head of Applied ML) at Stripe Sessions, May 2025. Payments are "like language in some ways (structural patterns similar to syntax and semantics, temporally sequential) and extremely unlike language in others (fewer distinct tokens, contextual sparsity)."

**Tokenization — the key design decision:**

```
  Transaction → "sentence" of tokens:
  │
  ├── Categorical features (card BIN, MCC, payment method)
  │     → Learned embedding vectors
  │       (discovers latent relationships between categories)
  │
  ├── Continuous features (amount, timestamps)
  │     → Binning into discrete buckets or
  │       positional base encoding
  │
  └── Textual features (merchant names, descriptors)
        → Standard BPE subword tokenizer

  User's transaction history → long token sequence
  (analogous to how LLMs process text documents)
```

**Self-supervised pre-training (three tasks, no manual labels):**
1. **Masked Feature Modeling** — mask random features within a transaction, predict from context (analogous to BERT's MLM)
2. **Contrastive Learning** — distinguish similar vs. dissimilar transaction sequences
3. **Next Transaction Prediction** — forecast subsequent transactions (analogous to GPT's autoregressive training)

**Training data:** Tens of billions of real-world Stripe transactions. Includes cross-product data from customers using Stripe's anti-fraud tools even without processing payments through Stripe.

**Primary output:** A dense behavioral embedding vector (hundreds of dimensions) for each transaction. Downstream classifiers consume these embeddings for specific tasks.

**The 64% improvement — what it actually means:**

| Metric | Before PFM | After PFM |
|--------|-----------|-----------|
| Card-testing attack detection (large businesses) | **59%** | **97%** |
| Relative improvement | — | **64%** ((97-59)/59) |
| Time to achieve | 2 years of specialized engineering | **Overnight** (upon deployment) |

The "64% improvement" is a **relative** improvement in detection rate: from 59% to 97% = 38 percentage points = 64% relative gain. The prior specialized model took two years to reach 80% attack reduction; the PFM achieved a larger gain upon deployment.

**How it works for card-testing specifically:** The transformer's self-attention identifies long-range dependencies — high velocity of transactions from a new device, correlations between seemingly unrelated IP addresses, sequential probing of card numbers. A downstream classifier ingests sequences of PFM embeddings and predicts whether a traffic slice is under attack.

**In production?** Yes. Powers Stripe Radar (fraud detection) and is integrated across authorization optimization, dispute management, and currency conversion. Performs "comparably or slightly better" than previous specialized systems for non-fraud tasks. Designed to "scale horizontally" with parallel processing.

**Not disclosed:** Exact parameter count, layer count, embedding dimensionality (only "hundreds"), context window, specific hardware, whether encoder-only (BERT-style) or decoder-only (GPT-style).

**Key learning:** The transformer architecture is more general than "language model." Stripe proved that self-supervised pre-training on structured sequential data (payments) produces universal embeddings that outperform years of task-specific feature engineering. The critical insight: instead of building separate models per fraud type, build one foundation model that produces embeddings usable for any downstream task. But this requires a custom architecture — you can't fine-tune a chat model for transaction sequences.

**Source:** Gautam Kedia's Stripe Sessions announcement (May 2025), [TechCrunch](https://techcrunch.com/2025/05/07/stripe-unveils-ai-foundation-model-for-payments/), [Analytics India Magazine](https://analyticsindiamag.com/global-tech/how-stripe-used-ai-to-boost-fraud-detection-from-59-97-overnight/)

---

#### Airbnb — Multi-Model Customer Support Pipeline (33% → 10% WER)

**Problem:** Airbnb's customer support handles millions of interactions across text and voice. They needed ASR, intent detection, article recommendation, agent assistance, and chatbot paraphrasing — all tuned to Airbnb-specific vocabulary ("listing," "superhost," "entire home," booking statuses).

**What they actually built:** Not a single 12B model — a **multi-model pipeline** where each component uses a different architecture optimized for its specific task:

```
  ┌─────────────────────────────────────────────────────────┐
  │  VOICE SUPPORT PIPELINE (in production)                 │
  │                                                         │
  │  Caller → IVR Greeting                                  │
  │    → ASR Transcription (WER: 33% → ~10%)                │
  │      ├── Domain-adapted ASR model + phrase list          │
  │      └── Fixes: "listing"≠"lifting",                    │
  │           "help with my stay"≠"happy Christmas Day"      │
  │    → Contact Reason Detection (sub-50ms, T-LEAF taxonomy)│
  │    → Decision: Self-Service or Escalation                │
  │      ├── Self-Service:                                   │
  │      │   Help Article Retrieval (30 candidates in ~60ms) │
  │      │   → LLM re-ranking model                         │
  │      │   → Paraphrasing (>90% precision)                 │
  │      │   → SMS/App notification with article link        │
  │      └── Escalation: Route to human agent with context   │
  │                                                          │
  ├──────────────────────────────────────────────────────────┤
  │  TEXT SUPPORT MODELS (each fine-tuned separately)        │
  │                                                          │
  │  1. Content Recommendation: XLMRoBERTa → MT5            │
  │     (reformulated classification as language generation) │
  │                                                          │
  │  2. Real-Time Agent Assistant: t5-base + Narrativa       │
  │     (combined annotation + logging-based training data)  │
  │                                                          │
  │  3. Chatbot Paraphrasing: T5                             │
  │     (millions of unsupervised labels from agent logs:    │
  │      "I understand that you..." contains paraphrase)     │
  │     (generic reply filtering via Sentence-Transformers   │
  │      text clustering)                                    │
  │                                                          │
  ├──────────────────────────────────────────────────────────┤
  │  INFRASTRUCTURE (for large model training)               │
  │                                                          │
  │  Ray + DeepSpeed + PyTorch on elastic AWS cluster        │
  │  8x A100 GPUs, 150 TFLOPS per GPU                       │
  │  Capability: models up to 12B parameters                 │
  │  (investigating 30B+ with model parallelism)             │
  └──────────────────────────────────────────────────────────┘
```

**The 33% → 10% WER correction:** This was NOT fine-tuning a 12B LLM — it was adapting an ASR model with a domain-specific phrase list ensuring Airbnb terminology is recognized correctly. The specific ASR provider was not disclosed. The 12B parameter capability is infrastructure — trained on 8x A100s with Ray + DeepSpeed — but the specific purpose of the 12B model was never disclosed publicly.

**Chatbot paraphrasing — clever unsupervised data creation:** Instead of labeling paraphrase data manually, they extracted natural paraphrases from agent conversation logs. When agents say "I understand that you want to cancel your reservation," that sentence contains a paraphrase of the user's problem. This generated **millions of labels** without manual annotation. They then used Sentence-Transformers clustering to identify and filter generic meaningless replies, producing a high-quality production model.

**Concrete numbers:**

| Component | Metric | Value |
|-----------|--------|-------|
| ASR | Word error rate | **33% → ~10%** |
| Intent detection | Latency | **sub-50ms** |
| Article retrieval | Speed | **30 candidates in ~60ms** |
| Paraphrasing | Precision | **>90%** |
| Training | GPU throughput | **150 TFLOPS per A100** |
| Training | Infrastructure | **8x A100, elastic Ray cluster** |
| Paraphrase data | Labels generated | **Millions** (unsupervised) |
| DeepSpeed | Training time reduction | **Weeks → days** |

**In production?** Yes — the voice support pipeline, content recommendation, agent assistant, and chatbot paraphrasing are all in production. The 12B training capability and 30B+ investigation are infrastructure milestones, not specific products.

**Key learning:** Airbnb's approach is the opposite of "one model to rule them all." Each task gets the right-sized model: sub-50ms intent detection doesn't need 8B parameters; paraphrasing quality needs T5 but not Llama. The unsupervised paraphrase data creation is the insight worth stealing — mining real conversations for natural paraphrases scales better than human annotation.

**Source:** [medium.com/airbnb-engineering](https://medium.com/airbnb-engineering) (customer support text generation), [airbnb.tech](https://airbnb.tech/ai-ml/) (voice support), [anyscale.com/blog](https://www.anyscale.com/blog/optimizing-llm-training-with-airbnbs-next-gen-ml-platform)

---

#### Stitch Fix — GPT-3 Fine-Tune: 10K Descriptions Every 30 Minutes

**Problem:** Stitch Fix needs personalized product descriptions for hundreds of thousands of clothing styles, plus ad copy for Facebook and Instagram campaigns. Human stylists wrote these initially — high quality, but a single campaign took ~2 weeks to plan, strategize, and draft.

**Model + technique:** GPT-3 full fine-tuning through OpenAI's API (2023, before LoRA was available via API). Trained on "several hundred" high-quality product descriptions written by in-house copywriters. Product attributes served as prompts; copywriter descriptions served as completions.

**How it works in production:**

```
  Product attributes + customer style profile
       │
       ▼
  Fine-tuned GPT-3 (trained on copywriter examples)
       │
       ├── Product descriptions (Freestyle product pages)
       │   10,000 descriptions every 30 minutes
       │   Review time: <1 minute per description
       │
       └── Ad copy (Facebook, Instagram)
           Previously: ~2 weeks per campaign
           Now: algorithmically generated, <1 min review/asset
           77% pass rate on first expert review
       │
       ▼
  Expert-in-the-loop: copywriters review + edit all output
```

**The "beat human writers" claim — what it actually means:** In a blinded quality assessment, AI-generated descriptions "achieved higher quality scores" than human-written ones. No specific numerical scores, statistical tests, or exact preference percentages were disclosed — only the qualitative claim that AI scored higher in blind review.

**Scale:**
- **10,000 product descriptions generated every 30 minutes**
- Descriptions span "hundreds of thousands of styles" across Freestyle
- Each description reviewed by expert in **under 1 minute**
- Ad copy: **77% first-pass approval rate** (23% need editing)
- Campaigns that took **~2 weeks** now take minutes to generate

**Other GPT uses at Stitch Fix:**
- GPT-4 client note recaps — summarizes clients' tens to hundreds of stylist notes into concise recaps (in production for stylists)
- 43 million outfit combinations showcased daily; 13 million new combinations generated daily (separate system, not GPT-based)

**In production?** Yes — product descriptions and ad copy are both in production at scale.

**Key learning:** Fine-tuning worked here because Stitch Fix had exactly the right data: expert-written examples from professional copywriters, not crowd-sourced labels. "Several hundred" examples was enough because the examples were uniformly high quality and the task is well-defined (product attributes → compelling description). The expert-in-the-loop model (generate → expert review → publish) catches the 23% of outputs that need editing while scaling the 77% that don't. This is a realistic production pattern: don't aim for 100% automation, aim for 77% and let experts handle the rest.

**Source:** [multithreaded.stitchfix.com/blog/2023/03/06/expert-in-the-loop-generative-ai-at-stitch-fix](https://multithreaded.stitchfix.com/blog/2023/03/06/expert-in-the-loop-generative-ai-at-stitch-fix)

---

### Pattern F: "Safety and Compliance First" — Guardrail Fine-Tuning

These companies fine-tune specifically for safety-related tasks: detecting toxic content, preventing prompt injection, ensuring output compliance, or adding safety guardrails to domain-specific applications.

#### Gaming Industry (AWS) — Toxic Speech Detection

**Problem:** Online gaming platforms need to detect toxic speech in real-time across chat, voice transcripts, and user-generated content. The challenge: gaming communication uses slang, abbreviations, and coded language that general toxicity classifiers miss.

**Model + technique:** BERTweet (BERT variant pre-trained on Twitter data) fine-tuned for toxic speech classification in gaming contexts.

**Data strategy:** Gaming-specific toxicity labels — examples of toxic behavior in gaming contexts, including coded language, gaming-specific slurs, and contextual toxicity (phrases that are toxic in gaming but innocent elsewhere).

**Results:** 88% precision on gaming toxicity detection — significantly higher than general-purpose toxicity classifiers on gaming-specific content.

**In production?** Yes, deployed as an AWS reference architecture for gaming companies.

**Key learning:** Toxicity is context-dependent. A general classifier misses gaming-specific patterns; a gaming-specific classifier catches them. This is a strong argument for domain-specific fine-tuning on classification tasks: when the distribution of your target concept differs from the training distribution, fine-tuning provides targeted improvement that prompt engineering cannot.

**Source:** [aws.amazon.com/blogs/machine-learning](https://aws.amazon.com/blogs/machine-learning) (gaming toxic speech detection)

---

#### Lasso Security — Prompt Injection Detection

**Problem:** As LLMs are deployed in production, prompt injection attacks become a real security threat. Detecting prompt injection requires understanding both the structure of injection attacks and the context of legitimate prompts — a classification task that evolves as attack techniques evolve.

**Model + technique:** Fine-tuned classifier for prompt injection detection. The specific model architecture isn't fully public, but the approach is SFT on labeled injection/non-injection examples.

**Data strategy:** Curated dataset of prompt injection attacks (jailbreaks, instruction overrides, data extraction attempts) paired with legitimate prompts. The challenge is that injection techniques evolve rapidly, requiring continuous data collection and retraining.

**Results:** High-precision prompt injection detection deployed as a security layer for LLM applications.

**In production?** Yes, as a security product.

**Key learning:** Security classification is a moving target — attack techniques evolve, and the classifier must evolve with them. This is one of the strongest cases for continuous fine-tuning: not a one-time training run, but an ongoing process of collecting new attack patterns and retraining.

**Source:** [lasso.security/blog](https://lasso.security/blog)

---

#### AT&T — 40% Accuracy Boost with Mistral 7B via NeMo

**Problem:** AT&T needed to improve accuracy on telecom-specific NLP tasks — network troubleshooting documentation, customer query routing, and technical knowledge retrieval.

**Model + technique:** Mistral 7B fine-tuned using NVIDIA NeMo framework with LoRA.

**Data strategy:** Telecom-specific instruction pairs from internal documentation and customer interaction logs.

**Results:** 40% accuracy improvement on their benchmark tasks compared to the base model.

**In production?** Yes.

**Key learning:** NVIDIA NeMo provides an enterprise-ready fine-tuning framework that handles the infrastructure complexity (distributed training, data processing, evaluation) — useful for companies that want to fine-tune but don't want to build training infrastructure from scratch.

**Source:** [developer.nvidia.com/blog](https://developer.nvidia.com/blog) (AT&T NeMo case study)

---

#### Shell — Chemistry Chatbot via NeMo

**Problem:** Shell needed a conversational AI for chemistry and materials science — answering questions about chemical processes, material properties, and safety procedures. General-purpose models lack depth in specialized chemistry and give unreliable answers on safety-critical topics.

**Model + technique:** Fine-tuned via NVIDIA NeMo for chemistry-domain question answering.

**Data strategy:** Domain-specific chemistry and materials science Q&A pairs, curated from internal knowledge bases and technical documentation.

**Results:** 30% accuracy improvement on chemistry-domain questions compared to the base model. For safety-critical applications, this improvement is significant — wrong answers about chemical processes can have physical consequences.

**In production?** Yes, internal deployment.

**Key learning:** Safety-critical domains (chemistry, medicine, engineering) have zero tolerance for hallucination on certain topics. Fine-tuning on domain-specific Q&A, combined with retrieval from verified knowledge bases, provides the accuracy floor these applications need.

**Source:** [developer.nvidia.com/blog](https://developer.nvidia.com/blog) (Shell NeMo case study)

---

## The Patterns Matrix — When to Apply What

Now that we've seen 24 companies in action, let's extract the patterns.

### Task Type → Best Approach

Based on the evidence from the case studies above:

| Task | Traditional ML | SFT Only | SFT + DPO/RL | Full Custom |
|------|---------------|----------|--------------|-------------|
| Binary classification | Often still best | When context matters | Rarely needed | No |
| Structured extraction | Good for fixed schemas | Better for evolving schemas | No | No |
| Text generation | N/A | Minimum viable | When quality matters | No |
| Recommendations | Collaborative filtering | Emerging use | Netflix/Spotify pattern | No |
| Code generation/repair | N/A | Essential baseline | For correctness | Replit |
| Fraud detection | Still strong baseline | Emerging | No | Stripe |
| Search/ranking | Embedding models | Reranker fine-tuning | DraftWise pattern | No |
| Multimodal tasks | Traditional CV+NLP | Multimodal SFT | Emerging | Shopify |

### The Classification Decision Ladder

One common question: when should I use traditional ML vs. a fine-tuned LLM for classification?

```
Simple, fixed classes, structured features
  → Logistic regression / XGBoost
    (Fast, interpretable, well-understood. Don't overthink it.)
      │
      ↓ (if accuracy plateaus or text is primary input)
      │
Moderate complexity, text-heavy
  → Fine-tuned BERT/encoder model
    (Gaming industry example: BERTweet for toxicity, 88% precision)
      │
      ↓ (if nuance matters, classes evolve, long context needed)
      │
Complex, evolving, long-context
  → Fine-tuned LLM 7B-8B + LoRA
    (Instacart: 96.4% precision on evolving intent taxonomy)
    (AT&T: 40% accuracy boost on telecom tasks)
      │
      ↓ (if quality beyond format needed)
      │
Quality-critical
  → Multi-stage SFT + DPO
    (Netflix: SFT → GRPO → DPO for recommendation quality)
    (Spotify: SFT + DPO + RLHF for 4x engagement)
```

Each step up the ladder brings more capability but also more complexity, cost, and maintenance burden. Don't climb higher than your task requires.

### The Fine-Tuning Technique Selection Flowchart

```
Do you have prompt-completion pairs?
  │
  ├── Yes → SFT (+ LoRA unless deep adaptation needed)
  │         │
  │         ├── Do you also have preference data (chosen/rejected)?
  │         │     → Yes → Add DPO after SFT
  │         │
  │         ├── Only binary feedback (thumbs up/down)?
  │         │     → Consider KTO (simpler data requirements)
  │         │
  │         └── Do you have verifiable rewards?
  │               (code passes tests, math is correct, engagement measured)
  │               → Yes → Consider GRPO or RFT
  │
  ├── No, but I have a domain corpus
  │     → Continued pre-training first, then SFT
  │       (Bloomberg, Replit pattern)
  │
  └── No data at all
        → Distill from a larger model
          (Generate training data with GPT-4/Claude, train small model)
          (Instacart, Checkr, Convirza pattern)
```

### When LoRA vs. Full Fine-Tuning

```
Default: LoRA
  │
  ├── Unless: multimodal task requiring deep adaptation → Full FT (Shopify)
  ├── Unless: building from scratch with domain pre-training → Full FT (Bloomberg)
  ├── Unless: very limited GPU budget → QLoRA (4-bit quantization)
  └── Unless: multi-task serving with shared base → Multi-LoRA (Instacart, Uber)
```

Anyscale's benchmark data shows LoRA matching full fine-tuning quality in 90%+ of cases tested on the Llama-2 model family. The cases where full fine-tuning wins are when you need the model to learn fundamentally new capabilities (new modalities, new domains from scratch) rather than adapting existing ones.

---

## Best Practices from Reliable Sources

These best practices are drawn from companies and research labs that have fine-tuned hundreds or thousands of models and published their findings.

### Anthropic

**Core position:** Start with prompting. Fine-tune only when prompting demonstrably fails at your task.

Anthropic's guidance emphasizes that evaluation is the bottleneck in fine-tuning — most teams underinvest in evaluation and overinvest in training. Without a reliable evaluation framework, you can't tell whether fine-tuning actually helped or whether you just overfit to your test set.

### OpenAI

**Key guidance:**
- 50-100 high-quality examples is the minimum for noticeable improvement
- Fine-tuning is for style and format, not knowledge injection
- New: Reinforcement Fine-Tuning (RFT) for tasks with verifiable rewards
- DPO available through API for preference alignment

**What they've learned:** Their fine-tuning documentation explicitly states that fine-tuning works best for behavioral adaptation (output format, tone, style) and poorly for knowledge injection (teaching the model new facts). For knowledge, they recommend RAG.

**Source:** OpenAI fine-tuning documentation and DPO cookbook

### Anyscale

**Core position:** "Fine-tuning is for form, not facts."

This is the most concise summary of when fine-tuning works. Anyscale's extensive benchmarks with Llama-2 showed:
- LoRA handles 90%+ of fine-tuning use cases as well as full fine-tuning
- The gap between LoRA and full fine-tuning appears primarily in deep domain adaptation
- Data quality matters more than quantity — 1,000 excellent examples outperform 100,000 mediocre ones

**Source:** Anyscale blog (fine-tuning benchmarks and best practices)

### Modal

**Core position:** "Start small: 20-50 best examples."

Modal's guidance for practitioners starting out:
- Begin with your 20-50 best examples. If the model improves, add more.
- Iterate on data quality before scaling quantity — fix bad labels before adding new ones
- If 50 perfect examples don't move the needle, the problem likely isn't solvable with fine-tuning

**Source:** Modal blog

### OpenPipe

**"Ten Commandments of Fine-Tuning in Production":**

1. Use production data, not synthetic data alone
2. Always reserve a test set
3. Monitor continuously — accuracy drifts
4. Version your models and datasets
5. A/B test before full rollout
6. Measure cost alongside quality
7. Log everything (inputs, outputs, user feedback)
8. Retrain on a schedule, not ad hoc
9. Start with the smallest model that works
10. Don't fine-tune what prompting can solve

**Source:** [openpipe.ai/blog/the-ten-commandments-of-fine-tuning-in-prod](https://openpipe.ai/blog/the-ten-commandments-of-fine-tuning-in-prod)

### Predibase

**Key insights:**
- Multi-LoRA serving: one base model, many adapters. Under $8 per adapter for serving costs.
- LoRA Land benchmarks showed small fine-tuned models matching or exceeding GPT-4 on specific tasks
- RL (reward-based optimization) can beat SFT when labeled data is scarce but reward signals are available

**Source:** [predibase.com/blog/lora-land](https://predibase.com/blog/lora-land) and Predibase RL blog

### Databricks

**Scale evidence:** Over 200,000 custom models trained by Databricks customers in a single year through their Mosaic AI platform. This is the strongest evidence that fine-tuning has moved from research novelty to production standard — though it also raises the question of how many of those models actually outperform prompted baselines.

**Source:** Databricks blog (Mosaic AI training)

### Synthesized Best Practices

| Practice | Why | Who Proved It |
|----------|-----|---------------|
| Prompt first, fine-tune second | Most problems don't need fine-tuning | Honeycomb ($30/month), Anthropic |
| Start with 20-50 golden examples | Quality > quantity for initial signal | Modal, OpenPipe |
| Use LoRA unless you have a reason not to | 90%+ of cases, 10x memory savings | Anyscale, Predibase |
| Evaluate on YOUR task, not benchmarks | MMLU ≠ production quality | Grammarly (beat GPT-3 on editing, not on MMLU) |
| Build for re-training, not one-shot | Models drift, tasks evolve, data changes | FinGPT (financial models stale in weeks) |
| Distill big → serve small | Best cost-performance tradeoff | Instacart, Walmart, Checkr |
| Add DPO/RL only after SFT plateaus | Don't over-engineer early | Netflix evolved to GRPO only after SFT wasn't enough |
| Log production data from day one | Your best training data is production usage | OpenPipe commandment #7 |

---

## The Cost Reality

Fine-tuning costs vary by 1000x depending on approach. Here's what companies have actually reported:

### Reported Costs by Company

| Company | Approach | Reported Cost / Infrastructure | What They Got |
|---------|----------|-------------------------------|---------------|
| Bloomberg | 50.6B from scratch | 1.3M GPU hours, 512 A100s, ~53 days (~$3M) | 62.51 avg on financial benchmarks (vs. 54.35 for BLOOM-176B) |
| Shopify | Qwen2VL 7B full SFT | Triton + Kafka + FP8 serving stack | 40M inferences/day, 16B tokens/day, 500ms latency |
| Replit | 7B code repair + 2.7B code gen | 8 H100s (repair) + 256 A100s (gen) | Matches GPT-4 on real error fixes |
| Writer | Palmyra family (128M–70B) + DPO | ~$1M GPU (X5), NVIDIA TensorRT-LLM | 73% CFA Level III (GPT-4: 33%), 85.9% medical avg |
| Harvey | Custom-trained via OpenAI | Azure OpenAI + 10B tokens legal data | 0.2% hallucination rate, 97% lawyer preference, $100M ARR |
| Instacart | Llama-3-8B + LoRA | A100 → H100 upgrade for 300ms latency | 96.4% precision, 50% complaint reduction |
| Uber | Multi-model platform | On-prem A100 + cloud H100, 32 GPUs | GPT-4 comparable at scale cost |
| Checkr | Llama-3-8B LoRA via Airtrain | Platform fees | 5x cost reduction, 30x speed vs GPT-4 |
| Convirza | Llama-3-8B LoRA via Predibase | <$8/adapter to serve | 10x cost reduction vs OpenAI |
| Grammarly | FLAN-T5 (770M–11B) | Internal compute | Preferred 6.4:1 over GPT-3 175B |
| Stripe | Custom transformer on payments | Tens of billions of transactions | 59% → 97% card-testing detection overnight |

### The a16z Counterpoint

a16z's 2025 enterprise AI report sounds a cautionary note: most enterprises attempting fine-tuning are not seeing positive ROI. The costs they cite include:

- **Data collection and curation** — often the largest hidden cost. Expert-labeled data (Harvey's lawyers, Stitch Fix's stylists) is expensive.
- **Training compute** — significant for full fine-tuning, manageable for LoRA
- **Evaluation infrastructure** — building reliable evaluations is as expensive as training
- **Ongoing maintenance** — models drift, base models update, data distributions shift
- **Opportunity cost** — engineering time spent on fine-tuning vs. improving prompts or adding RAG

The companies that succeed with fine-tuning share two characteristics: (1) they have a specific, measurable task where fine-tuning outperforms prompting, and (2) they have enough volume to justify the fixed costs of training and infrastructure.

### Break-Even: Self-Hosted vs. API

The rough calculation:

```
API cost per inference:       $0.001 - $0.01 (depending on model/tokens)
Self-hosted cost per inference: $0.0001 - $0.001 (after GPU amortization)

Break-even volume (rough):
  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  API cheaper          Self-hosted cheaper        │
  │  ◄──────────────┼──────────────────────►        │
  │                 │                               │
  │          ~100K-1M inferences/day                │
  │          (depends on token length,              │
  │           GPU costs, model size)                │
  │                                                 │
  └─────────────────────────────────────────────────┘

  Shopify:    40M/day    → self-hosted, no question
  Instacart:  high       → self-hosted makes sense
  Harvey:     low-medium → API fine-tuning, reasonable
  DraftWise:  low-medium → API RFT, reasonable
```

The formula is simple: `(GPU monthly cost) / (inferences per month) < (API cost per inference)`. But the hidden variable is engineering time — self-hosting requires infrastructure maintenance that API providers handle for you.

---

## What Can Go Wrong

Fine-tuning failure modes, drawn from documented cases:

### Overfitting

The most common failure mode. With small datasets (common in domain-specific fine-tuning), the model memorizes training examples instead of learning generalizable patterns. Symptoms: near-perfect training accuracy, poor performance on held-out data.

**Mitigation:** Hold out 10-20% of data for validation. Monitor validation loss — stop training when it starts increasing. LoRA naturally regularizes by limiting the number of trainable parameters.

For a deeper explanation of overfitting mechanics in fine-tuning, see the [engineer's guide](sft-from-scratch-engineer.md#catastrophic-forgetting-the-cost-of-new-behavior).

### Catastrophic Forgetting

Fine-tuning overwrites pre-trained knowledge. A model fine-tuned for legal text might forget how to do basic math. A model fine-tuned for customer support might lose its ability to summarize long documents.

**Mitigation:** LoRA (by freezing base weights), mixing pre-training data into the fine-tuning dataset, and keeping fine-tuning runs short. The [engineer's guide](sft-from-scratch-engineer.md#catastrophic-forgetting-the-cost-of-new-behavior) demonstrates this empirically with our Shakespeare model.

### Fine-Tuning Theater

Companies fine-tuning when prompting would work. This is more common than any technical failure. Signs you're doing fine-tuning theater:
- You haven't tried systematic prompt engineering with examples
- You haven't tried RAG for knowledge-dependent tasks
- Your baseline comparison is "zero-shot GPT-4" instead of "GPT-4 with good prompts and few-shot examples"
- Your fine-tuned model barely outperforms a well-prompted baseline

**Mitigation:** Before any fine-tuning project, establish a strong prompted baseline. If few-shot prompting with 5-10 examples gets you 90% of the way there, the marginal improvement from fine-tuning may not justify the cost.

### Temporal Drift

Models become stale. FinGPT (open-source financial LLM) demonstrated this clearly: financial models need retraining on the order of weeks, not months, because market conditions, terminology, and relevant precedents change continuously.

**Mitigation:** Build retraining pipelines from day one. Log production data and user feedback for future training rounds. Monitor production accuracy and set alerts for degradation.

### Security Risks

Fine-tuned models can inadvertently memorize and leak training data. They can also be more susceptible to prompt injection if fine-tuning weakens safety training. Lasso Security's work on prompt injection detection highlights the evolving threat landscape.

**Mitigation:** Don't fine-tune on data you wouldn't want the model to output. Apply safety fine-tuning after domain fine-tuning. Test for data extraction attacks before deployment.

---

## Conclusion + Quick Reference

### Three Questions Before You Fine-Tune

1. **Have you exhausted prompting + RAG?** If not, do that first. It's cheaper, faster, and works for most use cases.

2. **Is this a form problem or a facts problem?** Fine-tuning fixes form (output structure, style, behavior). RAG fixes facts (domain knowledge, current information). Doing the wrong one wastes time and money.

3. **Do you have enough volume to justify the investment?** At 1,000 inferences/day, API fine-tuning might work. At 100,000+/day, self-hosting a fine-tuned model pays for itself. In between, do the math.

### Quick Reference Card

| I want to... | Start with... | Then consider... | Case study |
|--------------|--------------|-----------------|------------|
| Reduce API costs | Distill to 7-8B + LoRA | Multi-LoRA for multiple tasks | Checkr, Convirza |
| Specialize for my domain | SFT on domain data | Continued pre-training if deep adaptation needed | Harvey, Grammarly |
| Improve output quality | SFT first | Add DPO/GRPO if quality plateaus | Netflix, Spotify |
| Handle multiple tasks | LoRA adapters per task | Shared base + adapter swapping | Instacart, Uber |
| Classify at scale | BERT/encoder for simple | LLM + LoRA for complex/evolving | Gaming (BERTweet), AT&T |
| Build from scratch | Full pre-training | Only if domain is truly unique | Bloomberg, Replit |
| Safety/compliance | Fine-tune classifier | Continuous retraining as threats evolve | Lasso Security, Gaming |

### Further Reading

- **How SFT works mechanically:** [SFT from Scratch: The Complete Implementation Guide](sft-from-scratch-engineer.md) — loss masking, chat templates, training loops
- **When to fine-tune (builder perspective):** [The Builder's Guide to Fine-Tuning](sft-from-scratch-builder.md) — decision frameworks, cost considerations
- **The full post-training pipeline:** [Post-Training Roadmap](post-training-roadmap.md) — SFT → RLHF → DPO explained with code

---

## References

### Company Engineering Blogs and Case Studies

- **Shopify:** [shopify.engineering/leveraging-multimodal-llms](https://shopify.engineering/leveraging-multimodal-llms)
- **Instacart:** [instacart.com/company/tech-innovation/building-the-intent-engine](https://www.instacart.com/company/tech-innovation/building-the-intent-engine) — also available at [tech.instacart.com](https://tech.instacart.com/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms-3ac8051ae7ac)
- **Uber:** [uber.com/blog/open-source-and-in-house-how-uber-optimizes-llm-training](https://www.uber.com/blog/open-source-and-in-house-how-uber-optimizes-llm-training)
- **Netflix:** [netflixtechblog.com/scaling-llm-post-training-at-netflix](https://netflixtechblog.com/scaling-llm-post-training-at-netflix-0046f8790194) and [arxiv.org/abs/2601.02764](https://arxiv.org/abs/2601.02764) (artwork personalization paper)
- **Walmart:** [tech.walmart.com](https://tech.walmart.com) (product catalogs blog)
- **Spotify:** [research.atspotify.com](https://research.atspotify.com) — Contextualized Recommendations (Dec 2024), Preference Optimization (Sep 2025), Semantic IDs (Nov 2025), Query Expansion (Jul 2025)
- **Grammarly:** [grammarly.com/blog/engineering/coedit-text-editing](https://grammarly.com/blog/engineering/coedit-text-editing) and [arxiv.org/abs/2305.09857](https://arxiv.org/abs/2305.09857) (EMNLP 2023 paper). Code: [github.com/vipulraheja/coedit](https://github.com/vipulraheja/coedit)
- **Replit:** [blog.replit.com/llm-training](https://blog.replit.com/llm-training) and [blog.replit.com/code-repair](https://blog.replit.com/code-repair). Model: [huggingface.co/replit/replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b)
- **Bloomberg:** [arxiv.org/abs/2303.17564](https://arxiv.org/abs/2303.17564) (BloombergGPT paper)
- **Harvey AI:** [openai.com/index/harvey](https://openai.com/index/harvey), [harvey.ai/blog](https://harvey.ai/blog), and [microsoft.com/customers/story/harvey](https://www.microsoft.com/en/customers/story/19750-harvey-azure-open-ai-service)
- **Stitch Fix:** [multithreaded.stitchfix.com/blog/2023/03/06/expert-in-the-loop-generative-ai-at-stitch-fix](https://multithreaded.stitchfix.com/blog/2023/03/06/expert-in-the-loop-generative-ai-at-stitch-fix)
- **Stripe:** [TechCrunch](https://techcrunch.com/2025/05/07/stripe-unveils-ai-foundation-model-for-payments/), [Analytics India Magazine](https://analyticsindiamag.com/global-tech/how-stripe-used-ai-to-boost-fraud-detection-from-59-97-overnight/)
- **Airbnb:** [medium.com/airbnb-engineering](https://medium.com/airbnb-engineering) (customer support text generation), [airbnb.tech](https://airbnb.tech/ai-ml/) (voice support), [anyscale.com/blog](https://www.anyscale.com/blog/optimizing-llm-training-with-airbnbs-next-gen-ml-platform) (LLM training infrastructure)
- **DraftWise:** [microsoft.com/customers/story/draftwise](https://microsoft.com/customers/story/draftwise) and [cohere.com/customer-stories/draftwise](https://cohere.com/customer-stories/draftwise)
- **Writer:** [writer.com/blog/palmyra](https://writer.com/blog/palmyra) and [developer.nvidia.com/blog](https://developer.nvidia.com/blog) (Writer Med & Fin release)
- **Convirza:** [predibase.com/blog/convirza-case-study](https://predibase.com/blog/convirza-case-study)

### Platform and Framework References

- **Predibase / LoRA Land:** [predibase.com/blog/lora-land](https://predibase.com/blog/lora-land)
- **Anyscale:** [anyscale.com/blog](https://anyscale.com/blog) (fine-tuning benchmarks)
- **OpenPipe:** [openpipe.ai/blog/the-ten-commandments-of-fine-tuning-in-prod](https://openpipe.ai/blog/the-ten-commandments-of-fine-tuning-in-prod)
- **NVIDIA NeMo (AT&T, Shell, Amdocs):** [developer.nvidia.com/blog](https://developer.nvidia.com/blog)
- **Databricks:** [databricks.com/blog](https://databricks.com/blog) (Mosaic AI training)
- **AWS (gaming toxicity):** [aws.amazon.com/blogs/machine-learning](https://aws.amazon.com/blogs/machine-learning)
- **Lasso Security:** [lasso.security/blog](https://lasso.security/blog)

### Industry Reports

- **a16z:** [a16z.com/ai-enterprise-2025](https://a16z.com/ai-enterprise-2025) (enterprise AI report)
- **Menlo Ventures:** [menlovc.com/perspective/2025-mid-year-llm-market-update](https://menlovc.com/perspective/2025-mid-year-llm-market-update)
- **ZenML:** [zenml.io/blog](https://zenml.io/blog) (LLMOps database)

### Technique Papers and Documentation

- **GRPO:** DeepSeek R1 paper
- **KTO:** Contextual AI blog
- **DPO:** OpenAI DPO cookbook
- **RFT:** OpenAI Reinforcement Fine-Tuning documentation
- **FinGPT:** Open-source financial LLM (temporal drift evidence)
