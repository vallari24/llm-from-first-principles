# LLMs in Recommendation Systems: How Industry Actually Does It

*Architecture patterns, verified case studies, and an honest comparison with traditional RecSys — across ads, video, music, jobs, food, travel, e-commerce, and more.*

> **For the fine-tuning mechanics** behind many of these systems — SFT, DPO, GRPO, distillation, LoRA — see the companion post: [Fine-Tuning in Production](fine-tuning-in-production.md).

---

**Table of Contents**

1. [Why This Post Exists](#why-this-post-exists)
2. [The Traditional RecSys Stack](#the-traditional-recsys-stack)
3. [Five Architecture Patterns for LLMs in Recs](#five-architecture-patterns-for-llms-in-recs)
4. [Category Deep Dives](#category-deep-dives)
5. [Mega-Scale Custom Architectures](#mega-scale-custom-architectures)
6. [LLM vs Traditional — The Honest Comparison](#llm-vs-traditional--the-honest-comparison)
7. [Research Frontier](#research-frontier)
8. [Best Practices](#best-practices)
9. [Summary Table](#summary-table)
10. [Conclusion](#conclusion)
11. [References](#references)

---

## Why This Post Exists

Every tech company is bolting LLMs onto their recommendation systems right now. The discourse is split into two camps: "LLMs will replace all of RecSys" and "LLMs are too slow and expensive for recommendations." The truth is more nuanced than either side admits.

This post maps out what's actually happening in production. Not what a research paper proposes, not what a startup's landing page claims — what companies with hundreds of millions of users have actually shipped, measured, and written about publicly.

A note on methodology: every case study here is traced back to a public engineering blog post, published paper, or official company announcement. If I couldn't find a verified source, it's not in here. Many companies use LLMs in their recommendation stack but haven't published details — those are excluded.

What you'll find:
- **Five architecture patterns** for integrating LLMs into recommendations, each with different latency/quality tradeoffs
- **Deep dives across 10+ categories** — video, music, e-commerce, ads, jobs, food, travel, visual, news, and conversational
- **An honest comparison** of when LLMs beat traditional methods and when they don't
- **Practical guidance** for choosing the right pattern

What this post is *not*: a tutorial on building a recommendation system from scratch. It assumes you know the basics of collaborative filtering, embeddings, and two-stage retrieval pipelines. If you want to understand the fine-tuning techniques referenced here (SFT, DPO, GRPO, distillation), see [Fine-Tuning in Production](fine-tuning-in-production.md).

---

## The Traditional RecSys Stack

Before we can understand what LLMs change, we need to understand what they're replacing — or more often, augmenting.

### The Core Methods

| Method | How It Works | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **Collaborative Filtering** | "Users who liked X also liked Y" — matrix factorization on user-item interactions | Simple, fast, proven at scale | Cold-start problem, no content understanding |
| **Content-Based** | Match user profile features to item features | Works for new items | Limited discovery, feature engineering heavy |
| **Two-Tower** | Separate neural networks encode user and item into same embedding space, dot-product similarity | Fast retrieval (ANN), handles both signals | Interaction features lost in dot product |
| **DLRM (Deep Learning Rec Model)** | Meta's architecture: sparse embeddings for categorical features + dense MLP | Handles sparse features well, proven at Meta scale | Complex infrastructure, feature engineering |
| **Sequential Models** | GRU4Rec, SASRec, BERT4Rec — model user behavior as a sequence | Captures temporal patterns | Training complexity, limited to interaction history |

### The Classic Three-Stage Pipeline

Nearly every production recommendation system follows this pattern:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL RECSYS PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Stage 1: RETRIEVAL          Stage 2: RANKING       Stage 3: RERANK│
│  ┌─────────────────┐        ┌──────────────┐       ┌─────────────┐ │
│  │ Candidate Gen    │───────▶│ Scoring Model │──────▶│ Business    │ │
│  │                  │        │              │       │ Rules +     │ │
│  │ • ANN search     │  ~1000 │ • Two-tower  │ ~100  │ Diversity   │ │
│  │ • Collaborative  │ items  │ • Cross-net  │ items │ • Dedup     │ │
│  │ • Popular items  │        │ • xgboost    │       │ • Freshness │ │
│  │ • Rules          │        │              │       │ • Fairness  │ │
│  └─────────────────┘        └──────────────┘       └─────────────┘ │
│                                                                     │
│  Latency: ~5ms               Latency: ~3ms         Latency: ~2ms  │
│  Corpus: millions             Candidates: ~1000     Candidates: ~100│
│  Output: ~1000                Output: ~100          Output: ~20     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
          Total latency budget: <50ms (often <10ms for ads)
```

**The key tension**: traditional systems are *fast*. A retrieval + ranking pipeline at YouTube, Meta, or LinkedIn serves results in under 10 milliseconds. LLMs, even small ones, typically need 50-500ms per inference. This latency gap is the central engineering challenge that every LLM-in-recs deployment must solve.

The solutions fall into five distinct architecture patterns.

---

## Five Architecture Patterns for LLMs in Recs

Every production LLM-in-recs deployment we surveyed fits one of these five patterns. They differ in *where* the LLM sits in the pipeline and *when* it runs (online vs offline).

### Pattern Overview

| Pattern | How It Works | Latency Impact | Example Companies |
|---------|-------------|---------------|-------------------|
| **A: Reranker** | LLM reranks top-K from traditional pipeline | High (online) | Netflix |
| **B: Feature Extractor** | LLM generates embeddings offline, fed to traditional ranker | None (offline) | LinkedIn JUDE, Pinterest |
| **C: Generator** | LLM directly generates item IDs via semantic IDs | High (online) | Spotify Text2Tracks, YouTube LRM |
| **D: Conversational** | Chat interface with retrieval + generation | High (online) | Amazon Rufus, Booking.com |
| **E: User/Item Understanding** | LLM builds rich natural-language profiles | None (offline) | DoorDash Cx, Meta GEM |

### Pattern A: LLM as Reranker

```
Traditional Pipeline                    LLM Reranker
┌──────────┐    ┌──────────┐           ┌──────────────┐
│ Retrieval │───▶│ Ranking  │──top-K──▶│ LLM Reranks  │──▶ Final List
│ (~1000)   │    │ (~100)   │  (20-50) │ (understands │
└──────────┘    └──────────┘           │  content +   │
                                        │  context)    │
                                        └──────────────┘
```

The LLM only sees a small candidate set (20-50 items), so inference cost is bounded. The traditional pipeline handles the heavy lifting of narrowing millions of items down to a manageable set.

**Tradeoff**: adds 100-500ms of online latency. Works when users tolerate slightly slower responses (browsing, not real-time bidding).

**Production example**: Netflix uses a fine-tuned Llama 3.1 8B to rerank artwork candidates for title recommendations.

### Pattern B: LLM as Feature Extractor

```
OFFLINE                                    ONLINE
┌──────────┐     ┌──────────────┐         ┌──────────────┐
│ Items /   │────▶│ LLM generates│────────▶│ Traditional  │
│ Users     │     │ embeddings   │  stored │ Ranking with │──▶ Results
│           │     │ or features  │  in DB  │ LLM features │
└──────────┘     └──────────────┘         └──────────────┘
                  (batch, async)           (fast, <10ms)
```

The LLM runs offline on a batch schedule. It produces embeddings or features that get stored and served as inputs to a fast traditional model. Zero online latency impact.

**Tradeoff**: features may be stale (updated hourly/daily). Can't capture real-time context.

**Production examples**: LinkedIn JUDE generates job embeddings with LoRA-finetuned Mistral 7B. Pinterest uses CLIP embeddings for visual similarity.

### Pattern C: LLM as Generator

```
User Query ──▶ ┌──────────────────────┐ ──▶ Semantic IDs ──▶ Item Lookup
               │ LLM generates item   │
               │ IDs as token sequence │
               │ (trained on semantic  │
               │  ID vocabulary)       │
               └──────────────────────┘
```

The most radical pattern: the LLM *directly produces* recommended items as a sequence of tokens. Items are assigned "semantic IDs" — learned token sequences that capture item similarity. The LLM generates these IDs autoregressively, essentially "writing" a recommendation.

**Tradeoff**: requires a custom vocabulary of semantic IDs. Inference is generative (slow). But it handles cold-start remarkably well because it reasons about item content, not just interaction history.

**Production examples**: Spotify Text2Tracks generates playlist candidates from natural-language descriptions. YouTube's LRM (Large Recommendation Model) uses Gemini-adapted architecture with RQ-VQE semantic IDs.

### Pattern D: LLM as Conversational Recommender

```
User ──▶ ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
         │ Intent +      │───▶│ Retrieval     │───▶│ LLM generates│──▶ Response
         │ Preference    │    │ (catalog,     │    │ personalized │
         │ Extraction    │    │  reviews,     │    │ recommendation│
         └──────────────┘    │  knowledge)   │    │ with reasoning│
              ▲               └───────────────┘    └──────────────┘
              │                                           │
              └───────────────────────────────────────────┘
                        Multi-turn refinement
```

The LLM is the interface itself. Users describe what they want in natural language, and the system retrieves candidates, reasons about them, and generates a personalized response. Multi-turn dialogue allows progressive refinement.

**Tradeoff**: high latency (1-5 seconds), but users expect it in a chat context. Great for complex decisions (travel, high-consideration purchases). Poor for casual browsing.

**Production examples**: Amazon Rufus for product recommendations. Booking.com AI Trip Planner for travel planning.

### Pattern E: LLM for User/Item Understanding

```
OFFLINE                                     ONLINE
┌──────────────┐    ┌──────────────────┐   ┌──────────────┐
│ User history  │───▶│ LLM generates    │──▶│ Traditional  │
│ Item metadata │    │ rich text         │   │ pipeline uses│──▶ Results
│ Reviews       │    │ profiles:         │   │ profiles as  │
│ Behavior logs │    │ "Likes spicy     │   │ features     │
│               │    │  Thai, orders    │   │              │
│               │    │  late, budget-   │   └──────────────┘
└──────────────┘    │  conscious"      │
                     └──────────────────┘
```

The LLM synthesizes diverse signals (behavior, reviews, metadata) into rich natural-language profiles that capture nuance traditional feature vectors miss. These profiles are then used as features in the existing recommendation pipeline.

**Tradeoff**: requires careful prompt engineering and periodic regeneration. The profiles can capture subtlety ("prefers hole-in-the-wall restaurants over chains, but makes exceptions for sushi") that no feature vector could represent.

**Production examples**: DoorDash Consumer Profiles (Cx) builds 360-degree preference narratives. Meta's GEM system generates ad understanding profiles.

---

## Category Deep Dives

### 4.1 Video & Movie Recommendations

**Netflix — LLM-Powered Artwork Personalization**

Netflix's most detailed public deployment of LLMs for recommendations focuses on personalizing which artwork (thumbnail) to show for each title, per user.

Architecture:
- **Base model**: Llama 3.1 8B
- **Training pipeline**: SFT → distillation from Qwen3-32B teacher → DPO alignment
- **Data**: 110K (user context, artwork set, selection) tuples
- **Pattern**: Reranker (Pattern A) — LLM reranks artwork candidates generated by the traditional pipeline

Results:
- SFT alone: 5% improvement over baseline
- Adding DPO: +3% on top of SFT
- Revenue impact: 0.5% lift (significant at Netflix's scale)
- In production since May 2025

Key engineering decisions:
- Used distillation from 32B teacher to improve the 8B student — a common pattern when you need production-viable latency
- DPO was chosen over RLHF because artwork preferences are naturally expressed as pairwise comparisons ("this thumbnail is better than that one for this user")
- Also experimented with GRPO using verifiable rewards

What makes this notable: Netflix published one of the most detailed accounts of LLM fine-tuning for recommendations, including the full training pipeline, data sizes, and incremental improvements from each technique.

Source: [arxiv.org/abs/2601.02764](https://arxiv.org/abs/2601.02764)

---

**YouTube / Google — Large Recommendation Model (LRM)**

Google's approach is architecturally the most ambitious: adapt Gemini (their foundation model) into a recommender that *generates* item recommendations as token sequences.

Architecture:
- **Base**: Gemini-adapted model
- **Key innovation**: Semantic ID tokenization via RQ-VQE (Residual-Quantized Variational Quantized Embeddings)
- **Pattern**: Generator (Pattern C) — the model generates semantic IDs of recommended videos

How semantic IDs work:
```
Video: "Making perfect sourdough bread at home"
  ↓ (content encoder)
Dense embedding: [0.23, -0.15, 0.87, ...]
  ↓ (RQ-VQE quantization)
Semantic ID: [42, 187, 3, 91]    ← a sequence of tokens from a learned codebook
```

Similar videos get similar semantic IDs, so the LLM can generalize to unseen items by reasoning about the token patterns.

Results:
- Excellent cold-start performance (can recommend new videos with no interaction history)
- Required 95%+ cost reduction from the baseline to be production-viable
- Distillation was essential to meet latency and cost constraints

Source: Google Research publications on semantic IDs and large recommendation models

---

**TikTok — Multimodal Video Understanding**

TikTok's recommendation system processes billions of recommendations daily using their Monolith architecture, with LLMs increasingly used for:
- Multimodal video understanding (extracting content features from video, audio, text overlays)
- Content categorization and safety classification
- Cross-lingual content matching

The core ranking system remains a traditional deep learning pipeline, but LLM-generated features are increasingly important inputs to that pipeline (Pattern B/E hybrid).

Source: TikTok engineering publications, Monolith system paper

---

### 4.2 Music Recommendations

**Spotify — Three LLM Applications**

Spotify has published the most comprehensive set of LLM-for-recommendations research across three distinct applications:

**1. AI DJ (RLHF/DPO-trained commentary)**

The AI DJ doesn't just select songs — it generates spoken commentary explaining *why* it picked them. The commentary model is fine-tuned with preference optimization (RLHF and DPO) to match Spotify's editorial voice.

**2. Text2Tracks (Generative Retrieval)**

The most architecturally interesting application. Text2Tracks is a generative retrieval system where the model generates track recommendations directly from natural-language descriptions.

```
Input:  "upbeat songs for a morning run, nothing too aggressive"
          ↓
Model:  Generates semantic IDs of matching tracks
          ↓
Output: [Track A, Track B, Track C, ...]
```

This uses Pattern C (Generator) with semantic IDs — similar to YouTube's approach but for music. The model learns a mapping from track content/vibe to semantic token sequences, then generates those sequences autoregressively.

**3. Narrative Explanations**

LLMs generate natural-language explanations for why a track or playlist was recommended. Not just "because you listened to X" — but contextual narratives that reference the user's listening patterns, time of day, and mood.

Results across all three applications:
- Explained recommendations: **4x CTR** compared to unexplained
- Listening time: **+4% increase** when recommendations include narrative explanations
- Text2Tracks enables new discovery surfaces impossible with traditional collaborative filtering

Source: [research.atspotify.com](https://research.atspotify.com) — publications on Text2Tracks, narrative recommendations, semantic IDs, and preference alignment

---

### 4.3 E-Commerce Product Recommendations

**Amazon Rufus — Conversational Product Recommendations**

Rufus is Amazon's conversational shopping assistant, deployed across the Amazon app. It represents Pattern D (Conversational Recommender) at massive scale.

Architecture:
- **Models**: Custom Nova models, with multi-model strategy (Nova, Claude, custom models selected per query type)
- **Training data**: Full product catalog + 51M customer reviews + Q&A data
- **Retrieval**: RAG with novel evidence sources + RL for response quality
- **Memory**: Account memory learns shopping activity, hobbies, pets, family composition over time

Key capabilities:
- "Help Me Decide" — conversational product comparison based on user-specific needs
- Contextual understanding of vague queries ("something for my nephew's birthday, he's into dinosaurs")
- Multi-turn refinement ("actually, make it educational too")

What makes Rufus architecturally interesting: it's not just a chatbot over a product catalog. It uses RL to optimize response quality, maintains long-term user memory, and routes different query types to different models based on complexity and domain.

Source: [amazon.science](https://amazon.science), [aws.amazon.com/blogs/machine-learning](https://aws.amazon.com/blogs/machine-learning)

---

**Instacart — Head-Tail Architecture**

Instacart's approach is pragmatic: use traditional methods where they work well, and LLMs only where they add unique value.

Architecture:
- **Head queries** (popular, well-understood): Traditional retrieval and ranking — fast, cheap, proven
- **Tail queries** (novel, ambiguous, long-tail): LLM-based understanding and matching
- **Pattern**: Hybrid of Pattern B (feature extraction) and Pattern D (conversational for complex queries)

Results:
- 96.4% SRL (Search Result Level) precision on tail queries
- The key insight: you don't need LLMs for "milk" — you need them for "something creamy for my coffee that's not dairy but not too sweet"

This head-tail split is becoming a common architectural pattern: let traditional systems handle the ~80% of queries that are routine, and invest LLM compute only in the ~20% where content understanding genuinely matters.

Source: [instacart.com/company/tech-innovation](https://instacart.com/company/tech-innovation)

---

### 4.4 Advertising

**Meta GEM — Generative Ads Model**

Meta's GEM (Generative Explore and Match) applies LLM-scale models to ad recommendation — one of the highest-stakes, most latency-sensitive recommendation domains.

Architecture:
- **Scale**: Trained on thousands of GPUs
- **Pattern**: Pattern E (User/Item Understanding) — the model generates rich ad and user understanding features that feed into the existing ad ranking pipeline
- **Key innovation**: Using generative models to understand ad creative content, user intent, and match quality at a level traditional feature engineering can't reach

Results:
- ~5% more conversions on Instagram
- ~3% more conversions on Facebook
- Extended to Threads
- At Meta's ad revenue scale, even small percentage improvements represent billions in value

What makes this notable: ads are where the latency constraint is hardest (real-time bidding operates in <10ms). Meta solved this by using the LLM offline (Pattern E) to generate features, not by putting the LLM in the online serving path.

Source: [engineering.fb.com](https://engineering.fb.com)

---

### 4.5 Job & Professional Matching

LinkedIn has published the most comprehensive set of LLM-for-recommendations work, with three distinct systems addressing different aspects of professional matching.

**LinkedIn 360Brew — Unified Foundation Model**

The most architecturally ambitious system: one model to rule all LinkedIn recommendations.

Architecture:
- **Model**: 150B-parameter decoder-only foundation model built on Mixtral 8x22 MoE (Mixture of Experts)
- **Scope**: Replaces thousands of specialized models with ONE unified model handling 30+ ranking tasks across all LinkedIn surfaces — feed, jobs, People You May Know (PYMK), ads, and more
- **Key innovation**: Processes raw interaction history as natural language — no feature engineering required

```
Traditional approach:
  User features (1000+ engineered) + Item features (500+ engineered) → Ranking Model

360Brew approach:
  "User viewed 3 data science jobs in NYC last week, connected with 2 ML engineers,
   shared an article about transformer architectures" → LLM → Rankings for ALL surfaces
```

Results:
- Zero-shot generalization to new LinkedIn surfaces (no retraining needed)
- Eliminated the need to maintain thousands of specialized models
- Single model handles feed ranking, job recommendations, connection suggestions, and ad targeting

What makes this radical: most companies add LLMs to *one* recommendation surface. LinkedIn is trying to collapse their entire recommendation stack into a single LLM that understands professional context across all surfaces.

Source: [arxiv.org/abs/2501.16450](https://arxiv.org/abs/2501.16450), [zenml.io](https://zenml.io)

---

**LinkedIn JUDE — Job Understanding via LLM Embeddings**

A more focused application: using LLMs specifically for job recommendation embeddings.

Architecture:
- **Model**: Mistral 7B, fine-tuned with LoRA
- **Pattern**: Pattern B (Feature Extractor) — generates job embeddings offline
- **Serving**: Sub-300ms p95 latency for embedding lookup

Results:
- 6x reduction in duplicate job recommendations
- +2.07% qualified applications
- -5.13% dismiss-to-apply ratio (users see fewer irrelevant jobs)
- In production

The key insight: LoRA fine-tuning on a 7B model was sufficient to dramatically improve job understanding. They didn't need a 70B model or complex RL — SFT with LoRA on domain data was enough.

Source: [linkedin.com/blog/engineering](https://linkedin.com/blog/engineering)

---

**LinkedIn Semantic Job Search**

Redesigning job search from keyword matching to semantic understanding.

Architecture:
```
User Query: "remote ML roles at startups paying $200k+"
     ↓
NLU Query Understanding (LLM-powered intent + entity extraction)
     ↓
GPU-Accelerated Exhaustive Nearest-Neighbor Search
     ↓
Top 250 candidates
     ↓
Distilled Cross-Encoder Reranker
     ↓
Final ranked results
```

Key engineering decisions:
- Exhaustive (not approximate) nearest-neighbor search — they found ANN quality loss was unacceptable for job search
- GPU acceleration made exhaustive search viable at scale
- Cross-encoder reranker distilled from a larger model for production serving speed
- Synthetic training data generated from fine-tuned models (a common pattern when labeled data is expensive)
- 75x throughput improvement from the architecture redesign

Source: [linkedin.com/blog/engineering](https://linkedin.com/blog/engineering)

---

### 4.6 Travel Recommendations

**Booking.com — AI Trip Planner**

Architecture:
- **Pattern**: Pattern D (Conversational Recommender)
- **Model**: Built with OpenAI
- **Interface**: Chat-based multi-turn trip planning

How it works:
- Users describe their trip preferences in natural language ("family beach vacation in Europe, kid-friendly, under $3000")
- The system asks clarifying questions, refines preferences, and generates personalized itineraries
- Recommendations pull from Booking.com's inventory with real-time availability and pricing
- Multi-turn dialogue allows progressive refinement ("actually, we'd prefer something more secluded")

Travel is a natural fit for conversational recommendation because:
1. Decisions are high-consideration (users want to deliberate)
2. Preferences are complex and multi-dimensional (location, budget, activities, dates, group composition)
3. The traditional filter-and-browse UX is painful for trip planning
4. Users expect longer response times (seconds, not milliseconds)

Source: [news.booking.com](https://news.booking.com), [openai.com/index/booking-com](https://openai.com/index/booking-com)

---

### 4.7 Food & Restaurant Recommendations

**DoorDash — Consumer Profiles (Cx)**

DoorDash's approach centers on building rich, LLM-generated consumer profiles that capture preferences traditional feature vectors can't express.

Architecture:
- **Pattern**: Pattern E (User/Item Understanding)
- **System**: Consumer Profiles (Cx) — 360-degree natural-language preference narratives
- **Input signals**: Order history, browse behavior, time-of-day patterns, cuisine preferences, dietary constraints

What a Cx profile looks like:
```
"Orders Thai and Indian frequently, especially on weekday evenings.
 Prefers spicy dishes (consistently rates spicy options 4-5 stars).
 Weekend orders skew toward pizza and burgers — likely ordering for family.
 Price-sensitive on weekday lunches (avg $12-15), less so on weekends ($25-35).
 Has ordered from 3 different pho restaurants — seems to be searching for a regular spot.
 Avoids seafood entirely (0 orders containing fish/shellfish in 6 months)."
```

Additional LLM applications at DoorDash:
- **Query rewriting**: Transforming vague searches ("something warm and comforting") into concrete restaurant/cuisine matches
- **Menu understanding**: Parsing unstructured menu descriptions into structured features

Source: [careersatdoordash.com/blog](https://careersatdoordash.com/blog)

---

### 4.8 Visual & Style Recommendations

**Pinterest — Visual Language Model Integration**

Pinterest's recommendation challenge is uniquely visual: users often can't articulate what they want in words, but know it when they see it.

Architecture:
- **CLIP embeddings**: Visual similarity powered by CLIP (Contrastive Language-Image Pre-training) — Pattern B
- **VLM-powered assistant**: Visual Language Model that understands both pin content and user queries
- **LLM relevance labeling**: Using LLMs to generate high-quality training labels for the recommendation model (replacing expensive human annotation)
- **"Styled for you"**: AI-generated collages that combine pins into personalized mood boards

Key applications:
- Visual search: "Find more like this" uses CLIP embeddings for visual similarity
- Style understanding: LLMs parse visual content into style attributes (bohemian, minimalist, industrial, etc.)
- Relevance labeling: LLM-generated training labels improved recommendation quality while reducing annotation costs

Source: [medium.com/pinterest-engineering](https://medium.com/pinterest-engineering)

---

### 4.9 News & Content Feed

**X/Twitter — Content Understanding**

LLMs are used for:
- Content understanding: classifying tweet topics, sentiment, and quality
- Feed ranking: LLM-generated features as inputs to the ranking model
- Safety: content moderation and policy compliance

**Reddit — Contextual Recommendations**

LLMs help with:
- Understanding subreddit context and topic relationships
- Improving content recommendations across subreddits
- Search result quality via query understanding

Both platforms primarily use Pattern B (LLM as Feature Extractor) and Pattern E (content understanding) rather than putting LLMs directly in the ranking path.

---

### 4.10 Conversational Recommendation

This section covers systems where the LLM *is* the recommendation interface — not a feature generator feeding a traditional pipeline, but the system users directly interact with.

**How LLM-Native Recommenders Personalize**

The core architectural pattern across all conversational recommenders:

```
User utterance ──▶ Intent understanding ──▶ Preference extraction
      ▲                                            │
      │                                  Memory / profile update
      │                                            │
      │                                            ▼
Refined response ◀── LLM generation ◀── Retrieval (RAG / catalog)
```

Each turn in the conversation updates the system's understanding of the user, which feeds back into retrieval and generation. This is fundamentally different from traditional RecSys, which learns from *actions* (clicks, purchases) — conversational systems learn from *stated preferences* in real time.

#### When a User Doesn't Like a Recommendation — What Happens to That Signal?

**Short answer: No.** In-session adaptation is handled entirely through prompt context and in-context learning — not gradient updates. Cross-session memory is stored as text summaries or user profiles injected into prompts. Model weights update on hourly-to-weekly cycles, never per-request.

The production architecture is a three-layer system:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  THE FEEDBACK LOOP REALITY                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Layer 1: REAL-TIME FEATURES          Latency: milliseconds         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Feature store serves fresh values (user just clicked X)     │   │
│  │ Model weights frozen — only inputs change                   │   │
│  │ Everyone does this: Meta, Spotify, Netflix, DoorDash        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Layer 2: NEAR-LINE EMBEDDINGS        Latency: minutes to hours     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ User embeddings recomputed from recent behavior             │   │
│  │ Spotify: within minutes via NRT triggers                    │   │
│  │ Instagram Explore: hourly model fine-tune                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Layer 3: BATCH MODEL RETRAIN         Latency: daily to weekly      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Full model weight updates on accumulated data               │   │
│  │ Validation, canary testing, gradual rollout                 │   │
│  │ Everyone does this on different cadences                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Layer 0: BANDIT EXPLORATION          Latency: per-request          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Thompson Sampling / UCB / epsilon-greedy on top of ranking  │   │
│  │ Netflix artwork (125M users), Spotify homepage, DoorDash    │   │
│  │ Handles uncertainty without retraining                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Company-specific update cadences:**

| Company | What Updates | How Fast | Source |
|---------|-------------|----------|--------|
| **Meta / Instagram Explore** | Model fine-tune | **Every hour** via continual online training | [engineering.fb.com — Scaling Instagram Explore](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/) |
| **Meta Reels** (Jan 2026) | User satisfaction alignment | Daily via UTIS survey model (+5.4% high ratings, +5.2% engagement) | [engineering.fb.com/2026/01/14](https://engineering.fb.com/2026/01/14/ml-applications/adapting-the-facebook-reels-recsys-ai-model-based-on-user-feedback/) |
| **Spotify** | User embeddings | **Within minutes** via NRT triggers (3 temporal windows: 6mo/1mo/1wk) | [engineering.atspotify.com/2026/1](https://engineering.atspotify.com/2026/1/why-we-use-separate-tech-stacks-for-personalization-and-experimentation) |
| **Netflix** | Proxy rewards | Days-weeks (explicitly acknowledges delayed feedback problem) | [netflixtechblog.com — Behind the Streams](https://netflixtechblog.com/behind-the-streams-real-time-recommendations-for-live-events-e027cb313f8f) |
| **Netflix Live Events** | Real-time broadcast | Milliseconds (Kafka + WebSocket, 38M events/sec) | Same source |
| **Tubi** | Inference-time feature override | Intra-day (model frozen, inputs freshened — +0.47% engagement) | [arxiv.org/abs/2512.14734](https://arxiv.org/html/2512.14734) |
| **YouTube** | Model retrain | Hours to days ("Example Age" feature to avoid staleness bias) | Google Research |

**In-session "learning" is prompt context, not model updates.** A 2025 survey on LLM-based recommender systems ([arxiv.org/abs/2507.21117](https://arxiv.org/html/2507.21117v2)) confirms the mechanism: LLMs perform "zero-shot adaptation through prompt composition rather than gradient-based embedding updates." When a user says "no, something cheaper," the system appends feedback to the conversation context and re-generates. Three constraints prevent real-time weight updates: (1) latency — recs must resolve in ~200ms, a gradient step takes orders of magnitude longer; (2) catastrophic forgetting — updating on one user degrades others; (3) infrastructure complexity.

**Cross-session memory systems:**

| System | Memory Type | What's Stored | Persistence |
|--------|------------|---------------|-------------|
| **Amazon Rufus** | Account memory | Shopping activity, hobbies, pets, family, preferences ("plastic-free packaging") | Weeks to months |
| **ChatGPT Shopping** | Conversation memory | Natural-language summaries of past conversations and stated preferences | Persistent (if Memory enabled) |
| **Perplexity** | Session memory | Past searches, aesthetic/style/functional preferences | Cross-session |
| **Tolan AI** | Vector memory | Facts, preferences, emotional signals → embedded with text-embedding-3-large → Turbopuffer | Persistent |

None of them update LLM weights per-user. It's all prompt injection of stored preferences.

**Bandits for exploration:** Contextual bandits are widely deployed *on top of* recommendation models to handle the explore/exploit tradeoff without retraining:
- **Spotify**: BaRT system (counterfactual risk minimization), homepage calibration via neural contextual bandits with epsilon-greedy (2025), centralized exploration for creator audience building
- **Netflix**: Contextual bandits for artwork personalization tested on 125M users — "the regret incurred by exploration is typically very small and is amortized across our large member base"
- **DoorDash**: UCB bandits for homepage exploitation/exploration, Thompson Sampling for cuisine preferences

Sources: [eugeneyan.com/writing/bandits](https://eugeneyan.com/writing/bandits/), [research.atspotify.com](https://research.atspotify.com)

---

**ChatGPT Shopping**

OpenAI's entry into recommendation — shopping integrated directly into ChatGPT.

Architecture:
- **Model**: GPT fine-tuned with RL specifically for shopping tasks
- **Personalization signals**: ChatGPT memory (past conversations, stated preferences), collaborative filtering signals, multi-step web search across review sites
- **Output**: Buyer's guides personalized to user context

How personalization works:
- The system remembers preferences from past conversations ("you mentioned you prefer minimal design" or "you have a small kitchen")
- Collaborative filtering signals from aggregate user behavior
- Real-time web search across review sites for up-to-date information
- Multi-step reasoning: searches, compares, synthesizes across sources
- Signals: availability, price, quality, maker status (favoring independent makers when relevant)

What makes this architecturally novel: it combines long-term memory (conversation history), collaborative signals (aggregate behavior), and real-time retrieval (web search) in a single conversational flow. As of 2026, Instant Checkout (Stripe-powered Agentic Commerce Protocol) enables in-chat purchasing. ChatGPT Pulse (Pro users) can proactively suggest personalized buyer's guides based on past conversations.

Source: [openai.com/index/chatgpt-shopping-research](https://openai.com/index/chatgpt-shopping-research), [openai.com/index/buy-it-in-chatgpt](https://openai.com/index/buy-it-in-chatgpt)

---

**Amazon Rufus** *(detailed in Section 4.3)*

Rufus is covered in the e-commerce section above. To summarize its conversational architecture: custom Nova models, RAG over full product catalog + 51M reviews, RL-optimized response quality, long-term account memory, and multi-model routing by query type. As of 2025-2026, account memory persists preferences for weeks/months and learns hobbies, pets, and family composition. Users engaging with Rufus show **60% higher purchase likelihood** (250M+ users).

---

**Perplexity Shopping**

Architecture:
- **Pattern**: Conversational search + recommendation
- **Personalization**: Remembers past searches, learns user patterns over time
- **Key differentiator**: Transparent and unbiased — decided against advertising (Feb 2026) to preserve trust
- **Memory**: Learns aesthetic preferences, style preferences, and functional needs across sessions
- **Visual search**: "Snap to Shop" — upload a photo to find matching/similar items
- **Virtual try-on**: Create a virtual avatar to preview apparel

How it differs from traditional e-commerce search: Perplexity analyzes contextual data for personalized results without being influenced by advertising. The conversational interface allows multi-turn refinement, and the system remembers preferences across sessions. "Buy with Pro" enables one-click purchase with pre-filled payment and free shipping.

Source: [perplexity.ai/hub/blog](https://perplexity.ai/hub/blog)

---

**AI Companions and Matching**

LLMs are also being applied to *people* recommendations — not products.

**Tolan AI — Voice-First AI Companion Matching**

Architecture:
- **Matching**: Personality interview → preference extraction → companion matching
- **Memory system**: Retains facts, preferences, and emotional "vibe" signals
- **Embeddings**: text-embedding-3-large, stored in Turbopuffer vector DB
- **Context window**: Rebuilt each turn from message summaries, persona card, vector-retrieved memories, tone guidance, and real-time signals
- **Model**: GPT for conversational flow
- **Latency**: Sub-50ms vector lookup for memory retrieval

```
New conversation turn:
  ┌─ Message summaries (recent conversation)
  ├─ Persona card (companion personality)
  ├─ Vector-retrieved memories (relevant past facts)
  ├─ Tone guidance (emotional calibration)
  └─ Real-time signals (time of day, conversation momentum)
       ↓
  Context window assembly → LLM generation → Response
       ↓
  Memory update: new facts/preferences → embed → store
```

Source: [openai.com/index/tolan](https://openai.com/index/tolan)

---

**Hinge — Algorithmic Dating Recommendations**

Architecture:
- **Core algorithm**: Gale-Shapley (stable marriage problem) adapted for dating
- **ML layer**: Models trained on revealed preference data (who users actually like/message vs skip)
- **"Most Compatible"**: Iteratively learns from user behavior to improve match quality
- **Result**: 75% of users who went on dates met through algorithm recommendations

Hinge represents the traditional ML approach to people-matching — not LLM-powered yet, but a strong baseline that illustrates the challenge of preference elicitation that LLM systems are trying to improve upon.

Source: [d3.harvard.edu research](https://d3.harvard.edu)

---

### 4.11 LLM-Assisted Personalization (Emerging)

These companies use LLMs to *enhance* their traditional recommendation systems — not replace them. The LLM improves specific components (content understanding, feature generation, personalization logic) while the core recommendation pipeline remains traditional.

**DoorDash GenAI Homepage**

Architecture:
- **Framework**: LLM-assisted personalization blending affordability, familiarity, and novelty
- **Retrieval**: Hierarchical RAG for fast inference
- **Knowledge**: Product knowledge graph built with LLMs
- **Pattern**: Hybrid — LLM-generated knowledge graph feeds traditional ranking

DoorDash has published at least 5 blog posts detailing their LLM-assisted personalization work, covering consumer profiles, query rewriting, grocery recommendations, and the overall personalization framework. Their approach is notably iterative — adding LLM capabilities incrementally rather than rebuilding from scratch.

Source: [careersatdoordash.com/blog](https://careersatdoordash.com/blog)

---

**Uber Eats — Embedding-Based Personalization**

Architecture:
- **Embeddings**: Two-Tower model for ~100ms personalized retrieval
- **Graph learning**: Bipartite graphs modeling restaurant-dish and user-restaurant relationships
- **Multi-objective optimization**: Balancing relevance, delivery time, restaurant quality, and user preferences

Uber has published 5+ blog posts on their recommendation architecture, covering:
- Two-tower embeddings for restaurant and dish retrieval
- Graph-based learning for relationship modeling
- Food discovery through content understanding
- Multi-objective ranking that balances relevance with operational constraints (delivery time, restaurant capacity)

Source: [uber.com/blog](https://uber.com/blog)

---

**Airbnb — Listing Embeddings and EBR**

Architecture:
- **Listing embeddings**: Dense vector representations capturing listing characteristics, photos, descriptions, and booking patterns
- **EBR (Embedding-Based Retrieval)**: Real-time personalized retrieval using learned embeddings
- **Experimentation**: 50x sensitivity improvement via interleaving experiments (comparing two ranking algorithms side-by-side within a single search results page)

Key contributions:
- Similar-listing recommendations powered by embedding similarity
- Real-time personalization based on in-session behavior (searches, clicks, saves)
- Interleaving experiments allowed much faster iteration on recommendation quality

Source: [medium.com/airbnb-engineering](https://medium.com/airbnb-engineering)

---

## Mega-Scale Custom Architectures

Some companies have built recommendation architectures that bridge the gap between traditional deep RecSys and full LLMs. These aren't LLM-powered recommendations in the usual sense — they're custom transformer architectures designed specifically for recommendation at massive scale.

### Meta HSTU — Hierarchical Sequential Transduction Units

The most important development in bridging traditional RecSys and LLMs.

Architecture:
- **Scale**: Trillion-parameter models
- **Design**: Sequential transducers that process user interaction histories as sequences (similar to how LLMs process text)
- **Key innovation**: Applies the "scaling laws" insight from LLMs to recommendation — bigger models with more data reliably improve
- **Inference**: M-FALCON system provides 900x inference speedup

Results:
- 12.4% improvement over DLRM (Meta's previous state-of-the-art)
- Demonstrates that transformer-based sequential models can beat traditional deep learning approaches at Meta scale

Why this matters: HSTU suggests the future of RecSys might not be "bolt an LLM onto your pipeline" but rather "build a transformer architecture native to recommendation data." The model processes interaction sequences the way an LLM processes text, but with architecture choices specific to recommendation (heterogeneous actions, timestamps, multi-modal features).

```
Traditional DLRM:
  Sparse features → Embedding lookup → MLP → Prediction

HSTU:
  User action sequence → Transformer layers → Sequential prediction
  [view_item_1, click_item_2, purchase_item_3, ...] → next action prediction
```

Source: Meta AI research publications

---

### Google Semantic IDs (RQ-VQE)

The foundation technology behind YouTube LRM and other generative recommendation systems.

How it works:
```
Step 1: Encode items into dense embeddings (content encoder)
Step 2: Quantize embeddings into discrete codes via Residual Quantization
         [0.23, -0.15, 0.87, ...] → [42, 187, 3, 91]
Step 3: Use these codes as "words" in the LLM's vocabulary
Step 4: LLM generates recommendation by producing code sequences

Key property: similar items → similar code sequences
  "Italian romantic comedy" → [42, 187, 3, 91]
  "French romantic comedy" → [42, 187, 3, 88]   ← differs only in last code
  "Horror documentary"     → [91, 23, 156, 7]   ← completely different
```

This enables LLMs to "generate" recommendations without needing to know every item ID — they learn patterns in the semantic space that generalize to new items.

Source: Google Research

---

### Alibaba SAID — Cold-Start Embeddings

Architecture:
- Uses LLM item understanding to generate embeddings for new items with no interaction history
- The LLM processes item metadata (title, description, attributes) to produce embeddings that are compatible with the existing embedding space
- New items immediately get reasonable recommendations without the traditional cold-start ramp-up period

Source: Alibaba research publications

---

### Kuaishou OneRec — End-to-End Generative Recommendation (2025-2026)

The first end-to-end generative model to significantly outperform cascaded retrieve-then-rank pipelines in production.

Architecture:
- **Design**: Encoder-decoder with Mixture of Experts (MoE) for parameter scaling
- **Key innovation**: Session-wise list generation — generates an entire recommendation slate at once, not point-by-point next-item prediction
- **Alignment**: Iterative preference alignment via DPO (Direct Preference Optimization) and Early-clipped GRPO
- **Reward**: Learned P-Score blending clicks, watch time, and other signals per user

Results:
- On 25% of Kuaishou traffic, the pure generator matches the full multi-stage stack
- Adding reward-model selection: **+0.54% app stay time** on main app, **+1.24% on Kuaishou Lite**
- +1.68% total watch time, +6.56% average view duration
- Training MFU of 23.7%, inference MFU of 28.8% — 5x improvement over the old ranker

Open-sourced as **OpenOneRec** (2026): 1.7B and 8B parameter foundation models built on Qwen3 backbone, with both standard (open data) and Pro (hundred-billion-token industrial corpus) versions.

Source: [arxiv.org/abs/2502.18965](https://arxiv.org/abs/2502.18965), [github.com/Kuaishou-OneRec/OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec)

---

### Netflix Foundation Model for Recommendations (2025-2026)

Netflix built a transformer-style sequential foundation model trained with next-token and multi-token prediction at large scale for personalization.

Architecture:
- **Training**: Next-token and multi-token prediction on user interaction sequences
- **Efficiency**: Sparse attention, sliding-window sampling, KV caching for low-latency inference
- **Integration**: Three production approaches — embeddings (Pattern B), subgraph extraction, and fine-tuning for specific tasks
- **Infrastructure**: Complex data pipelines, distributed state across multi-node GPU clusters

This represents Netflix moving from the task-specific Llama fine-tuning (artwork personalization) toward a general-purpose recommendation foundation model.

Source: [netflixtechblog.com/foundation-model-for-personalized-recommendation](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39)

---

### Pinterest PinRec — Production Generative Retrieval (2025-2026)

Pinterest's PinRec demonstrates that generative retrieval is production-ready for visual recommendations, with solutions for latency, cost, and control challenges in recommendation serving.

Source: [shaped.ai/blog/pinrec-teardown](https://www.shaped.ai/blog/pinrec-teardown-inside-pinterests-production-ready-generative-retrieval-model)

---

## LLM vs Traditional — The Honest Comparison

The question everyone asks: "Should we use an LLM for recommendations?" The answer depends entirely on the scenario.

### Evidence Table

| Scenario | Winner | Why | Evidence |
|----------|--------|-----|----------|
| Warm users, rich interaction data | **Traditional CF/DL** | Collaborative signals from millions of interactions are hard to beat with text understanding | Multiple benchmarks; "Can LLMs Outshine Conventional Recommenders?" finds mixed results |
| Cold-start (new users or items) | **LLM** | Content understanding fills the gap when there's no interaction history | YouTube LRM, Alibaba SAID both show strong cold-start performance |
| Explainability | **LLM** | Can generate natural-language explanations for recommendations | Spotify: 4x CTR on explained recommendations |
| Latency-critical (<10ms) | **Traditional** | LLMs are 10-100x slower, even distilled | Ads, real-time bidding — no LLM can serve in <10ms without being reduced to a feature generator |
| Conversational discovery | **LLM** | No traditional equivalent exists | Amazon Rufus, Booking.com, ChatGPT Shopping — entirely new UX category |
| Cross-domain understanding | **LLM** | Pre-training encodes world knowledge that crosses domain boundaries | LinkedIn 360Brew: single model across feed, jobs, PYMK, ads |
| Tail queries (long-tail, ambiguous) | **LLM** | Content understanding handles what sparse interaction data can't | Instacart: 96.4% precision on tail queries |
| Head queries (popular, routine) | **Traditional** | Fast, cheap, and already optimized | Instacart's head-tail architecture explicitly uses traditional for head |

### The Mixed Results Paper

"Can LLMs Outshine Conventional Recommenders?" ([arxiv.org/abs/2503.05493](https://arxiv.org/abs/2503.05493)) is the most honest academic assessment. Key findings:

- LLMs do **not** consistently outperform well-tuned traditional models on standard benchmarks
- LLMs excel in specific scenarios: cold-start, cross-domain transfer, explainability
- The gap narrows significantly when traditional models have access to content features (not just interaction data)
- LLM fine-tuning on recommendation data helps, but the cost-benefit ratio is often unfavorable compared to improving traditional models

### The Hybrid Consensus

The industry has largely converged on a hybrid architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    THE HYBRID ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Collaborative Filtering        LLM Understanding          │
│   ┌────────────────────┐        ┌────────────────────┐     │
│   │ Behavioral signals │        │ Semantic signals    │     │
│   │ • Click patterns   │        │ • Content meaning   │     │
│   │ • Purchase history │        │ • User intent       │     │
│   │ • Co-occurrence    │        │ • Item attributes   │     │
│   │ • Implicit feedback│        │ • Cross-domain      │     │
│   └────────┬───────────┘        └────────┬───────────┘     │
│            │                             │                   │
│            └──────────┬──────────────────┘                   │
│                       ▼                                      │
│              ┌────────────────┐                              │
│              │  Fusion Layer  │                              │
│              │  (ranking      │                              │
│              │   model)       │                              │
│              └────────┬───────┘                              │
│                       ▼                                      │
│              Final Recommendations                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

CF for behavioral signals + LLM for semantic understanding. Not "LLM replaces RecSys" — "LLM extends RecSys."

### Decision Tree for Practitioners

```
Do you have rich interaction data for this user/item?
├── YES: Is the primary goal explainability or conversational?
│   ├── YES → Add LLM (Pattern A reranker or Pattern D conversational)
│   └── NO → Traditional CF/DL is likely sufficient
│       └── Consider LLM for: diversity, cross-domain, content understanding
└── NO (cold-start):
    ├── Do you have item content/metadata?
    │   ├── YES → LLM embeddings (Pattern B) or generative (Pattern C)
    │   └── NO → Popularity-based fallback (LLM can't help much either)
    └── Can you tolerate >100ms latency?
        ├── YES → LLM reranker (Pattern A) or conversational (Pattern D)
        └── NO → LLM features offline (Pattern B/E), traditional serving
```

### Cost Reality

The elephant in the room: LLM inference is expensive.

- YouTube needed **95%+ cost reduction** from baseline LLM serving to make their Large Recommendation Model production-viable
- Netflix distills from a 32B teacher to an 8B student for production serving
- Meta runs GEM offline (Pattern E) precisely to avoid online LLM inference costs
- LinkedIn JUDE precomputes embeddings offline rather than running the LLM at query time

The **distillation pattern** is essential: train or prompt a large LLM to generate high-quality features/labels, then distill into a smaller model (or traditional features) for production serving. Nearly every successful deployment follows this pattern in some form.

---

## Research Frontier

### Key Papers

| Paper | Year | Contribution |
|-------|------|-------------|
| **P5** | 2022 | First unified text-to-text framework for recommendation (rating, retrieval, explanation, all as text generation) |
| **TALLRec** | 2023 | Demonstrated LoRA fine-tuning of LLaMA for recommendation with minimal data |
| **LlamaRec** | 2023 | Sequential recommendation using LLM with retriever-reranker pipeline |
| **HSTU** (Meta) | 2024 | Trillion-parameter sequential transducers, 12.4% over DLRM |
| **Data-Efficient FT** | 2024 | 2% of fine-tuning samples achieves full performance (SIGIR '24) |
| **Netflix Artwork** | 2025 | SFT + distillation + DPO pipeline for personalized artwork |
| **Process-Supervised LLM** | 2025 | Process supervision for LLM recommenders (SIGIR '25) |
| **Rec-R1** | 2025 | RL framework bridging LLMs with RecSys via real-time reward signals from black-box rec models — no GPT-4 synthetic data needed |
| **Agentic Feedback Loop (AFL)** | 2025 | Two LLM agents (recommender + user) iterate in feedback loop; +11.52% over single agent; does NOT amplify popularity bias |
| **OneRec** (Kuaishou) | 2025 | First end-to-end generative model outperforming cascaded pipelines in production |
| **HELM** | 2026 | Human-centered evaluation framework for LLM recommenders across 5 dimensions (WWW '26) |
| **AlignUSER** | 2026 | World-model-driven LLM agents for user simulation in rec evaluation |

### RL for Recommendations — Production Deployments

Reinforcement learning in recommendation is mostly **batch/offline RL**, not fully online. The constraint is safety: a bad online policy update degrades the experience for millions of users.

| Company | Method | Application |
|---------|--------|-------------|
| **Google/YouTube** | REINFORCE (policy gradient) | Video recommendations |
| **Kuaishou** | Early-clipped GRPO + DPO | OneRec generative recommendation |
| **Netflix** | GRPO with verifiable rewards | Artwork personalization |
| **ByteDance** | Cascading DQNs | Joint rec + ad placement |
| **Alibaba** | Actor-Critic (first deployed 2016 Double Eleven) | Search ranking; Virtual Taobao simulation environment |
| **Microsoft** | DQN | News recs (click reward + long-term user activeness at 0.05 weight) |
| **Amazon** | LAAC (LLM-guided Adversarial Actor Critic) | LLM as reference policy for novel item suggestion + lightweight RL to refine |

**Rec-R1** ([arxiv.org/abs/2503.24289](https://arxiv.org/html/2503.24289v4)) is the most significant recent development: a general RL framework that optimizes LLM generation using real-time reward signals from a fixed, black-box recommendation model — without relying on synthetic data from proprietary models. This represents the convergence of RLHF techniques (from LLM alignment) with recommendation optimization.

Source: [eugeneyan.com/writing/reinforcement-learning-for-recsys-and-search](https://eugeneyan.com/writing/reinforcement-learning-for-recsys-and-search/)

### Frameworks and Tools

| Framework | What It Does | Link |
|-----------|-------------|------|
| **RecBole** | Unified recommendation library with 90+ models | [recbole.io](https://recbole.io) |
| **Microsoft RecAI** | LLM-powered recommendation agent toolkit | [github.com/microsoft/RecAI](https://github.com/microsoft/RecAI) |
| **Microsoft InteRecAgent** | Interactive recommendation agent built on LLMs | Part of RecAI |
| **OpenOneRec** (Kuaishou) | Open-source generative rec foundation models (1.7B, 8B) on Qwen3 | [github.com/Kuaishou-OneRec/OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) |
| **Eugene Yan's Guide** | Comprehensive practitioner's guide to LLMs for RecSys | [eugeneyan.com/writing/recsys-llm](https://eugeneyan.com/writing/recsys-llm) |

### Active Research Labs

- **Google DeepMind**: Semantic IDs, Large Recommendation Models, Gemini for recs
- **Meta AI**: HSTU, M-FALCON inference, GEM for ads, Reels UTIS feedback alignment
- **Microsoft Research**: RecAI, InteRecAgent, conversational recommendation
- **Alibaba DAMO Academy**: SAID cold-start, industrial-scale LLM recs
- **Kuaishou**: OneRec generative recommendation, open-source foundation models

### 2026 Trends

**1. Agentic Recommendation is the dominant new paradigm.** Systems that proactively plan, remember, and act — not just passively respond to queries. Criteo launched an Agentic Commerce Recommendation Service via Model Context Protocol (MCP), leveraging 720M daily shoppers and $1T in transaction data, showing 60% improvement in relevancy. OpenAI + Stripe open-sourced the Agentic Commerce Protocol. WWW 2026 has a dedicated workshop on "LLM & Agents for Recommendation Systems." Source: [criteo.com](https://www.criteo.com/news/press-releases/2026/02/criteo-introduces-agentic-commerce-recommendation-service-to-power-ai-shopping-assistants/), [llmandagents4recsys.github.io](https://llmandagents4recsys.github.io/)

**2. Generative recommendation models going mainstream.** Kuaishou's OneRec, Netflix's Foundation Model, and Pinterest's PinRec all demonstrate that end-to-end generative models (encoder-decoder, next-token prediction) are outperforming traditional cascaded retrieve-then-rank pipelines in production. Kuaishou open-sourced 1.7B and 8B parameter models.

**3. In-chat purchasing (zero-click commerce).** ChatGPT Instant Checkout (Stripe), Google Gemini checkout (Etsy, Wayfair, Shopify, Target, Walmart), and Perplexity "Buy with Pro" all enable purchasing without leaving the conversation. About one-third of U.S. consumers would let AI make purchases for them. Source: [gorgias.com/state-of-conversational-commerce-2026](https://www.gorgias.com/state-of-conversational-commerce-2026)

**4. Dual-pipeline architectures.** Spotify's Jan 2026 engineering blog confirms the production pattern: separate tech stacks for stable personalization (low-latency, high-availability) and experimentation (bold LLM experiments without risking production). Source: [engineering.atspotify.com/2026/1](https://engineering.atspotify.com/2026/1/why-we-use-separate-tech-stacks-for-personalization-and-experimentation)

**5. Evaluation frameworks catching up.** HELM (WWW '26) evaluates LLM recommenders across intent alignment, explanation quality, interaction naturalness, trust/transparency, and fairness/diversity. GPT-4 scores 4.35/5.0 on interaction naturalness but shows significant popularity bias (Gini 0.73). Source: [arxiv.org/abs/2601.19197](https://arxiv.org/abs/2601.19197)

### Upcoming Conferences

- **WWW 2026** (Sydney, Apr-May): LLM & Agents for RecSys workshop
- **RecSys 2026** (Minneapolis, Sep 28 - Oct 2): 20th ACM Conference on Recommender Systems
- **SIGIR 2026**: Submissions open on OpenReview
- **ICLR 2026** (Singapore, Apr): MemAgents workshop on memory for LLM-based agents

### Open Questions

1. **Billion-item generative retrieval**: Can LLMs generate recommendations over billion-item catalogs efficiently? Current semantic ID approaches work for millions but haven't proven at the largest scales.

2. **Semantic ID standardization**: Every company invents their own semantic ID scheme. No shared vocabulary or transfer learning across domains. Will a standard emerge?

3. **Real-time LLM serving for recs**: Can we get LLM inference fast enough for latency-critical paths (ads, real-time ranking) without reducing the model to a feature generator? Or is offline feature extraction the permanent architecture?

4. **User simulation**: Can LLMs simulate user behavior well enough to generate synthetic training data for traditional recommenders? Early results (LinkedIn, Instacart) suggest yes. AlignUSER (2026) uses world-model-driven LLM agents for this.

5. **Unification**: Is the future one model for all recommendation tasks (like LinkedIn 360Brew, Kuaishou OneRec) or specialized models per surface? The field is split, but evidence is tilting toward unified generative models.

6. **Agentic commerce protocols**: Will MCP / Agentic Commerce Protocol become the standard integration layer between AI shopping assistants and merchant inventory? Early signals point yes.

---

## Best Practices

### Three-Stage Integration Strategy

If you're adding LLMs to an existing recommendation system, don't start with Pattern C (Generator). Follow this progression:

**Stage 1: Offline Features (Low Risk, Immediate Value)**
- Use LLMs to generate embeddings or text features for items/users (Pattern B/E)
- Run offline, batch-processed, stored in feature store
- Zero latency impact on serving
- Easiest to A/B test against existing features
- *Start here*

**Stage 2: Reranker (Medium Risk, Higher Value)**
- Add LLM reranker on top of existing pipeline (Pattern A)
- Only processes top-K candidates, so cost is bounded
- Can be toggled off instantly if quality degrades
- Gives you the highest quality lift per LLM inference dollar
- *Move here once Stage 1 is validated*

**Stage 3: Primary Recommender (High Investment, Highest Potential)**
- LLM as primary generator (Pattern C) or conversational interface (Pattern D)
- Requires significant infrastructure investment (semantic IDs, low-latency serving)
- Offers capabilities impossible with traditional systems (generative retrieval, conversational discovery)
- *Only move here with strong evidence from Stage 1-2*

### Quick Reference Card

| "I want to..." | Pattern | Case Study |
|----------------|---------|------------|
| Improve cold-start recommendations | B (Feature Extractor) or C (Generator) | YouTube LRM, Alibaba SAID |
| Add explainability to existing recs | A (Reranker) with explanation generation | Spotify narrative explanations |
| Build a conversational shopping experience | D (Conversational) | Amazon Rufus, ChatGPT Shopping |
| Improve content understanding | E (User/Item Understanding) | DoorDash Cx, Meta GEM |
| Replace my entire RecSys stack | Don't. Use hybrid. | LinkedIn 360Brew is the exception, not the rule |
| Reduce annotation costs | B (Feature Extractor) for auto-labeling | Pinterest LLM relevance labeling |
| Handle long-tail queries | Hybrid (traditional for head, LLM for tail) | Instacart head-tail architecture |
| Unify multiple recommendation surfaces | Foundation model approach | LinkedIn 360Brew (ambitious, high risk) |
| Personalize visual content | B (Feature Extractor) with VLMs | Pinterest CLIP + VLM, Netflix artwork |

### Common Mistakes

1. **Using LLMs for real-time retrieval over millions of items**: LLMs are too slow. Use them for feature extraction (offline) or reranking (small candidate set), not for searching through your entire catalog in real-time.

2. **Ignoring the cold/warm user distinction**: LLMs shine for cold-start. If you have rich interaction data, a well-tuned traditional model is probably better. Know which users benefit from LLM features and which don't.

3. **Not distilling**: Running a 70B model for every recommendation request is economically insane. Distill into a smaller model or precompute features. Every successful production deployment does this.

4. **Treating recommendations like text generation**: Recommendation has hard constraints (inventory, availability, business rules, fairness) that pure text generation ignores. Always wrap LLM outputs in a validation layer.

5. **Skipping the hybrid architecture**: The answer is almost never "replace everything with an LLM" or "LLMs are useless for recs." It's "use LLMs where they add unique value and traditional methods where they're sufficient."

6. **Overfitting on academic benchmarks**: Papers showing LLMs beating traditional methods on MovieLens or Amazon Review datasets don't translate directly to production. Production has latency constraints, cost constraints, and data distributions that benchmarks don't capture.

---

## Summary Table

| Question | Answer | Evidence |
|----------|--------|----------|
| Does in-session negative feedback update the model? | No — it's prompt context / in-context learning only | Survey [arxiv:2507.21117] confirms "zero-shot adaptation through prompt composition, not gradient updates" |
| How fast do negative signals reach the model? | Features: ms. Embeddings: minutes. Weights: hourly-to-weekly. | Instagram Explore: hourly fine-tune. Spotify embeddings: minutes. Netflix: days-weeks. |
| What negative signals are captured? | Dismissals, skips, thumbs-down, "not interested" clicks, short dwell time, session abandonment | Meta uses composite reward (clicks + watch time + survey responses). Netflix uses proxy rewards for delayed feedback. |
| Is there cross-session memory of dislikes? | Yes — via stored text summaries or user profiles, injected into prompts | Rufus account memory (weeks/months), ChatGPT Memory, Perplexity session memory, Tolan vector memory |
| Are bandits used to reduce bad recs? | Yes, widely deployed — bandits naturally reduce exposure to disliked content as uncertainty drops | Netflix artwork (125M users), Spotify homepage, DoorDash UCB |
| Does anyone do real-time model weight updates from feedback? | No production system does per-request weight updates | Latency (~200ms budget), catastrophic forgetting, infrastructure complexity prevent it |
| What's the dominant 2026 trend? | Agentic recommendation — proactive, planning, memory-equipped agents | Criteo MCP launch, OpenAI+Stripe Agentic Commerce Protocol, WWW 2026 workshop |
| Are generative rec models production-ready? | Yes, outperforming cascaded pipelines | Kuaishou OneRec (+1.68% watch time), Netflix Foundation Model, Pinterest PinRec |
| LLM or traditional for warm users? | Traditional CF/DL wins | Multiple benchmarks; [arxiv:2503.05493] finds mixed results for LLMs |
| LLM or traditional for cold-start? | LLM wins | YouTube LRM, Alibaba SAID, Instacart tail queries |

---

## Conclusion

LLMs extend recommendation systems. They don't replace them.

The pattern across every successful production deployment is consistent: LLMs add value where content understanding, explainability, cold-start handling, or conversational interaction matters. Traditional collaborative filtering and deep learning models remain superior for warm users with rich interaction data, latency-critical paths, and routine queries.

The hybrid architecture — collaborative filtering for behavioral signals, LLMs for semantic understanding — isn't a compromise. It's the right design. Each component does what it's best at.

Three predictions based on the evidence:
1. **Offline LLM features become table stakes**: Within two years, every major recommendation system will use LLM-generated embeddings or features as inputs. The pattern is too effective and low-risk not to adopt.
2. **Conversational recommendation grows, but doesn't dominate**: Chat-based discovery (Rufus, Booking.com, ChatGPT Shopping) will carve out a meaningful niche for high-consideration decisions, but won't replace browse-and-click for routine use cases.
3. **Semantic IDs and generative retrieval mature**: The YouTube/Spotify pattern of LLMs generating item IDs will expand, especially as distillation techniques reduce serving costs.

For the fine-tuning techniques behind these systems — SFT, DPO, GRPO, distillation, LoRA — see [Fine-Tuning in Production](fine-tuning-in-production.md).

---

## References

### Company Engineering Blogs

| Company | System | Source |
|---------|--------|--------|
| Netflix | Artwork personalization (Llama 3.1 8B, SFT+DPO) | [arxiv.org/abs/2601.02764](https://arxiv.org/abs/2601.02764) |
| Netflix | Foundation Model for recommendations | [netflixtechblog.com](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39) |
| Spotify | Text2Tracks, AI DJ, narrative recs | [research.atspotify.com](https://research.atspotify.com) |
| Spotify | Dual personalization/experimentation stacks (Jan 2026) | [engineering.atspotify.com/2026/1](https://engineering.atspotify.com/2026/1/why-we-use-separate-tech-stacks-for-personalization-and-experimentation) |
| Meta | GEM (Generative Ads Model) | [engineering.fb.com](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/) |
| Meta | Reels UTIS feedback alignment (Jan 2026) | [engineering.fb.com/2026/01/14](https://engineering.fb.com/2026/01/14/ml-applications/adapting-the-facebook-reels-recsys-ai-model-based-on-user-feedback/) |
| Meta | Instagram Explore (hourly model updates) | [engineering.fb.com](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/) |
| YouTube/Google | Large Recommendation Model, Semantic IDs | Google Research |
| Google | Gemini shopping + Walmart partnership (Jan 2026) | [9to5google.com](https://9to5google.com/2026/01/12/google-and-walmart-partner-to-bring-ai-shopping-recommendations-to-gemini/) |
| LinkedIn | 360Brew (unified 150B model) | [arxiv.org/abs/2501.16450](https://arxiv.org/abs/2501.16450) |
| LinkedIn | JUDE (job embeddings) | [linkedin.com/blog/engineering](https://linkedin.com/blog/engineering) |
| LinkedIn | Semantic Job Search | [linkedin.com/blog/engineering](https://linkedin.com/blog/engineering) |
| Pinterest | VLM assistant, CLIP, LLM labeling, PinRec | [medium.com/pinterest-engineering](https://medium.com/pinterest-engineering) |
| Amazon | Rufus (conversational product recs, hydration architecture) | [amazon.science](https://amazon.science) |
| Amazon | Rufus scaling | [aws.amazon.com/blogs/machine-learning](https://aws.amazon.com/blogs/machine-learning) |
| DoorDash | Consumer Profiles, query rewriting, H-RAG | [careersatdoordash.com/blog](https://careersatdoordash.com/blog) |
| DoorDash | LLMs bridging behavioral silos (RecSys 2025) | [careersatdoordash.com/blog](https://careersatdoordash.com/blog/doordash-llms-bridge-behavioral-silos-in-multi-vertical-recommendations/) |
| Booking.com | AI Trip Planner | [news.booking.com](https://news.booking.com) |
| Instacart | Head-tail LLM search | [instacart.com/company/tech-innovation](https://instacart.com/company/tech-innovation) |
| OpenAI | ChatGPT Shopping + Instant Checkout (2026) | [openai.com/index/chatgpt-shopping-research](https://openai.com/index/chatgpt-shopping-research), [openai.com/index/buy-it-in-chatgpt](https://openai.com/index/buy-it-in-chatgpt) |
| Perplexity | Shopping personalization + Buy with Pro | [perplexity.ai/hub/blog](https://perplexity.ai/hub/blog) |
| Tolan AI | Voice companion matching | [openai.com/index/tolan](https://openai.com/index/tolan) |
| Uber Eats | Two-tower, graph learning, Michelangelo LLMOps | [uber.com/blog](https://uber.com/blog) |
| Airbnb | Listing embeddings, EBR, interleaving | [medium.com/airbnb-engineering](https://medium.com/airbnb-engineering) |
| Criteo | Agentic Commerce Recommendation via MCP (Feb 2026) | [criteo.com](https://www.criteo.com/news/press-releases/2026/02/criteo-introduces-agentic-commerce-recommendation-service-to-power-ai-shopping-assistants/) |
| Kuaishou | OneRec + OpenOneRec (open-source, 2026) | [github.com/Kuaishou-OneRec/OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) |
| Tubi | Inference-time feature injection | [arxiv.org/abs/2512.14734](https://arxiv.org/html/2512.14734) |

### Research Papers

| Paper | Year | Link |
|-------|------|------|
| P5: Pretrain, Personalized Prompt, Predict | 2022 | [arxiv.org/abs/2203.13366](https://arxiv.org/abs/2203.13366) |
| LlamaRec: Two-Stage Recommendation with LLMs | 2023 | [arxiv.org/abs/2311.02089](https://arxiv.org/abs/2311.02089) |
| Data-Efficient Fine-Tuning for LLM Recs (SIGIR '24) | 2024 | [arxiv.org/abs/2401.17197](https://arxiv.org/abs/2401.17197) |
| Prompting LLMs for Recommendation | 2024 | [arxiv.org/abs/2401.04997](https://arxiv.org/abs/2401.04997) |
| LLM RecSys Survey | 2024 | [arxiv.org/abs/2412.13432](https://arxiv.org/abs/2412.13432) |
| Cold-Start Recommendation Survey | 2025 | [arxiv.org/abs/2501.01945](https://arxiv.org/abs/2501.01945) |
| Can LLMs Outshine Conventional Recommenders? | 2025 | [arxiv.org/abs/2503.05493](https://arxiv.org/abs/2503.05493) |
| Netflix Artwork Personalization | 2025 | [arxiv.org/abs/2601.02764](https://arxiv.org/abs/2601.02764) |
| Rec-R1: RL for LLM-Based Recommendations | 2025 | [arxiv.org/abs/2503.24289](https://arxiv.org/html/2503.24289v4) |
| Agentic Feedback Loop (AFL) for RecSys (SIGIR '25) | 2025 | [arxiv.org/abs/2410.20027](https://arxiv.org/abs/2410.20027) |
| OneRec: Generative Recommendation (Kuaishou) | 2025 | [arxiv.org/abs/2502.18965](https://arxiv.org/abs/2502.18965) |
| LLM-Based RecSys Survey (feedback mechanisms) | 2025 | [arxiv.org/abs/2507.21117](https://arxiv.org/html/2507.21117v2) |
| HELM: Human-Centered LLM RecSys Evaluation (WWW '26) | 2026 | [arxiv.org/abs/2601.19197](https://arxiv.org/abs/2601.19197) |
| AlignUSER: World-Model LLM Agents for Rec Evaluation | 2026 | [arxiv.org/abs/2601.00930](https://arxiv.org/abs/2601.00930) |

### RL and Feedback Loop Research

| Paper | Key Finding | Link |
|-------|-------------|------|
| Rec-R1 | RL framework bridging LLMs with RecSys via black-box reward signals | [arxiv.org/abs/2503.24289](https://arxiv.org/html/2503.24289v4) |
| Agentic Feedback Loop (SIGIR '25) | Two-agent loop: +11.52% improvement, no popularity bias amplification | [arxiv.org/abs/2410.20027](https://arxiv.org/abs/2410.20027) |
| ConUCB (WWW 2020) | Contextual bandits with conversational feedback on key-terms | [arxiv.org/abs/1906.01219](https://arxiv.org/abs/1906.01219) |
| Multi-Armed Bandits Meet LLMs (2025) | Survey on combining bandits with LLMs for adaptive recommendation | [arxiv.org/abs/2505.13355](https://arxiv.org/html/2505.13355v1) |
| Continual Recommender Systems (2025) | Tutorial on stability-plasticity balance, streaming feedback | [arxiv.org/abs/2507.03861](https://arxiv.org/abs/2507.03861) |
| Dynamic Feedback Loops in RecSys (2026) | Framework modeling fairness/bias in iterative feedback loops | [Springer JIIS](https://link.springer.com/article/10.1007/s10844-026-01025-y) |

### LLM Fine-Tuning for Recommendations

| Paper | Key Finding | Link |
|-------|-------------|------|
| Data-efficient FT (SIGIR '24) | 2% of training samples achieves full performance | [arxiv.org/abs/2401.17197](https://arxiv.org/abs/2401.17197) |
| Process-supervised LLM recommenders (SIGIR '25) | Process supervision improves LLM rec quality | SIGIR 2025 proceedings |
| Prompting strategies for LLM recs | Systematic comparison of prompting approaches | [arxiv.org/abs/2401.04997](https://arxiv.org/abs/2401.04997) |

### Frameworks and Open Source

| Framework | Link |
|-----------|------|
| RecBole | [recbole.io](https://recbole.io) |
| Microsoft RecAI / InteRecAgent | [github.com/microsoft/RecAI](https://github.com/microsoft/RecAI) |
| OpenOneRec (Kuaishou, 1.7B/8B generative rec models) | [github.com/Kuaishou-OneRec/OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) |
| Agentic Commerce Protocol (OpenAI + Stripe) | [github.com/agentic-commerce-protocol](https://github.com/agentic-commerce-protocol/agentic-commerce-protocol) |
| Eugene Yan's LLM RecSys Guide | [eugeneyan.com/writing/recsys-llm](https://eugeneyan.com/writing/recsys-llm) |
| Eugene Yan's Bandits for RecSys Guide | [eugeneyan.com/writing/bandits](https://eugeneyan.com/writing/bandits/) |
| Awesome-LLM-for-RecSys (paper collection) | [github.com/CHIANGEL/Awesome-LLM-for-RecSys](https://github.com/CHIANGEL/Awesome-LLM-for-RecSys) |
