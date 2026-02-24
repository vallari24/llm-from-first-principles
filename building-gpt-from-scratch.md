# What I Learned Building a Language Model From Scratch

*A deep technical walkthrough for ML engineers — the intuitions no paper gives you.*

---

**Table of Contents**

1. [What Building a Language Model Actually Teaches You](#what-building-a-language-model-actually-teaches-you)
2. [The Simplest Language Model: Bigram](#the-simplest-language-model-bigram)
3. [How Training Actually Works: Chunks, Batches, and Random Sampling](#how-training-actually-works-chunks-batches-and-random-sampling)
4. [Cross-Entropy Loss: The Model's Scorecard](#cross-entropy-loss-the-models-scorecard)
5. [The Shape Journey: From Token IDs to Predictions](#the-shape-journey-from-token-ids-to-predictions)
6. [From Averaging to Attention: The Core Leap](#from-averaging-to-attention-the-core-leap)
7. [Self-Attention: Query, Key, Value](#self-attention-query-key-value)
8. [Softmax Deep Dive](#softmax-deep-dive)
9. [Multi-Head Attention: Multiple Perspectives](#multi-head-attention-multiple-perspectives)
10. [FeedForward: Letting Tokens Think](#feedforward-letting-tokens-think)
11. [Stacking Blocks: Depth Creates Understanding](#stacking-blocks-depth-creates-understanding)
12. [The Gradient Problem and Residual Connections](#the-gradient-problem-and-residual-connections)
13. [Layer Normalization](#layer-normalization)
14. [Dropout: Learning Not to Memorize](#dropout-learning-not-to-memorize)
15. [The Full Architecture: Putting It All Together](#the-full-architecture-putting-it-all-together)
16. [What This Model Is (and Isn't)](#what-this-model-is-and-isnt)
17. [What Building It Teaches You](#what-building-it-teaches-you)

---

## What Building a Language Model Actually Teaches You

I followed Andrej Karpathy's ["Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) and built a character-level transformer from the ground up. Not because I needed another Shakespeare generator — but because I wanted to understand *why* each component of the transformer architecture exists.

Papers tell you what the architecture is. Building it teaches you what breaks when you remove a piece.

By the end of this article, you'll have the intuition to build your own small LLM. Not "I've read about transformers" intuition — "I know exactly why we divide by sqrt(d_k) because I saw what happens when we don't" intuition. Every section ties back to one number: the validation loss. Each architectural addition either earns its keep by pushing that number down, or it doesn't belong.

This is a technical article for ML engineers, researchers, and anyone who wants to understand transformers deeply. We're building a character-level GPT trained on Shakespeare — the same architecture powering GPT-2, GPT-3, and the foundation behind ChatGPT. Just smaller.

---

## The Simplest Language Model: Bigram

The first thing you build isn't a transformer. It's a **bigram model**: given one character, predict the next. No attention, no memory — just a lookup table.

The model is `nn.Embedding(65, 65)` — a 65×65 grid (Shakespeare has 65 unique characters). Row = current character, column = score for next character. Feed in token 20 ("H"), get back 65 raw scores. Softmax turns those into probabilities. Sample from those probabilities to pick the next token.

But the real point of the bigram model isn't the model — it's the **scaffolding**. Data loading, batching, the training loop, loss estimation, text generation — all of that stays identical as the model gets smarter. Every improvement from here on is just swapping out the model internals while everything around it stays the same. This is a design pattern worth internalizing: get the infrastructure right on a toy model, then iterate on the model itself.

That's the whole model. Training nudges the table so correct answers get higher scores. After 5,000 steps, the loss drops from **4.17** (random guessing — that's just `-log(1/65)`) to **~2.58**.

The generated text:

```
MADOY'
'tr thSStlleel, noisuan os
```

Gibberish — but *patterned* gibberish. The model knows "t" is often followed by "h", "q" is almost always followed by "u". Character-pair frequencies, nothing more.

> **Key insight:** This is your baseline. Every architectural addition from here on is measured against this number. If a component doesn't beat the bigram, it's not earning its keep. The journey from 4.17 → 1.98 is the entire story of this article.

**Checkpoint:** We have a lookup table that predicts the next character based on the current one alone. Loss: 2.58. No context awareness at all.

---

## How Training Actually Works: Chunks, Batches, and Random Sampling

You never feed the whole dataset into a transformer at once. Training works on **small random chunks**.

**Block size = context window.** If `block_size` is 8, the model sees 8 tokens and predicts the next one. Why not infinite? Because attention is O(n²) — every token attends to every other token. Double the context, quadruple the cost.

```
block_size 128    →    128² =     16,384 attention scores
block_size 1024   →  1,024² =  1,048,576 attention scores
block_size 8192   →  8,192² = 67,108,864 attention scores
```

**Random sampling** prevents memorizing document order. Each training step grabs a random chunk from a random position. The model sees variety on every step and learns general patterns.

**One subtle thing:** a single chunk of 8 characters gives you 8 training examples, not one. The model simultaneously learns to predict from 1 token of context, 2 tokens, 3 tokens, all the way up to `block_size`. This is how transformers squeeze so much learning out of each piece of data — and why they work with both short prompts and long ones at inference time.

**Batch processing** runs multiple chunks in parallel. A GPU doing matrix math on 1 chunk vs 32 chunks takes nearly the same wall-clock time — feeding one chunk at a time leaves most of the hardware idle. Averaging gradients over 32 chunks also gives a more stable learning signal. One chunk gives a noisy gradient — maybe that chunk was a weird part of the text. Thirty-two chunks average out the noise and give a more reliable direction for updating weights.

The tensor shape makes the two knobs clear:

```
x shape: (batch_size, block_size)    e.g. (32, 8)
          ─────────  ──────────
          32 chunks,  each 8 tokens long
```

**batch_size** = how many chunks to process in parallel (limited by GPU memory). **block_size** = how much context each chunk has (limited by O(n²) attention cost). The model doesn't know or care that chunks are batched together — each chunk is processed independently. Batching is purely a hardware efficiency trick.

> **Key insight:** The training loop teaches the model to predict from any position with any amount of context. This is why LLMs work with short prompts AND long ones — they've trained on every context length from 1 up to `block_size`.

---

## Cross-Entropy Loss: The Model's Scorecard

The model outputs 65 probabilities — one per character. The loss function asks: **how much probability did you put on the correct answer?**

```
loss = -log(probability of correct answer)
```

The scaling tells the whole story:
- 90% confident in correct answer → loss = 0.1
- 30% → loss = 1.2
- 1% → loss = 4.6
- near 0% → loss explodes toward infinity

Being confidently wrong is punished *far* more than being uncertain.

**Why initial loss is ~4.17:** a random model spreads probability equally across all 65 characters. Each gets ~1/65 chance. And `-log(1/65) = log(65) ≈ 4.17`. If your untrained model's loss isn't near this, something is broken — this is a useful sanity check.

> **Key insight:** Loss is measured in nats (natural log units). Every 0.1 drop represents a meaningful improvement in prediction quality. The journey from 4.17 → 1.98 is the entire narrative spine of this article — every section earns its place by pushing this number down.

---

## The Shape Journey: From Token IDs to Predictions

Every tensor in this model has three dimensions:

- **B (Batch)** — how many independent sequences (e.g., 4)
- **T (Time)** — how many tokens in each sequence (e.g., 8)
- **C (Channels)** — how many numbers describe each token (e.g., 32)

The whole model is a pipeline that transforms shapes:

```
"hello" → [27, 4, 11, 11, 14] → [ [0.2, -0.1, ...], ... ] → (B=4, T=8, C=32)
           token IDs              embedding vectors           tensor shape
```

In the bigram model, the token embedding table is `nn.Embedding(65, 65)` — each token looks up 65 scores directly. But once we add attention, the embedding dimension separates from vocabulary size. Two tables now:

```
token_embedding_table    = nn.Embedding(vocab_size, n_embd)   # (65, 32) — "what am I?"
position_embedding_table = nn.Embedding(block_size, n_embd)   # (8, 32)  — "where am I?"
lm_head                  = nn.Linear(n_embd, vocab_size)      # (32 → 65)
```

The flow:

```
idx: (B, T)              ← raw token IDs, just integers
    ↓ token embedding
tok_emb: (B, T, 32)      ← each token → 32-number description
    ↓ + position embedding
x: (B, T, 32)            ← each token now carries what + where
    ↓ lm_head linear layer
logits: (B, T, 65)       ← 65 prediction scores per position
    ↓ softmax → sample
next_token: (B, 1)       ← one prediction per sequence
```

**Why addition, not concatenation?** You could stick them side by side `[token_32 | position_32]` to get 64 numbers. But addition keeps the dimension the same — no extra parameters downstream — and works surprisingly well. Both approaches are valid; addition is simpler and is what the original transformer uses.

> **Key insight:** After embedding, every token is a vector that encodes both "what I am" and "where I am." Everything that follows is about transforming these vectors — teaching them to incorporate information from other tokens in the sequence.

---

## From Averaging to Attention: The Core Leap

The bigram model's fatal flaw: tokens are isolated. Token 47 ("i") always produces the exact same 65 scores regardless of context. "i" after "th" and "i" after "zz" → identical predictions.

We want tokens to communicate — but with a hard rule: **no peeking at the future.** During generation, position 5 can't see positions 6, 7, 8 because those haven't been generated yet.

**Attempt 1: average all past tokens equally.** For each position, take the mean of all token embeddings before it (enforcing the "no future" rule with a lower triangular mask). It works — sort of. But consider predicting the next word in `"The cat sat on the ___"`. The words "sat" and "on" carry the real signal. "The" barely matters. Equal averaging dilutes the useful tokens with irrelevant ones. We need a mechanism where each token can decide **how much to listen to each other token**, based on what's actually relevant.

The mechanism is a lower triangular matrix that enforces "only look backward":

```
Equal weights:              Learned weights (attention):
[1.0  0    0    0  ]        [0.8  0    0    0  ]
[0.5  0.5  0    0  ]        [0.3  0.7  0    0  ]
[0.33 0.33 0.33 0  ]        [0.1  0.1  0.8  0  ]
[0.25 0.25 0.25 0.25]       [0.05 0.6  0.15 0.2]
```

The left matrix is the averaging version — fixed, boring, treats every past token equally. The right matrix is what we're building toward: **data-dependent weights** where the model looks at the actual tokens and computes how relevant each one is.

The trick that makes this work: set future positions to `-inf` before softmax. Why `-inf` and not zero? Because `softmax(0)` is still positive (`e^0 = 1`), which would mean "attend somewhat." But `e^(-∞) = 0` — exactly zero, guaranteed. No information leaks from the future.

```
scores = query @ key.T / sqrt(d)    ← data-dependent (replaces the fixed zeros)
scores = scores.masked_fill(mask == 0, -inf)
weights = softmax(scores)
output = weights @ value
```

> **Key insight:** Attention is just weighted averaging where the weights are computed from the data itself. That's the entire idea. Everything else is engineering to make it work well.

---

## Self-Attention: Query, Key, Value

Each token needs to answer three different questions, and a single vector can't serve all three roles:

```
token → [Q projection] → "What am I looking for?"
      → [K projection] → "What do I contain?"
      → [V projection] → "What do I send if selected?"

score = Q · K^T / √d    →  softmax  →  weights × V  →  output
```

**Why two projections (Q, K) instead of one?** A pronoun like "he" needs to be *found* by verbs that need a subject — that's its key. But "he" itself is *looking for* the noun it refers to — that's its query. "What makes me findable" and "what I look for in others" are different things. One vector can't point in both directions.

**Why a third projection (V)?** Now you know *who* to attend to (from Q·K scores). But the raw embedding carries everything about a token. Maybe you only need a specific aspect. The token "not" might be *found* because it's a negation (key), but what's *useful to extract* is different from what made it findable. Value decouples routing from content.

The actual computation:

```
q = Wq @ x    # (B, T, head_size) — each token's search vector
k = Wk @ x    # (B, T, head_size) — each token's findability vector
v = Wv @ x    # (B, T, head_size) — each token's readable content

scores = q @ k.T    → (B, T, T) — every query dot-producted with every key
```

**Scaling by √d_k:** Without it, dot products grow with dimension. With `head_size=64`, scores land around ±8 instead of ±1. Large scores push softmax into saturation — one token gets ~100% weight, everything else ~0%. Gradients vanish. Dividing by `sqrt(head_size)` keeps scores around ±1, keeping softmax in its useful range.

After masking the future and applying softmax, each token gets a weighted blend of value vectors from the tokens it found relevant:

```
pos2's output = 0.35 × v[pos0] + 0.26 × v[pos1] + 0.39 × v[pos2]
```

Compare with the old fixed `[0.33, 0.33, 0.33]`. Now the weights are different for every position and change depending on the actual tokens. That's the whole point.

> **Key insight:** Q, K, V aren't arbitrary — they solve a specific design problem. Q = "what am I looking for?", K = "what do I have to offer?", V = "what do I actually send?" Three separate questions, three separate learned projections.

**Why "self" attention?** Queries, keys, and values all come from the **same sequence**. Each token decides what to attend to within its own sequence. In encoder-decoder models (like translation), "cross-attention" has queries from one sequence and keys/values from another. Same mechanism, different source. Understanding self-attention means you understand cross-attention too — just change where Q, K, V come from.

**Loss after single-head attention: ~2.40** (down from 2.58). The model can now use context. Generated text shifts from character soup to word-shaped blobs:

```
Thabee othy theeandurdeves nd thitle, swhiem dsy an'g ply minds
```

You can see word-like structures emerging — "theeandurdeves" has recognizable English fragments. The model is learning that certain *sequences* of characters go together, not just pairs.

---

## Softmax Deep Dive

Softmax turns raw scores into probabilities. Why this specific function?

**Why e^x?** You need something always positive, that preserves ordering, and amplifies differences:

```
x:    -5      0      1      5
e^x:   0.007  1.0    2.7    148.4
```

Always positive. Bigger input → bigger output. And crucially, gaps get stretched exponentially:

```
Inputs:   [1,    2,    3]        — gaps of 1
softmax:  [0.09, 0.24, 0.67]    — biggest input dominates
```

A linear scheme would give `[0.17, 0.33, 0.50]`. Softmax gives `[0.09, 0.24, 0.67]`. The winner wins *more*. This is what lets attention be sharp — when one token is more relevant, it gets *much* more weight, not slightly more.

**Temperature** controls sharpness by dividing scores before softmax:

```
scores: [2.0, 1.0, 0.1]

T = 1.0 (normal):    → [0.66, 0.24, 0.10]   standard
T = 0.5 (sharper):   → [0.87, 0.12, 0.01]   commits hard to the winner
T = 2.0 (smoother):  → [0.43, 0.26, 0.17]   hedges its bets
T → 0:               → [1.0,  0.0,  0.0]    argmax — winner takes all
```

This is exactly what ChatGPT's temperature slider does. And it's why attention divides by `√d_k` — dot product scores can get large, making softmax too sharp. Dividing calms it down.

**Numerical stability:** With big numbers, `e^1000` overflows. The fix: subtract the maximum first. `softmax(x) = softmax(x - max(x))`, mathematically identical but keeps numbers small. Every framework does this automatically.

> **Key insight:** Softmax is a competition. Higher scores win exponentially more weight. This is why attention is so effective — it can sharply focus on relevant tokens while nearly ignoring irrelevant ones.

---

## Multi-Head Attention: Multiple Perspectives

A single attention head computes one set of Q, K, V projections — one notion of "relevant." But language has many simultaneous relationships. A word like "it" needs to figure out: what noun does it refer to? What adjective described that noun? What action is being done to it? One attention pattern can't capture all of these at once.

```
Embedding (32 dims): [████████████████████████████████]
                           ↓ split into 4 heads
Head 1: [████████]   "grammar patterns"
Head 2: [████████]   "position relationships"
Head 3: [████████]   "character co-occurrence"
Head 4: [████████]   "semantic grouping"
                           ↓ concatenate
Output (32 dims):  [████████████████████████████████]
```

The math stays the same size: 1 head × 32 dims = 32-dim output. 4 heads × 8 dims = 32-dim output. Each head is smaller, but collectively they cover more ground because each can specialize. We don't tell the heads what to specialize in — gradient descent discovers useful patterns because different attention patterns reduce loss in different ways.

> **Key insight:** Multi-head attention doesn't add parameters — it redistributes them. 4 heads of 8 dimensions each learn richer patterns than 1 head of 32 dimensions, because each head can specialize. The diversity comes from random initialization + the pressure to collectively explain the data.

**Loss after multi-head: ~2.28** (down from 2.40). Generated text shows more structure — real names appear, punctuation lands in roughly the right places:

```
QARENSLOCOFOR:
Ta.
Martin,
Ox Jouatephiten!
Cius! Balland o bin a and wor'd
```

"Martin" is a real name. The colon-after-name pattern (speaker labels in the play) is emerging.

---

## FeedForward: Letting Tokens Think

Attention lets tokens **communicate** — gather information from other tokens. But after gathering, each token needs to **process** what it collected. That's the feedforward layer.

```
┌─────────────────────────┐
│     TRANSFORMER BLOCK    │
│                          │
│  Input ──→ LayerNorm     │
│            ↓             │
│         Attention        │  ← "Tokens talk to each other"
│            ↓             │
│  Input + Attention ──→   │  ← Residual connection
│            ↓             │
│         LayerNorm        │
│            ↓             │
│        FeedForward       │  ← "Each token thinks alone"
│            ↓             │
│  Input + FeedForward ──→ │  ← Residual connection
│            ↓             │
│          Output          │
└─────────────────────────┘
```

Structure: `Linear(n_embd → 4*n_embd) → ReLU → Linear(4*n_embd → n_embd)`. Expand 4x (room to think), apply ReLU (decide what's important), compress back down.

**Why ReLU matters:** Without the non-linearity, stacking attention + linear + attention + linear would collapse into one big attention + one linear. Linear operations compose into linear operations. ReLU breaks this — it zeros out negative values and passes positive ones through. Simple, but it means each layer can learn patterns that no single layer could represent. This is what makes deep networks "deep" in a meaningful sense.

Crucially, feedforward operates **per-token independently**. All cross-token communication happened in attention. Feedforward is purely "given what I've gathered, what do I make of it?"

> **Key insight:** The transformer block is a two-phase cycle: first tokens communicate (attention), then they compute (feedforward). Communication without computation would be like a meeting where nobody takes notes.

**Loss after feedforward: ~2.24** (down from 2.28). A small improvement at this scale, but this is the piece that unlocks depth. At scale, feedforward layers contain the majority of a model's parameters and are where much of the "knowledge" is stored (GPT-3's feedforward inner dimension is 4 × 12,288 = 49,152).

---

## Stacking Blocks: Depth Creates Understanding

Why stack multiple transformer blocks? Each block builds on the *output* of the previous block, not the raw input.

Block 1 sees raw character embeddings. After Block 1, the representation of "she" isn't just "she" anymore — it's "she (who appears after the doctor)." When Block 2's attention connects "would" to this enriched "she", it's effectively learning a pattern about the *doctor*, even though it never directly attended to "doctor" itself.

The critical thing: **Block 2 does NOT see raw tokens.** It sees tokens that have already been transformed by Block 1. Each block's attention operates on enriched representations, which means each attention step can capture higher-order patterns:

```
Block 1: learns character pairs and simple patterns
Block 2: learns word-level patterns from Block 1's representations
Block 3: learns grammar, clause structure from Block 2's output
Block 4: learns longer-range dependencies, style patterns
```

Multi-head = breadth (multiple perspectives per layer). Stacking = depth (progressively abstract representations). You need both:

- **Multi-head only (wide but shallow):** 4 heads, all looking at raw tokens. Head 1 notices "she"→"doctor". Head 3 notices "would"→"recover". But no head can *combine* these observations — they independently process the same raw input. They can't build on each other's findings.
- **Single-head stacked (deep but narrow):** Block 1 notices one relationship. Block 2 builds on it. But at each level, only one attention pattern — it might catch tense but miss clause boundaries. One pattern per level limits what gets carried forward.
- **Both together:** each block has multiple heads finding different patterns, then the next block has multiple heads finding patterns *in those patterns*. It's like having a team of analysts at each level, where each level reads the previous team's combined report.

Research on what real GPT layers learn shows exactly this pattern: early blocks capture character patterns and common word pieces; middle blocks capture syntax, grammar, subject-verb agreement; late blocks capture semantics, factual recall, and reasoning.

> **Key insight:** This is the actual architecture of every GPT model. GPT-2 Small is 12 blocks × 12 heads = 144 simultaneous attention patterns across 12 levels. GPT-3 is 96 blocks × 96 heads = 9,216 patterns across 96 levels. The magic isn't in any single block — it's in what emerges from stacking them.

**Loss after 4 stacked blocks with residuals and LayerNorm: ~2.13** (down from 2.24). The jump from a single block to four is significant — depth compounds.

---

## The Gradient Problem and Residual Connections

Stacking blocks creates a problem. During backpropagation, the gradient for Block 1's weights must pass through every intermediate block. Each block multiplies the gradient by its weights — typically small numbers (0.1–0.8). The gradient shrinks at each step:

```
3 blocks:   gradient × 0.6 × 0.6           = 0.36
10 blocks:  gradient × 0.6^9               = 0.01
96 blocks:  gradient × 0.6^95              ≈ 0.0000000000000000001
```

Block 1 gets a gradient so small that `learning_rate × gradient ≈ nothing`. Block 1 stops learning. It's dead weight. This is the **vanishing gradient problem**.

The fix is exactly two characters: **`+`**.

```
# Without residual — each block REPLACES the input
x = self.attention(x)

# With residual — each block ADDS TO the input
x = x + self.attention(x)
```

With `output = input + block(input)`, the gradient at the `+` splits into two copies: one goes through the block (might shrink), and one goes directly backward (untouched). The skip path delivers the gradient at full strength — no multiplication by block weights.

```
Input ════════════════════════════════════════════► Output
           ↕            ↕            ↕
        Block 1      Block 2      Block 3
        (adds)       (adds)       (adds)

Gradient ◄════════════════════════════════════════ Loss
           (flows freely through the + highway)
```

The `+` also changes what each block learns. Without residual, each block must produce a complete output from scratch — the input is thrown away. With residual, each block only adds a **small correction** to the existing representation:

```
Without residual:
  "she" → [Block 1] → [completely new vector]
  Original embedding is gone.

With residual:
  "she" → [Block 1 produces small correction] → original + correction
  "she" is still there, just refined.
```

Each block is like an editor making notes on a document, not rewriting it from scratch. The original information is always preserved, and each block contributes an additive refinement.

By Block 4, the representation contains: raw embeddings + Block 1's correction + Block 2's correction + Block 3's correction + Block 4's correction. Nothing is lost. In the transformer literature, this main flow is called the **residual stream** — and understanding it as a highway that blocks read from and write to (in both forward and backward directions) is one of the most useful mental models for understanding how transformers work internally.

> **Key insight:** Without residual connections, deep transformers simply cannot train. The `+` is the single most important character in the entire architecture. It turns a fragile 96-layer chain into a robust highway where gradients flow freely.

---

## Layer Normalization

As data flows through many blocks, the scale of numbers drifts. One block might output values around 500, another around 0.002. Attention computes Q·K dot products — if values are in the thousands, dot products are in the millions, softmax gives 100% weight to one token and 0% to everything else. No blending, no learning.

LayerNorm normalizes each token's features to mean=0, std=1, then scales and shifts with learned parameters:

```
token features: [200, 800, 400, 600]
    mean = 500, std = 224
normalized:     [-1.34, 1.34, -0.45, 0.45]
```

The relative pattern is preserved. But values are now in a small, consistent range. Every downstream layer can count on receiving numbers roughly between -2 and 2.

**Why LayerNorm, not BatchNorm?** BatchNorm normalizes each feature *across the batch* — it asks "for feature #3, what's the average across all sequences?" This breaks for language in two ways. First, sequences have variable lengths — what does it mean to average feature #3 at position 150 when most sequences don't even have 150 tokens? Second, during generation `batch_size=1`, and there's no batch to normalize across.

LayerNorm normalizes each token *across its features* — it asks "for this one token, what's the average across all its features?" Each token normalizes itself using only its own numbers. Works with any batch size, any sequence length.

**Pre-norm (modern):** In our implementation, LayerNorm goes *before* attention and *before* feedforward, not after. The normalized values go into attention/feedforward, but the residual connection adds back the raw `x`:

```
x = x + self.attention(self.ln1(x))    # normalize → attend → add back
x = x + self.feedforward(self.ln2(x))  # normalize → feedforward → add back
```

Notice: the normalized values go into attention/feedforward, but the **residual connection adds back the raw `x`**, not the normalized version. The residual stream carries unnormalized values (preserving the gradient highway). Each block normalizes its own input right before processing it:

```
Residual stream:  x ────────────────────(+)────────────────────(+)───→
                          ↓              ↑           ↓          ↑
                       LayerNorm         │        LayerNorm     │
                          ↓              │           ↓          │
                       Attention ────────┘        FeedFwd ──────┘
                       (sees clean                (sees clean
                        numbers)                   numbers)
```

Both systems work without interfering. Each block gets clean, well-scaled input. The residual stream stays untouched.

LayerNorm also has learnable parameters — a scale (γ) and shift (β) per feature, initialized to γ=1, β=0. The model can learn to adjust them if the normalized range isn't optimal for specific features.

> **Key insight:** LayerNorm is housekeeping that enables everything else. It keeps numbers in a well-behaved range so softmax doesn't saturate and gradients don't explode. Unsexy but essential.

---

## Dropout: Learning Not to Memorize

Randomly zero out neurons during training — 20% in our model. Different random neurons every step. The model can't rely on any single neuron being available, so it's forced to build redundant representations.

Dropout shows up in three places: after attention weights (randomly drop some connections between tokens), after the multi-head projection, and after feedforward. During generation, dropout turns off — the full network is used.

With ~1M characters of Shakespeare and 209K parameters, the model can memorize the training data. You can see this in the training logs: without dropout, train loss keeps dropping while val loss plateaus — the classic overfitting signature. Dropout is what keeps them tracking together, forcing the model to learn *patterns* instead of memorizing specific sequences.

Why 20%? It's a common default that works well in practice. Too much dropout (50%+) cripples the model's capacity. Too little (5%) doesn't provide enough regularization. 20% is the sweet spot where the model learns robust representations without losing too much information during any given forward pass.

> **Key insight:** Dropout is the difference between a model that memorizes Shakespeare and one that learns English.

---

## The Full Architecture: Putting It All Together

Here's the complete model, bottom to top — what happens when a token sequence enters and a prediction comes out:

```
"The cat sat on the ___"
         │
         ▼
┌─────────────────────────────┐
│  Token Embedding            │  Each character → 64-dim vector
│  + Position Embedding       │  Each position → 64-dim vector
│  (added together)           │  Now: "what I am" + "where I am"
└─────────────────────────────┘
         │
         ▼
┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
│   Transformer Block (×4)     │
│                              │
│   LayerNorm                  │
│       ↓                      │
│   Multi-Head Attention       │
│   (4 heads × 16 dims)       │
│       ↓                      │
│   x = x + attention          │  ← residual
│       ↓                      │
│   LayerNorm                  │
│       ↓                      │
│   FeedForward                │
│   (64 → 256 → ReLU → 64)    │
│       ↓                      │
│   x = x + feedforward        │  ← residual
│                              │
└ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
         │
         ▼
┌─────────────────────────────┐
│  Final LayerNorm             │
│  Linear (64 → 65)           │  Project to vocabulary size
│  Softmax → Sample            │  Pick the next character
└─────────────────────────────┘
         │
         ▼
      next character
```

Final model specs: **209K parameters**, 4 layers, 4 heads, 64-dim embeddings, block_size=32.

### Walking through a single forward pass

Let's trace what happens for the input `"The cat"`:

1. **Tokenize:** `"The cat"` → `[32, 46, 43, 1, 41, 39, 58]` (7 character IDs)
2. **Token embedding:** each ID looks up a row in a (65, 64) table → shape becomes `(1, 7, 64)`
3. **Position embedding:** positions 0–6 look up rows in an (32, 64) table → `(7, 64)`, broadcast-added
4. **Now each token is a 64-dim vector** encoding both what it is and where it sits
5. **Block 1:** LayerNorm → 4-head attention (each head 16 dims) → residual add → LayerNorm → feedforward (64→256→ReLU→64) → residual add
6. **Blocks 2, 3, 4:** same operations, each refining the representations further
7. **Final LayerNorm** → **Linear (64→65)** → 65 logits per position
8. **Take the last position's logits** → softmax → sample → next character

During training, all 7 positions compute losses simultaneously against their targets. During generation, only the last position matters for the next prediction.

### The loss tells the whole story

Every component we added had a measurable impact:

```
Loss
4.17 |████████████████████████████████████████| Random guess
2.58 |██████████████████████████              | Bigram (lookup table)
2.40 |████████████████████████                | + Self-attention
2.28 |██████████████████████                  | + Multi-head (4 heads)
2.24 |█████████████████████                   | + FeedForward
2.13 |████████████████████                    | + Residual + LayerNorm (4 blocks)
1.98 |██████████████████                      | + Scale up + Dropout
     └────────────────────────────────────────┘
```

The generated text makes the improvement tangible:

**Bigram (2.58):** `MADOY' 'tr thSStlleel, noisuan os` — character soup with occasional recognizable pairs.

**+ Self-attention (2.40):** `Thabee othy theeandurdeves nd thitle, swhiem` — word-shaped blobs. You can almost read it.

**+ Multi-head (2.28):** `QARENSLOCOFOR: Ta. Martin, Ox Jouatephiten!` — real names appear, punctuation in roughly the right places, "and" used correctly.

**+ Full architecture, scaled up (1.98):**

```
Gown go thath all causeary barstens.
Be men Entymust think, my so tally,
So exay:
Thoun rich tashe Mo coneous own
Theis pllistonce.

CLEYMANUS:
What hert, hombsters witichous, hins, loodw--sworde son
me thou horeewn?
```

Speaker names with colons. Line breaks in verse-appropriate places. Words that almost parse as Elizabethan English. The structure is unmistakably Shakespeare — even if the content is nonsense.

### What the numbers tell you

**The biggest single win is the dumbest technique.** Going from random to a lookup table (bigram) drops loss by 1.59. That's just memorizing which characters follow which — no intelligence, pure statistics. Most of the "easy" prediction in language comes from local patterns.

**Context is the second biggest win.** Adding attention (0.18 drop) gives the model 8 characters of memory instead of 1. It can now learn "the" and "ing" as patterns, not just character pairs.

**Diminishing returns at each step — but they compound.** Each technique adds less than the previous one. But this is a toy model with 64 embedding dimensions and 32 tokens of context. Real transformers stack 96 blocks with 12,288-dimensional embeddings and 100k tokens of context. The same techniques that gave us 4.17 → 1.98 on character-level Shakespeare give us GPT-4 on the full internet. The architecture is identical — just scaled up.

> **Key insight:** Each component doesn't just lower a number — it unlocks a qualitatively different *kind* of pattern the model can learn. Bigram learns pairs. Attention learns sequences. Multi-head learns simultaneous relationships. Feedforward learns non-linear transformations. Stack them and scale them up, and you get language.

---

## What This Model Is (and Isn't)

This **is** a GPT. Same architecture as GPT-2 and GPT-3, just smaller:

| | Our model | GPT-2 Small | GPT-3 |
|---|---|---|---|
| Layers | 4 | 12 | 96 |
| Embedding dim | 64 | 768 | 12,288 |
| Heads | 4 | 12 | 96 |
| Parameters | 209K | 117M | 175B |
| Context window | 32 | 1,024 | 2,048 |
| Training data | 1M chars Shakespeare | ~10B tokens WebText | 300B+ tokens internet |

Same attention. Same residual connections. Same LayerNorm. Same feedforward. Same dropout. The scale changes, but the architecture doesn't. The principles you learned building this tiny model are the exact same principles running inside every major LLM.

**What we skipped and why:**

- **RoPE** (rotary position embeddings) instead of learned position tables — they encode relative distance between tokens rather than absolute position, and generalize better to sequences longer than those seen during training
- **Flash attention** — a memory-efficient implementation that avoids materializing the full (T, T) attention matrix, making the O(n²) cost practical at scale
- **KV caching** — during autoregressive generation, cache previously computed key/value vectors so you don't recompute them for every new token
- **RMSNorm** replaces LayerNorm — slightly faster because it skips the mean subtraction step
- **GeLU** replaces ReLU — smoother activation that doesn't have the "dead neuron" problem where neurons stuck at zero never recover
- **Modern training recipes** — cosine learning rate schedules, gradient clipping, mixed precision training, distributed data parallelism

These are important engineering improvements, but they're all optimizations on top of the same core architecture. The residual stream, the attention mechanism, the feedforward blocks, the overall structure — identical to what we built.

> **Key insight:** This IS a GPT — same architecture, different scale. When you read about a "96-layer transformer with 12,288-dimensional embeddings," you now know exactly what that means: 96 rounds of communicate-then-think, flowing through a residual stream. The principles are identical.

---

## What Building It Teaches You

Building a language model from scratch gives you intuition no paper can.

You understand *why* each component exists because you saw what happens without it. Attention without scaling? Softmax saturates and gradients vanish. Depth without residual connections? Early blocks stop learning. Communication without computation? The model can gather information but can't process it. Each piece solves a specific, observable problem.

These same blocks power every major LLM — ChatGPT, Claude, Gemini, Llama. The scale changes by five orders of magnitude. The architecture doesn't. When you read about a "96-layer transformer with 12,288-dimensional embeddings and 96 attention heads," you now know exactly what that means: 96 rounds of communicate-then-think, with 96 parallel perspectives per round, flowing through a residual stream that keeps gradients alive across all 96 layers.

The biggest surprise wasn't any single component. It was how *few* ideas make up the whole thing. The transformer is really just five ideas combined:

1. **Embedding** — turn tokens into vectors
2. **Attention** — let vectors talk to each other with learned, data-dependent weights
3. **Feedforward** — let each vector process what it heard
4. **Residual connections** — keep gradients alive through depth
5. **Normalization** — keep numbers in a well-behaved range

That's it. Everything else — multi-head, stacking, dropout, scaling, positional encoding — is engineering to make these five ideas work better at scale. When someone describes a new transformer variant, you can map every change back to one of these five.

The foundation for everything in this article is Andrej Karpathy's ["Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video. If you haven't watched it, watch it. If you have, build it yourself. The act of implementing each piece — typing the matrix multiplications, watching the loss drop, reading the generated text evolve from gibberish to almost-Shakespeare — cements the understanding in a way that reading never will.

For the formal treatment, read ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — the paper that started it all. After building the model yourself, every equation in that paper will feel like something you already know.

---

### References

- **Video:** Andrej Karpathy — ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- **Paper:** Vaswani et al. — ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017)
- **Notebooks:** All the code behind this article — from bigram to full transformer — is in my [llm-from-first-principles notebooks](https://github.com/vallari24/llm-from-first-principles/tree/main/notebooks). Follow along, run the cells, break things.

---

*If this was useful, [subscribe](https://vallari.substack.com) for more deep technical dives.*
