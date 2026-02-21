# Building GPT From Scratch: What I Actually Learned

Following Andrej Karpathy's ["Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video and building deep intuition for how LLMs work under the hood.

---

## The Bigram Model: Why Start So Simple?

The first thing you build isn't a transformer — it's a **bigram model**. Given one character, predict the next. No attention, no memory, just a lookup table.

The model itself is almost useless. But the point isn't the model — it's the **scaffolding**: data loading, batching, training loop, loss estimation, text generation. All of that stays identical as the model gets smarter. Every improvement from here on is just swapping out the model while everything around it stays the same.

You know it's working when the loss drops from ~4.17 (pure random guessing across 65 characters — that's just ln(65)) to ~2.5. The generated text looks like it *could* be English if you squint, but it's nonsense — because one character of context is never enough.

## Tokenization: The Codebook Tradeoff

A neural network can't read text. It only sees numbers. So the first decision is: how do you chop up text into numbers?

The simplest way: give every character its own ID. `'a'` → 0, `'b'` → 1, etc. Shakespeare has 65 unique characters, so your "codebook" has 65 entries. Easy to understand, easy to implement.

The problem? Sequences get very long. And transformers have a **fixed context window** — say 1024 tokens. At character-level, that's ~1024 characters, maybe a paragraph. The model can't see beyond that.

Real LLMs use **subword tokenizers** that learn to group common character patterns into single tokens — `" the"`, `"ing"`, `"Hello"` each become one token. Now 1024 tokens covers ~4000 characters, several pages. More context = better predictions.

But there's a cost. A bigger codebook means:
- A bigger embedding table (more parameters to train)
- Rare tokens get fewer training examples, so the model learns them worse
- The tokenizer itself becomes a source of weird behavior (ask any LLM to count letters in a word — it struggles because it doesn't see individual characters)

This is the core tradeoff: **small codebook + long sequences vs. large codebook + short sequences**. Real LLMs land around 32k–100k tokens as the sweet spot. For learning, character-level is perfect — you can see exactly what the model does without tokenization adding mystery.

## Train/Val Split: Why Hold Back Data?

You split the data so you can answer one question: **is the model learning language, or memorizing text?**

Training loss going down just means the model is fitting the data it's seen. That could be real learning, or it could be rote memorization. The validation set is text the model never trains on — if val loss tracks train loss, the model is genuinely picking up patterns. If val loss plateaus while train loss keeps dropping, it's overfitting: memorizing rather than understanding.

The intuition shifts at scale. When you're training on tiny Shakespeare (~1M characters), a small model can memorize the whole thing — the val split is critical. When you're pretraining a real LLM on hundreds of billions of tokens from the internet, the model barely sees the same text twice, so overfitting is less of a worry. The val set becomes more of a health check you glance at periodically, not the thing keeping you honest.

But the habit matters. Having a validation set is cheap (just hold back 10% of data) and the signal it gives you — "is the model still learning or has it started memorizing?" — is one of the most important things to watch during training.

## Chunks and Random Sampling: How Training Actually Works

You never feed the whole dataset into a transformer at once. Not even close. Training works on **small random chunks**.

**Why chunks?** The transformer has a fixed context window (`block_size`). If block_size is 8, the model sees 8 tokens and predicts the next one. Bigger block_size = more context = better predictions, but the cost of attention grows quadratically (every token attends to every other token). So you pick the largest block_size you can afford computationally.

**Why random?** Each training step grabs a random chunk from a random position in the data. If you fed data in order — page 1, then page 2, then page 3 — the model would learn patterns specific to that ordering, and early in training it'd only know how the beginning of the text sounds. Random sampling means every batch is a mix of different parts. The model sees variety on every step and learns general patterns instead of memorizing a sequence.

**At pretraining scale**, same idea, just bigger. Billions of documents, and each training step grabs a batch of random chunks from random documents. The model never processes "the whole internet" — it sees millions of small windows, randomly sampled, across many passes. The randomness is what keeps learning general.

**One subtle thing:** a single chunk gives you multiple training examples, not just one. Take the chunk `"Hello th"` (8 characters). Because each position can only see what came before it, the model is simultaneously learning to predict:

```
Given "H"           → predict "e"
Given "He"          → predict "l"
Given "Hel"         → predict "l"
Given "Hell"        → predict "o"
Given "Hello"       → predict " "
Given "Hello "      → predict "t"
Given "Hello t"     → predict "h"
Given "Hello th"    → predict "e"
```

That's 8 predictions from one chunk, all computed in a single forward pass (not 8 separate runs — that's what makes transformers efficient). A batch of 32 chunks of size 8 means 256 predictions per training step. This is how transformers squeeze so much learning out of each piece of data.

### Why train on all context lengths?

During generation, the model starts with just 1 token. Then it generates the next, and now has context of 2. Then 3. If you only trained on full-length context (always exactly block_size tokens), the model wouldn't know what to do with 1 or 3 tokens of context. By training on every length from 1 up to block_size, the model learns to make the best prediction it can with whatever context it has — because at inference time, it *will* encounter every length.

### Why not make block_size infinite?

Because attention is O(n²). Every token looks at every previous token. Double the context, quadruple the cost:

```
block_size 128    →    128² =     16,384 attention scores
block_size 1024   →  1,024² =  1,048,576 attention scores
block_size 8192   →  8,192² = 67,108,864 attention scores
```

So block_size is a **compute budget**. You pick the longest context you can afford, knowing anything beyond it is invisible to the model. This is a real limitation — it's why early GPT models would "forget" the beginning of long conversations. Newer models push context to 100k+ tokens with tricks to tame the quadratic cost, but the fundamental tradeoff remains: more context = better understanding = way more compute.

### Batch size: why process multiple chunks at once?

You *could* train on one chunk at a time. It would work. But it's slow for two reasons:

1. **GPUs are parallel machines.** A GPU doing matrix math on 1 chunk vs 32 chunks takes almost the same wall-clock time. Feeding one chunk at a time leaves most of the hardware idle. Batching is how you fill the GPU.

2. **Smoother learning signal.** One chunk gives you a noisy gradient — maybe that chunk was a weird part of the text. Averaging gradients over 32 chunks gives a more stable direction for updating weights. Too small = erratic updates. Too large = wastes memory for diminishing returns.

The tensor shape makes it clear:

```
x shape: (batch_size, block_size)    e.g. (32, 8)
         ─────────  ──────────
         32 chunks,  each 8 tokens long
```

These are two independent knobs:
- **batch_size** = how many chunks to process in parallel (limited by GPU memory)
- **block_size** = how much context each chunk has (limited by O(n²) attention cost)

The model doesn't know or care that chunks are batched together — each chunk is processed independently. Batching is purely a hardware efficiency trick. Running batch_size=1 thirty-two times would give mathematically identical learning to batch_size=32 once (aside from the gradient averaging, which just smooths things out).

## The Bigram Model: What's Actually Happening Inside

**Token** — just a number representing a character. After encoding, `"Hello"` becomes `[20, 43, 50, 50, 53]`. Each number is a token.

**The problem:** given a token, predict which token comes next.

**The solution: a lookup table.** Imagine a 65×65 grid (65 characters in Shakespeare). Row = current character, column = next character. Each cell holds a score — "how likely is this next character given the current one?" Row for "H" might have a high score in the "e" column (because "He" is common) and a low score in the "z" column.

In PyTorch, this table is called an **embedding**: `nn.Embedding(65, 65)` — 65 rows, 65 scores per row. Feed in token 20, get back row 20: 65 raw scores.

**Those raw scores are logits.** Just a word for "unnormalized scores" — they can be any number, positive or negative. They're not probabilities yet. To get probabilities, you pass them through **softmax**, which squishes them into numbers between 0 and 1 that add up to 1:

```
logits:        [2.1, -0.5, 0.8, ...]   ← raw scores
    ↓ softmax
probabilities: [0.4,  0.03, 0.11, ...]  ← 0 to 1, sum to 1
```

Higher logit = higher probability. The model samples from these probabilities to pick the next token.

**Training = making the table less wrong.** The table starts random — every next character is equally likely. The loss function measures "how surprised was the model by the actual answer?" If the real next character was "e" but the model only gave it 1% probability, that's high loss. Training nudges the table so correct answers get higher scores. Repeat a few thousand times and the table captures which characters tend to follow which.

That's the whole model: a lookup table that learns character-pair frequencies. No memory, no context beyond the single previous character. Everything after this in the video is about adding context.

## Cross-Entropy Loss: How the Model Knows It's Wrong

The model outputs 65 probabilities — one per character. The loss function asks: **how much probability did you put on the correct answer?**

High probability on the right answer → low loss. Low probability → high loss.

**Why not just subtract?** You might think: subtract the prediction from the answer, see how far off. But the model outputs 65 probabilities and the target is a single index — they're different kinds of things. It's like a multiple-choice exam with 65 options. You don't measure "how far is your answer from the right one" — you measure "how confident were you in the right answer?"

The formula: `loss = -log(probability of correct answer)`

```
Model's probabilities: [0.02, 0.01, ..., 0.30, ..., 0.01]
                                         ↑
                                    index 47 — the correct answer

Loss = -log(0.30) = 1.2
```

Why -log? Because of how it scales:
- 90% confident in correct answer → loss = 0.1 (good)
- 30% → loss = 1.2
- 1% → loss = 4.6
- near 0% → loss explodes toward infinity

Being confidently wrong is punished *far* more than being uncertain.

With multiple predictions, you average the losses. If you made 2 predictions and the correct answers got 30% and 15% probability: average loss = (-log(0.30) + -log(0.15)) / 2 = 1.55.

**Why initial loss is ~4.17:** a random model spreads probability equally across all 65 characters, so each gets ~1/65. And `-log(1/65) = log(65) ≈ 4.17`. If your untrained model's loss isn't near this, something is broken.

## Forward and Generate: A Complete Walkthrough

These are the two things the model does: **forward** (look up the cheat sheet) and **generate** (use it to write).

### forward — "what does the cheat sheet say?"

The embedding table IS the cheat sheet — a 65×65 table where each row is one character's scores for what comes next. Forward just looks up rows.

Say our input is `"Hi"` → `[20, 47]`:

```
Look up row 20 ("H") → [2.1, -0.5, 0.8, ..., 3.2, ...]   65 scores
Look up row 47 ("i") → [0.3,  1.7, -0.2, ..., 0.1, ...]   65 scores

Result shape: (1, 2, 65) — 1 sequence, 2 tokens, 65 scores each
```

That's all forward does. Look up rows in a table.

During training, we also check "how wrong were we?" If the real next characters were `"i"` and `" "`, cross-entropy checks: for each prediction, how much probability did you put on the correct answer? Average those → that's the loss.

### How targets work — "the input shifted by one"

If the input is `[20, 47]` ("Hi"), the model makes 2 predictions, not 3. Each token predicts what comes after it:

```
Input:    [20,  47]      "H" and "i"
           ↓    ↓
Predicts: [??,  ??]      "what comes after H?" and "what comes after i?"
Target:   [47,  32]      the actual answers: "i" and " "
```

Targets are always the input shifted by one position — "what actually came next in the text."

### Sampling — "rolling a weighted die"

Imagine a 65-sided die, but unfair. Side "e" takes up 30% of the surface. Side "z" takes 0.1%. When you roll, "e" comes up far more often, but you *could* land on "z". That's what `torch.multinomial` does — randomly picks one token where each token's chance of being picked equals its probability. This randomness is why the model generates different text each time.

### generate — "write using the cheat sheet, one character at a time"

Start with `idx = [[0]]` (just a newline character).

**Iteration 1:**
```
idx = [[0]]
Forward: look up row 0 → shape (1, 1, 65)
Take last position → (1, 65) — removes the middle dimension,
                      just "65 scores for what follows newline"
Softmax → probabilities
Roll weighted dice → say we get 20 ("H")
Append: [[0]] → [[0, 20]]
```

**Iteration 2:**
```
idx = [[0, 20]]
Forward: look up rows 0 AND 20 → but we only care about the LAST one
Take last position → 65 scores for "what follows H?"
Softmax → probabilities
Roll dice → say 43 ("e")
Append: [[0, 20]] → [[0, 20, 43]]
```

**Iteration 3:**
```
idx = [[0, 20, 43]]
Forward: look up rows 0, 20, 43 → only use the LAST one
Take last position → "what follows e?"
Roll dice → 50 ("l")
Append: [[0, 20, 43, 50]]
```

Decode `[0, 20, 43, 50]` → `"\nHel"` — the start of nonsense Shakespeare.

**The key limitation:** even though the sequence grows, the bigram model only ever uses the LAST token to predict. `"e"` after `"th"` and `"e"` after `"zzz"` produce identical predictions — same row in the table, same scores. That's the problem attention solves next.

## The Training Loop: Nudging Numbers Toward Less Wrong

Training has no magic. It's the same five steps, repeated thousands of times:

**1. Grab random data** — pick random chunks from the training set. Different chunks every time, so the model sees variety.

**2. Predict and measure** — forward pass. Look up the table, get scores, compare against targets, compute loss. PyTorch secretly records every math operation in a "computation graph" so it can work backwards later.

**3. Clear old gradients** — gradients are "which direction should each number move." They need to be reset each iteration, otherwise they'd pile up from previous steps and give wrong directions.

**4. Backpropagation** — the real work. PyTorch walks backward through the computation graph and asks: for every number in the model, if I nudged it up slightly, would loss go up or down, and by how much? That's the gradient. For the 65×65 table, it computes 4225 gradients.

The intuition: if "H" appeared and the model put too little probability on "e" (the right answer), the gradient for the (H, e) cell says "increase this." The gradient for (H, z) says "decrease this."

**5. Update** — nudge every number a small step in the direction that reduces loss:

```
new_value = old_value - learning_rate × gradient
```

The learning rate controls step size. Too big → overshoots and never settles. Too small → learns painfully slowly.

After thousands of these cycles — grab random data, predict, measure, figure out which direction to nudge, nudge — the table converges to something that reflects real Shakespeare character patterns. That's all training is.

## B, T, C — Reading Tensor Dimensions

Every tensor in this model has three dimensions, always the same meaning:

- **B (Batch)** — how many independent sequences (e.g., 4)
- **T (Time)** — how many tokens in each sequence (e.g., 8)
- **C (Channels)** — how many numbers describe each token (e.g., 65)

A tensor shaped `(4, 8, 65)` reads: "4 sequences, each 8 tokens long, each token described by 65 numbers." Read left to right: which sequence → which position → what information.

## The Shape Journey: From Token IDs to Predictions

The whole model is a pipeline that transforms shapes. Tracing the shapes tells you exactly what's happening at each step.

### In the bigram model (no position info)

The token embedding table is `nn.Embedding(65, 65)` — a 65×65 grid. Each token ID looks up its row:

```
idx: (B, T) = (4, 8)           ← raw token IDs, just integers

         t=0  t=1  t=2  t=3  t=4  t=5  t=6  t=7
batch 0: [47,  57,  46,   1,  40,  50,  53,  53]    "ish bloo"
batch 1: [56,  39,  63,   5,  57,   1,  40,  56]    "ray's br"
batch 2: [ 1,  15,  47,  58,  47,  64,  43,  52]    " Citizen"
batch 3: [54,  43,   6,   1,  58,  46,  53,  59]    "pe, thou"
```

Each token looks up its row in the embedding table — token 47 gets row 47 (65 numbers):

```
logits = token_embedding_table(idx)     →  (4, 8, 65)
                                             B  T  C

"4 sequences, 8 positions each, 65 scores per position"
```

The problem: token 47 ("i") always produces the SAME 65 scores, no matter where it appears. "i" after "th" and "i" after "zz" → identical predictions. No position information.

### Adding positional embeddings: separating "what" from "where"

Two things change. First, the embedding dimension separates from vocab size — `nn.Embedding(65, 32)` maps each token to 32 numbers instead of 65. These 32 numbers aren't predictions anymore, they're a *description* of the token for downstream layers to process. Second, a new table for positions:

```python
token_embedding_table    = nn.Embedding(vocab_size, n_embd)   # (65, 32)
position_embedding_table = nn.Embedding(block_size, n_embd)   # (8, 32)
lm_head                  = nn.Linear(n_embd, vocab_size)      # (32 → 65)
```

The full flow:

```
idx: (B, T) = (4, 8)                      raw token IDs

    ↓ token embedding — "what am I?"
tok_emb: (4, 8, 32)                        each token → 32-number description

    ↓ + position embedding — "where am I?"
    pos_emb: (8, 32)                        each position → 32-number description
    Broadcasting: (4, 8, 32) + (8, 32)      same positions added to every batch

x: (4, 8, 32)                              each token now carries what + where

    ↓ lm_head linear layer — "what do I predict?"
    Multiplies (4, 8, 32) @ (32, 65)

logits: (4, 8, 65)                          65 prediction scores per position
    ↓ softmax
probs: (4, 8, 65)                           probabilities
    ↓ sample
next_token: (B, 1)                          one prediction per sequence
```

The key difference in action:

```
Bigram (no position):
  "i" at pos 0 → embed("i")              → [0.3, -1.2, ..., 0.5]
  "i" at pos 5 → embed("i")              → [0.3, -1.2, ..., 0.5]
                                              IDENTICAL — can't tell apart

With positional embedding:
  "i" at pos 0 → embed("i") + embed(pos 0) → [0.4, -0.9, ..., 0.3]
  "i" at pos 5 → embed("i") + embed(pos 5) → [-0.1, 0.7, ..., 1.2]
                                                DIFFERENT — model knows where it is
```

### Why addition, not concatenation?

You might think: stick them side by side `[token_32 | position_32]` to get 64 numbers. You could! But addition keeps the dimension the same (no extra parameters downstream) and works surprisingly well. The model learns token and position embeddings that live in the same space and combine meaningfully when added. Both approaches work; addition is simpler and is what the original transformer uses.

### Why the position table has exactly `block_size` rows

`nn.Embedding(block_size, n_embd)` — if block_size is 8, there are rows for positions 0 through 7. Position 8 doesn't exist in the table. This is part of why the context window is a hard limit: the model literally has no position embedding for tokens beyond it.

## From Averaging to Attention: Tokens Learning to Talk

In the bigram model, tokens are isolated. We want them to communicate — but with a hard rule: **no peeking at the future.** During generation, position 5 can't see positions 6, 7, 8 because those haven't been generated yet. During training we know the full text, but we have to enforce this same constraint so the model doesn't cheat.

### Why not treat all past tokens equally?

Consider predicting the next word in `"The cat sat on the ___"`. The words `"sat"` and `"on"` are very relevant. `"The"` at the start barely matters. Equal averaging blends them all the same, throwing away the signal that some tokens matter more than others. We want the model to learn which tokens to pay attention to.

### The lower triangular matrix: enforcing "no future"

Visualize which positions each token can see:

```
         pos0  pos1  pos2  pos3
pos0:    [YES   no    no    no ]   ← only sees itself
pos1:    [YES  YES    no    no ]   ← sees 0 and itself
pos2:    [YES  YES   YES    no ]   ← sees 0, 1, itself
pos3:    [YES  YES   YES   YES ]  ← sees everything before
```

That's a triangle — the lower part is YES, the upper part (the future) is NO.

### Why -inf? How softmax uses it

Softmax turns numbers into probabilities (positive, sum to 1). Bigger input → bigger probability. But what if you want a position to get **exactly** zero probability? Putting in 0 doesn't work — softmax(0) is still positive (e^0 = 1). But e^(-infinity) = 0. Exactly zero. Guaranteed.

So you put `-inf` in every future position. Softmax turns those into exactly 0 weight. No information leaks from the future, no matter what.

```
Position 2's scores:   [0.0,  0.0,  0.0,  -inf]
                        past  past  self   FUTURE

After softmax:         [0.33, 0.33, 0.33,  0.00]
                        ↑ equal weight on what we CAN see
```

### The leap: from equal weights to learned weights

Right now all the visible scores are 0.0, giving equal weights. But what if the model could *compute* different scores?

```
Equal scores:          [0.0,  0.0,  0.0,  -inf]
After softmax:         [0.33, 0.33, 0.33,  0.00]  ← boring equal average

Learned scores:        [0.1,  0.5,  2.0,  -inf]
After softmax:         [0.10, 0.15, 0.68,  0.00]  ← pays most attention to itself
```

Now position 2 pays 68% attention to itself, 15% to position 1, 10% to position 0 — and still exactly 0% to the future. Different situations produce different scores, so the model learns what to pay attention to for each prediction. That's self-attention.

### Three versions of the same trick: building toward real attention

The goal is always the same: for each token at position `t`, compute a weighted average of all tokens from position `0` to `t`. No future. Karpathy shows three ways to get there, each one closer to how attention actually works.

**Version 1: Explicit loops — the obvious way**

```python
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]          # all tokens up to t
        xbow[b, t] = xprev.mean(0)   # average them
```

Position 3 gets the average of positions 0, 1, 2, 3. Python loops over every batch and position — correct but painfully slow. The point is to state *what* we want to compute in the clearest possible way. Everything else is just doing this faster.

**Version 2: Matrix multiply with a triangular matrix — the fast way**

```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(dim=1, keepdim=True)
xbow = wei @ x   # (T, T) @ (B, T, C) → (B, T, C)
```

The weight matrix:

```
[[1.00, 0.00, 0.00, 0.00],   ← pos 0: just itself
 [0.50, 0.50, 0.00, 0.00],   ← pos 1: avg of 0,1
 [0.33, 0.33, 0.33, 0.00],   ← pos 2: avg of 0,1,2
 [0.25, 0.25, 0.25, 0.25]]   ← pos 3: avg of 0,1,2,3
```

No loops — one matrix multiply does all the averaging at once. The triangular shape enforces "no future" because zeros in the upper triangle mean future tokens contribute nothing. The key insight: **weighted averaging is just matrix multiplication.** Attention is "just" choosing what numbers go in that weight matrix.

**Version 3: Softmax with masked fill — how attention actually works**

```python
wei = torch.zeros(T, T)
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow = wei @ x
```

Same result as version 2, but now using softmax. Start with all zeros (raw scores), set future positions to `-inf`, then softmax turns `-inf` → exactly 0 and normalizes the rest into weights that sum to 1.

This is the actual mechanism in transformers. The only difference between this and real self-attention is where those initial zeros come from. In real attention, the zeros get replaced by **query-key dot products** — learned scores that say "how relevant is token j to token i?" The masked fill + softmax + matrix multiply pattern stays identical:

```
scores = query @ key.T / sqrt(d)   ← replaces the zeros
scores = scores.masked_fill(mask == 0, -inf)
weights = softmax(scores)
output = weights @ value
```

So these three versions are literally building attention piece by piece. Version 1 defines the goal, version 2 reveals the mechanism (it's a matrix multiply), and version 3 is the implementation that real transformers use — just with learned scores instead of fixed ones.

## Positional Embeddings: Teaching the Model "Where"

The token embedding table tells the model **what** each token is — but "e" at position 0 and "e" at position 7 produce the exact same numbers. The model has no concept of order. `"Hello"` and `"olleH"` would look identical.

The fix: a second lookup table, indexed by position instead of token.

```
Token embedding for "e":      [0.2, -0.5, 0.8, 0.1]   ← always the same
Position embedding for pos 3:  [0.1,  0.3, -0.2, 0.7]  ← unique to position 3
Position embedding for pos 6: [-0.4,  0.1,  0.5, -0.3] ← unique to position 6

"e" at pos 3 = token + position = [0.3, -0.2, 0.6, 0.8]
"e" at pos 6 = token + position = [-0.2, -0.4, 1.3, -0.2]
```

Different numbers. Now the model can tell them apart. Every token carries both "what I am" and "where I am" in a single vector.

The position embeddings are learned — they start random and get trained alongside everything else. The model figures out useful position representations on its own.

One limitation: the position table has exactly `block_size` rows. If block_size is 8, positions 0-7 are all the model knows. This is part of why the context window is a hard limit.

## Softmax: Why This Specific Function?

Softmax turns raw scores into probabilities. Sounds simple, but *why this formula and not something else?*

### The problem

You have raw numbers — logits — that could be anything: `-50, 0, 100, 3.2`. You need a probability distribution: all positive, sum to 1, bigger input → bigger output.

### Why not something simpler?

**Divide by the sum?** `x_i / sum(x)`:

```
scores: [3, -1, 2]  →  [3/4, -1/4, 2/4] = [0.75, -0.25, 0.50]
```

Negative probability. Meaningless.

**Absolute values?** `|x_i| / sum(|x|)`:

```
scores: [-5, 1, 5]  →  [5/11, 1/11, 5/11]
```

Now -5 and +5 get the same probability. The model says "definitely NOT this" and "definitely this" and you treat them equally.

**Squares?** `x² / sum(x²)`:

```
scores: [-5, 1, 5]  →  [25/51, 1/51, 25/51]
```

Same problem — sign is destroyed.

### Why e^x

`e^x` has exactly the properties you need:

```
x:    -5      0      1      5
e^x:   0.007  1.0    2.7    148.4
```

1. **Always positive.** e^(anything) > 0. Even e^(-1000) is tiny but positive, never zero, never negative.

2. **Preserves order.** Bigger input → bigger output. Always.

3. **Amplifies differences.** This is crucial. Gaps get stretched exponentially:

```
Inputs:   [1,    2,    3]        — gaps of 1
e^x:      [2.7,  7.4,  20.1]    — gaps of 4.7, then 12.7
softmax:  [0.09, 0.24, 0.67]    — biggest input dominates
```

A linear scheme would give [0.17, 0.33, 0.50]. Softmax gives [0.09, 0.24, 0.67]. The winner wins *more*. This is what lets attention be sharp — when one token is more relevant, it gets *much* more weight, not slightly more.

### The formula, slowly

```
softmax(x_i) = e^(x_i) / (e^(x_0) + e^(x_1) + ... + e^(x_n))
```

Make each thing positive with e^x, then divide by the total so they sum to 1:

```
scores:   [2.0,  1.0,  0.1]
e^each:   [7.39, 2.72, 1.11]       ← all positive
sum:       11.22
divide:   [0.66, 0.24, 0.10]       ← sums to 1.0
```

### Temperature: the sharpness knob

Divide all scores by T before softmax: `softmax(x / T)`:

```
scores: [2.0, 1.0, 0.1]

T = 1.0 (normal):    → [0.66, 0.24, 0.10]    standard
T = 0.5 (sharper):   → [0.87, 0.12, 0.01]    commits hard to the winner
T = 2.0 (smoother):  → [0.43, 0.26, 0.17]    hedges its bets
T → 0  (freezing):   → [1.0,  0.0,  0.0]     argmax — winner takes all
T → ∞  (boiling):    → [0.33, 0.33, 0.33]    uniform — everything equal
```

Low temperature = confident, sharp. High temperature = uncertain, exploratory. This is exactly what ChatGPT's temperature slider does. And it's why attention divides by `sqrt(d_k)` — dot product scores can get large, making softmax too sharp. Dividing calms it down.

### Why equal inputs give equal outputs (the averaging effect)

Back to the attention weight matrix. After masking, position 2 has:

```
scores: [0, 0, 0, -inf, -inf]
```

All non-masked scores are 0. e^0 = 1. Equal inputs → equal probabilities:

```
e^each:   [1, 1, 1, 0, 0]
sum:       3
divide:   [1/3, 1/3, 1/3, 0, 0]
```

Uniform weights = simple average. The averaging is just what softmax does when it has nothing to differentiate. The moment scores become unequal (when we add query-key dot products), the average becomes weighted, and the exponential amplification makes the weighting sharp.

### Numerical stability: the practical detail

With big numbers, e^1000 overflows — a number with 434 digits. The fix: subtract the maximum first. `softmax(x) = softmax(x - max(x))`, which is mathematically identical but keeps numbers small:

```
scores:        [1000, 1001, 999]
subtract max:  [-1,    0,   -2]
e^each:        [0.37,  1.0,  0.14]     ← no overflow
```

Every deep learning framework does this automatically. You'll never implement it yourself, but it explains an otherwise mysterious line of code in softmax implementations.

## Self-Attention: The Design Problem and Its Solution

### The problem you're solving

You have a sequence of tokens. Each token has a vector describing "what I am + where I am" (from the embeddings). But each token is still **isolated** — it knows about itself and nothing else. To predict the next token well, tokens need information from other tokens. The question is: **how should tokens communicate?**

### Why averaging wasn't good enough

The previous section showed equal-weight averaging: every past token contributes equally. But consider predicting the next word in `"The cat sat on the ___"`. The words `"sat"` and `"on"` carry the real signal. `"The"` barely matters. Equal averaging dilutes the useful tokens with irrelevant ones. We need a mechanism where each token can decide **how much to listen to each other token**, based on what's actually relevant.

### The core idea: data-dependent weights

In the averaging version, the weights were hardcoded — `[0.33, 0.33, 0.33]` regardless of the actual tokens. What we want instead: weights that **depend on the data itself**. When the tokens change, the weights change. The model should look at the actual tokens and compute: "given what I am and what you are, how relevant are you to me?"

The simplest way to measure "how relevant are you to me?" between two vectors: the **dot product**. If two vectors point in a similar direction, their dot product is high. If they're unrelated, it's near zero. So the idea is: let each token produce a vector, dot-product it against every other token's vector, and use those scores as attention weights.

### Why two separate projections instead of one?

You could dot-product each token's embedding with every other token's embedding directly. But that forces a single vector to serve two roles: "what makes me *findable* by others" and "what I *look for* in others." These are different things. A pronoun like "he" needs to be found by verbs (that need a subject), but "he" itself is looking for the noun it refers to. One vector can't point in both directions at once.

The solution: project each token into **two different vectors** through two different learned weight matrices.

- **Query (q):** a version of this token optimized for *searching* — "here's what I need"
- **Key (k):** a version of this token optimized for *being found* — "here's what I offer"

When a query and a key point in a similar direction (high dot product), that pair gets high attention. The model learns what "similar direction" means through training — we don't tell it to look for verbs or nouns. We give it two learnable matrices, and gradient descent figures out projections where useful token pairs end up with high dot products.

### Why a third projection (value)?

Now you know *who* to pay attention to (from q·k scores). But what information do you actually *extract?* The raw embedding carries everything about a token — its identity, position, all of it. But maybe you only need a specific aspect. A token like "not" might be *found* because it's a negation (that's its key), but what's *useful to extract* from it is different from what made it findable.

So there's a third projection — **value (v)** — a version of the token optimized for *being read*. The output of attention is a weighted sum of values, not of raw embeddings. This separation gives the model more flexibility: keys control who gets found, values control what information flows.

### Putting it together: the mechanism

Each token starts as a C-dimensional vector (e.g., 32 numbers). Three weight matrices project it into three different head_size-dimensional vectors (e.g., 16 numbers):

```python
q = Wq @ x    # each token's "search" vector
k = Wk @ x    # each token's "findability" vector
v = Wv @ x    # each token's "readable content" vector
```

`Wq`, `Wk`, `Wv` are just matrices of numbers — they start random, and backpropagation trains them. `nn.Linear(32, 16)` in PyTorch is just a `(32, 16)` weight matrix. Calling it on a vector is matrix multiplication, nothing more.

Compute relevance scores — every query dot-producted with every key:

```
scores = q @ k.T     → (T, T) matrix

              key_pos0  key_pos1  key_pos2
query_pos0: [  0.8      -0.3       0.1  ]    ← pos0 vs everyone
query_pos1: [  0.2       1.1      -0.4  ]    ← pos1 vs everyone
query_pos2: [  0.4       0.1       0.7  ]    ← pos2 vs everyone
```

Cell (2, 0) = 0.4 means position 2's query aligns well with position 0's key. Cell (1, 2) = -0.4 means position 1 doesn't find position 2 relevant.

Mask the future (same as before) and softmax to get weights:

```
After mask + softmax:
pos0: [1.00, 0.00, 0.00]    ← only sees itself
pos1: [0.29, 0.71, 0.00]    ← focuses more on pos1
pos2: [0.35, 0.26, 0.39]    ← unequal, data-dependent weights
```

Compare with the old fixed `[0.33, 0.33, 0.33]`. Now the weights are different for every position and change depending on the actual tokens. That's the whole point.

Weighted sum of values:

```
pos2's output = 0.35 × v[pos0] + 0.26 × v[pos1] + 0.39 × v[pos2]
```

Each token gets a blend of information from the tokens it found relevant.

### The design decisions, summarized

If you were building attention from scratch, these are the choices and why:

1. **How should tokens decide relevance?** → Dot product of learned projections. Simple, differentiable, parallelizable.
2. **Why two projections (q, k) not one?** → "What I search for" ≠ "what makes me findable." Separate projections let tokens play both roles.
3. **Why a third projection (v)?** → "How to find me" ≠ "what to extract from me." Decouples routing from content.
4. **How to block the future?** → Mask with -inf before softmax. Softmax turns -inf to exactly 0.
5. **How to normalize?** → Softmax. Turns arbitrary scores into weights that sum to 1, amplifies differences.

The actual code is remarkably short — three matrix multiplies, a dot product, mask, softmax, weighted sum. The insight is in understanding *why each piece is there* and *what would go wrong without it*.

### Why "self" attention?

Queries, keys, and values all come from the **same sequence**. Each token in the sequence decides what to attend to within its own sequence. In encoder-decoder models (like translation), "cross-attention" has queries from one sequence and keys/values from another — "what in the English sentence is relevant to the French word I'm generating?" Same mechanism, different source. Understanding self-attention means you understand cross-attention too — just change where q, k, v come from.

### Why this changed everything

Before transformers, sequence models (RNNs) processed tokens one at a time, passing information forward like a game of telephone. Token 100 learning about token 1 meant information had to survive 99 handoffs — it degraded, got diluted, got forgotten.

Attention creates a **direct connection** between any two tokens, regardless of distance. Token 100 can look straight at token 1 if their query-key score is high. No telephone, no degradation. `"The cat, which was sitting on the mat and purring loudly, was happy"` — "was" needs to agree with "cat" 12 tokens back. An RNN would struggle. Attention connects them in one step.

This also made training **parallelizable**. RNNs process tokens sequentially — you can't compute token 5 until token 4 is done. Attention computes all positions simultaneously (it's matrix multiplies). This is why transformers could scale to billions of parameters on GPUs while RNNs couldn't.

---

*More sections coming as I work through the video...*
