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

## Attention as a Communication Graph

Forget neural networks for a moment. Think of a **directed graph**: nodes connected by edges, where each edge has a direction and a weight.

### Tokens are nodes, attention is edges

Each token in the sequence is a node. The question "who can communicate with whom" is a question about graph structure — which edges exist. Batches never talk to each other — with batch_size=4 and block_size=8, you have 4 completely separate pools of 8 nodes. Communication only happens within each pool.

```
5 tokens = 5 nodes:  [The]  [cat]  [sat]  [on]  [the]
```

In causal (decoder) attention, edges only point backward. Each node receives information only from past nodes:

```
"The": ← (only itself)
"cat": ← "The"
"sat": ← "The", ← "cat"
"on":  ← "The", ← "cat", ← "sat"
"the": ← "The", ← "cat", ← "sat", ← "on"
```

The lower triangular mask IS this graph's adjacency matrix. 1 = edge exists, 0 = no edge. The `-inf` masking before softmax is how you enforce this topology — zeroing out connections that shouldn't exist.

### Edge weights = attention scores

The graph structure says *who can talk*. The attention weights say *how much to listen*. After q·k + softmax, each edge gets a weight:

```
"on" receives:
  0.10 × v["The"]  ──→ ┐
  0.45 × v["cat"]  ──→ ├──→  weighted sum  ──→  "on"'s new representation
  0.15 × v["sat"]  ──→ ┤
  0.30 × v["on"]   ──→ ┘
```

Each node collects **messages** from its neighbors (value vectors, scaled by attention weight), sums them, and that becomes its updated representation. After attention, "on" isn't just "on" anymore — it carries information blended from "The" (10%), "cat" (45%), "sat" (15%), and itself (30%).

The weights are **data-dependent** — different input tokens produce different attention patterns, different edge weights. The graph shape stays the same (triangular), but the strength of every edge changes based on the actual content.

### Why this framing matters: you can invent variations by changing the graph

Once you see attention this way, every variant is just a different answer to three questions:

**1. What's the graph shape?** (who can talk to whom)
- **Causal (GPT):** triangular — each node sees only the past
- **Full (BERT):** fully connected — every node sees every other. No mask. Used when you have the whole input upfront (classification, not generation)
- **Sparse (Longformer):** each node sees a local window (e.g., 128 nearest tokens) plus a few global tokens. Reduces O(n²) to O(n)
- **Cross-attention:** queries from sequence A, keys/values from sequence B. Nodes in A have edges to nodes in B. Used in translation, image captioning

**2. How are edge weights computed?** (how relevant is each connection)
- **Dot-product:** q·k — standard transformer
- **Relative position:** weight depends on distance between nodes, not absolute position
- **Linear attention:** approximate softmax to avoid O(n²)

**3. What flows along edges?** (what gets communicated)
- **Values:** linear projection of input — standard
- **Gated values:** multiply by a learned gate that amplifies or suppresses parts of the message

The transformer is one specific answer to these three questions. But the framework is general — and most attention research is exploring different answers to one of these three.

### Attention has no notion of space or position

The dot product `q[i] · k[j]` only looks at the **content** of two vectors. It doesn't know that position 2 and position 0 are 2 apart, or that one comes before the other. If you scrambled the token order, the dot products would be exactly the same — just rearranged in the attention matrix. Attention treats tokens like **items in a bag**, not elements in a sequence.

This is a problem because word order matters — `"The dog bit the man"` and `"The man bit the dog"` are the same bag of words with very different meanings. Without position information, attention computes identical scores for both because the same tokens produce the same queries and keys.

This is why positional embeddings exist. By adding position information to the token embedding *before* computing Q and K, position gets baked into the queries and keys, making the dot products position-aware. But this is something we had to **add** — attention didn't come with it.

And this is exactly what makes attention so general. It's a **content-based communication mechanism over a set of vectors** that assumes nothing about what those vectors represent. Text, image patches, molecular atoms, chess moves — attention doesn't care. You add structure by choosing how to encode position: sequential for text, 2D grid for images, 3D for video, graph coordinates for molecules. Attention just operates on whatever vectors you give it. This is why the same architecture works across domains that look nothing alike.

### How attention differs from convolution

Convolution is also a communication mechanism — but with a **fixed-size sliding window** (e.g., 3 tokens wide), the **same learned weights** at every position, and **built-in spatial awareness** (weight 0 always means "left neighbor"). It's a fixed local graph with static edges.

Attention is the opposite: **global** (any token can attend to any other), **data-dependent** (weights change based on content), and **position-unaware** (must be added). Convolution excels where local patterns matter (images — a cat's ear is always near its head). Attention excels where dependencies are long-range and unpredictable (language — subject and verb can be 20 tokens apart). Same message-passing framework, different design choices.

### Encoder vs decoder: when to mask

The causal mask exists to prevent cheating — seeing future tokens that won't exist at inference time. Whether you need it depends on one question: **are you generating output one step at a time?**

**Decoder (causal mask):** the model generates sequentially — predict the next token/item given everything so far. Text generation, music generation, "what will this user watch next?" (SASRec). The future doesn't exist yet, so the model must not see it during training either. Triangle graph.

**Encoder (no mask):** the model receives the complete input and produces a single output — a classification, an embedding, a score. Sentiment analysis, user taste modeling from full history (BERT4Rec), item similarity. Every element should see every other because the full input exists upfront. Fully connected graph.

The mask isn't about "language vs not language." It's about "generating sequentially vs understanding a complete input." When designing an architecture, ask: does step N depend on step N+1 existing? If yes, mask. If no, don't.

**Cross-attention** is a third option — for when two different sequences need to interact. Queries come from sequence A, keys and values from sequence B. Translation: the French decoder queries against the English encoder. Image captioning: caption words query against image patches. Recommendations with context: user actions query against item descriptions. The full original transformer uses all three: encoder (self-attention, no mask) processes the source, decoder (self-attention, causal mask) generates the output, and cross-attention (in the decoder) lets each generated token look at the full source. GPT's key insight was dropping the encoder and cross-attention entirely — just a stack of decoder blocks. It works because the "source" and "target" are the same text stream: predict the next token given everything before it. No separate input to encode, no two sequences to bridge. This simplification made the architecture dramatically simpler to scale.

## Scaled Attention: Why Divide by sqrt(d_k)

A dot product is a sum of element-wise multiplications. More dimensions = more terms = larger magnitude. With head_size=64, the dot product of two random vectors has variance ~64, meaning scores land around ±8 instead of ±1. Large scores push softmax into saturation — one token gets ~100% weight, everything else gets ~0%. Two things break: the model can't blend information from multiple tokens (attention becomes a hard lookup), and gradients vanish (softmax at saturation is nearly flat, so backpropagation can't figure out which direction to nudge).

The fix: `scores = q @ k.T / sqrt(head_size)`. Dividing by sqrt(64) = 8 exactly counteracts the variance growth, keeping scores around ±1 regardless of dimension. It's the same idea as temperature — setting it so softmax stays in its useful range where weights are smooth and gradients flow. Without scaling, the model would still work mathematically, but training would be slow and unstable.

## The Attention Formula: Reading It After Understanding It

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

After everything above, every piece of this formula should feel familiar. Reading right to left, inside out:

1. **Q·Kᵀ** — every query dot-producted with every key. Produces the (T, T) score matrix: "how relevant is each token to each other token?" This is the data-dependent edge weight computation in the graph.

2. **/ √d_k** — scale the scores down by the square root of the key dimension. Without this, larger dimensions produce larger dot products, which push softmax into saturation. This keeps gradients healthy regardless of head size.

3. **softmax(...)** — turn raw scores into weights that are positive and sum to 1. The causal mask (−∞ on future positions) is applied before this step, so future tokens get exactly zero weight. This is where the exponential amplification happens — high scores get disproportionately more weight.

4. **· V** — weighted sum of values using those weights. Each token's output is a blend of value vectors from the tokens it found relevant. This is the message-passing step — information flows along the weighted edges of the graph.

The whole formula in one sentence: compute relevance between all pairs of tokens, normalize into weights, and use those weights to blend information. Everything else — the mask, the three projections, the scaling — is engineering to make this work reliably at scale.

## Why LLMs Lose the Middle: Self-Attention at Scale

Everything above uses `block_size=8`. Attention works beautifully at that scale. But real models have context windows of 100k+ tokens — and that's where a well-known problem emerges. LLMs pay the most attention to the **beginning** and **end** of the context, and struggle with information buried in the **middle**. The mechanism we just learned explains why.

### Softmax is a competition, and the middle loses

Each row of the attention matrix sums to 1.0 — softmax enforces this. With 8 tokens, the weight is split among 8 positions. Plenty to go around.

Scale to 100k tokens and that same 1.0 is split across **100,000 positions**. Softmax's exponential amplification — the thing that makes attention sharp — now works against you. A few high-scoring keys grab most of the weight, and everything else gets squeezed toward zero. A moderately relevant token in the middle that would've gotten 15% weight in an 8-token window gets 0.001% in a 100k-token window. It's not ignored on purpose — it's **drowned out** by the competition.

### Why the end of context gets high attention: positional decay

Karpathy's video uses a learned position lookup table. Modern models use **rotary position embeddings (RoPE)**, which have a built-in property: the Q·K dot product **decays with distance**. The further apart two tokens are, the lower their attention score, all else being equal.

The token being generated is always at the end. So tokens near the end — recent context — get a natural positional boost. That's the right side of the U-shape.

### Why the beginning gets high attention: learned bias, not math

The positional decay explains the end, but not the beginning. The beginning is far from the current token too — so why does it get special treatment?

**Training.** During pretraining, the model sees billions of examples where the first tokens carry structural information — document titles, topic sentences, headers, system prompts. The model learns: "the beginning usually tells me what this whole thing is about." This isn't from the attention formula — it's a **learned bias** baked into the Q, K, V weight matrices through gradient descent.

### Two different effects making one U-shape

```
BEGINNING: high attention  ←  learned bias ("start = important structure")
MIDDLE:    low attention   ←  far from current token + nothing special learned
END:       high attention  ←  positional proximity (RoPE decay)
```

### Can the model attend to the middle?

Yes — mechanically, nothing prevents it. If a key in the middle produces a huge Q·K dot product, softmax will give it high weight. And sometimes it does. The U-shape is a **statistical tendency across many inputs**, not a hard rule for every single attention computation.

But the model has to *learn* to look at the middle, and training gives it weak signal to do so. Most training documents are shorter than the max context window. When the model does see long contexts during training, the critical information usually sits near the beginning (document structure) or near the end (recent context). The middle of a 100k-token context is rarely where the answer lives in training data. So the Q and K weight matrices never get strongly trained to produce high scores for "middle of a very long context."

The attention formula doesn't create the U-shape. **Positional decay** explains the right side. **Training distribution** explains the left side. The middle loses not because the mechanism blocks it, but because nothing pushes the learned weights to favor it.

### The practical consequence

This connects directly to the context window discussion. In a long conversation with an AI coding agent:

- **System prompt** sits at the beginning → high attention (good)
- **Your latest message** sits at the end → high attention (good)
- **Important instructions from 30 messages ago** → buried in the middle → low attention (bad)

The fix isn't a bigger context window — it's keeping context lean, or re-stating important information so it moves to the end where attention is strong. The same self-attention mechanism that makes transformers powerful at short range creates this blind spot at long range.

## Multi-Head Attention: Why One Head Isn't Enough

A single attention head computes one set of Q, K, V projections. That means it learns one notion of "relevant" — maybe it learns that verbs attend to their subjects. But language has many simultaneous relationships. A word like "it" needs to figure out: what noun does it refer to? What adjective described that noun? What action is being done to it? One attention pattern can't capture all of these at once.

**Multi-head attention** runs multiple heads in parallel, each with a smaller dimension, then concatenates the results:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
```

### The math stays the same size

With `n_embd=32`:

```
Single head:  1 head × 32 dims = 32-dim output
Multi head:   4 heads × 8 dims = 32-dim output  (same!)
```

Each head is smaller individually, but collectively they cover more ground because they attend to different things. `torch.cat` glues them back into the same 32-dim vector. The downstream layers see the same shape — they don't know or care how many heads produced it.

### What different heads learn

Each head develops its own Q, K, V matrices through training. Since they're initialized randomly and updated independently, they naturally specialize:

- **Head 1** might learn syntactic relationships — subjects attending to verbs
- **Head 2** might learn positional patterns — nearby tokens attending to each other
- **Head 3** might learn semantic grouping — related concepts attending across distance
- **Head 4** might learn structural patterns — punctuation attending to clause boundaries

We don't tell the heads what to specialize in. Gradient descent discovers useful patterns because different attention patterns reduce loss in different ways. The diversity comes from random initialization + the pressure to collectively explain the data.

### Why not just make one big head?

A single head with dimension 32 computes one (T, T) attention matrix — one set of weights for who talks to whom. Four heads with dimension 8 compute four different (T, T) attention matrices. More heads = more distinct communication patterns per layer.

The tradeoff: each head has fewer dimensions to work with (8 instead of 32), so each individual head captures less nuance. But the ensemble of specialized heads outperforms one generalist head. In the notebook, multi-head attention (4 heads × 8 dims) drops val loss from ~2.40 (single-head) to ~2.28 — a meaningful improvement from the same parameter count.

## FeedForward: Letting Tokens Think

Attention lets tokens **communicate** — gather information from other tokens. But after gathering, each token needs to **process** what it collected. That's the feedforward layer.

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
```

It's remarkably simple: a linear layer followed by ReLU. But it serves a critical role.

### Attention = communication, feedforward = computation

Think of it as a two-phase process:

```
x → MultiHeadAttention → FeedForward → output
    (tokens talk)        (tokens think)
```

After multi-head attention, each token's vector is a blend of information from other tokens. The feedforward layer lets each token independently process that blended information — transform it, extract patterns from it, prepare it for the next layer's attention.

Crucially, feedforward operates **per-token independently**. Position 3's feedforward computation doesn't see position 5's. All the cross-token communication happened in attention. Feedforward is purely "given what I've gathered, what do I make of it?"

### Why ReLU matters

Without the non-linearity, stacking attention + linear layer + attention + linear layer would collapse into something equivalent to one big attention + one linear layer. Linear operations compose into linear operations — you'd get no benefit from depth.

ReLU (`max(0, x)`) breaks linearity. It zeros out negative values and passes positive values through. Simple, but it means the combination of layers can learn patterns that no single layer could represent. This is what makes deep networks "deep" in a meaningful sense — each layer builds non-linear features on top of the previous layer's non-linear features.

### The result

Adding feedforward after multi-head attention drops val loss from ~2.28 to ~2.24 in the notebook. A small improvement — but this is the fundamental building block. In a real transformer, this attention + feedforward pair gets repeated many times (GPT-3 has 96 layers), and the feedforward layer typically has 4× the hidden dimension (e.g., n_embd=768, feedforward inner dim=3072). At scale, the feedforward layers contain the majority of the model's parameters and are where much of the "knowledge" is stored.

### The transformer block

Attention + feedforward together form one **transformer block** — the repeating unit of every transformer:

```
Input
  ↓
  Multi-Head Attention   ← tokens communicate
  ↓
  FeedForward            ← tokens compute
  ↓
Output
```

Stack N of these blocks and you get a transformer. GPT-2 stacks 12, GPT-3 stacks 96. Each block refines the representation — early blocks might capture simple patterns (common character pairs), middle blocks might capture grammar, later blocks might capture meaning. The depth is what gives transformers their power.

## Watching the Loss Drop: What Each Technique Actually Buys You

Theory is one thing. Let's look at what actually happens when you train a character-level Shakespeare model (65-character vocabulary, block_size=8, n_embd=32) and stack these techniques one at a time. Each model trains for 5,000 steps on the same data. The number to watch is **validation loss** — how well the model predicts characters it's never trained on.

**The baseline: random guessing**

Before any training, the model spreads probability equally across all 65 characters. Each character gets 1/65 chance. Loss = -log(1/65) ≈ **4.17**. This is the "I know nothing" starting point. Any model that trains at all should beat this immediately.

### Step 1: Bigram model → val loss ~2.58

The simplest possible model. A 65×65 lookup table — given the current character, what's the probability of each next character? No context beyond the single previous character.

```
"H" → probably "e" or "a" or "i" (common after H)
"q" → almost certainly "u"
" " → could be anything (space precedes many characters)
```

Loss drops from 4.17 to **~2.58**. That's a huge jump from random, but the ceiling is low. The model knows that "q" is followed by "u" and "t" is often followed by "h", but it can't learn "the" as a unit because it only sees one character back. It generates text like:

```
MADOY'
'tr thSStlleel, noisuan os
```

Recognizable character pairs, but no words, no structure.

**What it learned:** character-pair frequencies. "t→h" is common, "z→z" is rare.
**What it can't learn:** anything requiring more than one character of context.

### Step 2: Single-head self-attention → val loss ~2.40

Now each token can look at all previous tokens (up to block_size=8) and decide how much to attend to each one. The model also gets positional embeddings — it knows *where* each token is, not just *what* it is.

Loss drops from 2.58 to **~2.40**. The improvement is immediate. With 8 characters of context, the model can start learning short word patterns and common sequences.

```
An
Pur veay quy woungca heane I poay th poudt;
Manikill, UNO wicke ale ln wat byovey
```

You can see word-like structures emerging — "the", "heane" (almost "heard"), "wicke" (almost "wicked"). The model is learning that certain *sequences* of characters go together, not just pairs.

**What it learned:** which past tokens matter for the current prediction. In "th", the "t" strongly attends to itself and the "h" attends back to "t" — learning that this pair behaves as a unit.
**What it can't learn:** multiple types of relationships simultaneously. The single attention head has one pattern — it can't track syntax AND semantics at the same time.

### Step 3: Multi-head attention (4 heads) → val loss ~2.28

Same total dimension (32), split into 4 independent heads of 8 dimensions each. Each head develops its own attention pattern.

Loss drops from 2.40 to **~2.28**. The model can now track multiple relationships in parallel — one head might focus on nearby characters, another on recurring patterns further back.

```
QARENSLOCOFOR:
Ta.
Martin,
Ox Jouatephiten!
Cius! Balland o bin a and wor'd
```

More structure. "Martin" is a real name. "and wor'd" almost looks like Shakespeare. The colon-after-name pattern (speaker labels in the play) is emerging. Multiple heads let the model simultaneously notice "this looks like a name" and "a colon usually follows."

**What it learned:** multiple simultaneous attention patterns. Different heads specialize — maybe one learns character adjacency, another learns word boundaries, another learns the speaker-label pattern.
**What it can't learn:** non-linear combinations of the information it gathers. Attention is linear — it's weighted averaging. The model can blend information from multiple tokens, but can't do complex transformations on that blend.

### Step 4: Add feedforward → val loss ~2.24

After attention gathers information, a feedforward layer (linear + ReLU) lets each token process what it collected.

Loss drops from 2.28 to **~2.24**. A smaller absolute improvement, but this is the piece that unlocks depth — without non-linearity, stacking more layers wouldn't help.

**What it learned:** non-linear patterns in the attended information. After attention blends nearby characters together, feedforward can detect patterns in that blend that a linear operation would miss.

### The full progression

```
Random guessing (no training)          4.17
                                        ↓  -1.59  (learning character pairs)
Bigram (lookup table, no context)      2.58
                                        ↓  -0.18  (8 tokens of context, learned weights)
+ Single-head attention                2.40
                                        ↓  -0.12  (4 parallel attention patterns)
+ Multi-head attention                 2.28
                                        ↓  -0.04  (non-linear processing)
+ FeedForward                          2.24
```

### What the numbers tell you

**The biggest single win is the dumbest technique.** Going from random to a lookup table (bigram) drops loss by 1.59. That's just memorizing which characters follow which — no intelligence, no context, pure statistics. Most of the "easy" prediction in language comes from local patterns. If you see "q", you already know the next letter.

**Context is the second biggest win.** Adding attention (0.18 drop) gives the model 8 characters of memory instead of 1. It can now learn "the" and "ing" and "tion" as patterns, not just character pairs. The more context you give a model, the better it predicts — this is why real models push context windows to 100k+ tokens.

**Parallelism helps more than non-linearity at this scale.** Multi-head (0.12 drop) beats feedforward (0.04 drop). Having multiple attention patterns — multiple ways to look at the same context — matters more than having non-linear processing, at least for this tiny model. At scale, feedforward becomes much more important because it's where the model stores factual knowledge.

**Diminishing returns at each step — but they compound.** Each technique adds less than the previous one. But this is a toy model with 32 embedding dimensions and 8 tokens of context. Real transformers stack 96 attention+feedforward blocks with 12,288-dimensional embeddings and 100k tokens of context. The same techniques that gave us 4.17 → 2.24 on character-level Shakespeare give us GPT-4 on the full internet. The architecture is identical — just scaled up.

### What the generated text tells you

The loss numbers are abstract. The generated text makes the improvement visceral:

**Bigram (2.58):** `MADOY' 'tr thSStlleel, noisuan os` — character soup with occasional recognizable pairs.

**Single-head (2.40):** `Pur veay quy woungca heane I poay th poudt` — word-shaped blobs. You can almost read it.

**Multi-head (2.28):** `Martin, Ox Jouatephiten! Cius! Balland o bin a and wor'd` — real names appear, punctuation in roughly the right places, "and" used correctly.

**Multi-head + feedforward (2.24):** Similar quality to multi-head at this scale, but the foundation is now set for stacking — this attention+feedforward block is the repeating unit that, when stacked 12–96 times with larger dimensions, produces coherent paragraphs.

Each technique doesn't just lower a number — it unlocks a qualitatively different *kind* of pattern the model can learn. Bigram learns pairs. Attention learns sequences. Multi-head learns simultaneous relationships. Feedforward learns non-linear transformations. Stack them and scale them up, and you get language.

## Breadth vs Depth: Why Transformers Need Both Multi-Head AND Stacked Blocks

At this point you might wonder: we already have multi-head attention running 4 heads in parallel — why do we also need to stack blocks? Aren't both just "doing more"? They are, but in fundamentally different ways.

### Multi-head = multiple perspectives on the same input

Imagine reading the sentence `"The doctor said she would recover"`. A single attention head can compute one set of attention weights — one pattern of "who pays attention to whom." Maybe it learns that `"she"` attends strongly to `"doctor"` (coreference). Great. But that same head can't *simultaneously* learn that `"recover"` attends to `"said"` (verb structure) and that `"would"` attends to `"recover"` (tense agreement). One head, one attention pattern.

Multi-head attention fixes this by running multiple heads in parallel, each with its own Q, K, V:

```
                    "The doctor said she would recover"
                      ↓     ↓      ↓    ↓     ↓     ↓
              ┌───────┬─────┬──────┬────┬─────┬──────┐
              │       same input goes to ALL heads     │
              └───┬───┴──┬──┴───┬──┴────┴─────┴──────┘
                  ↓      ↓      ↓      ↓
            ┌─────────┬─────────┬─────────┬─────────┐
            │ Head 1  │ Head 2  │ Head 3  │ Head 4  │
            │         │         │         │         │
            │ "she"→  │ "would" │ "said"→ │ nearby  │
            │ "doctor"│ →"recov"│ "doctor"│ tokens   │
            │         │         │         │ attend   │
            │ learns: │ learns: │ learns: │ learns:  │
            │ pronouns│ tense   │ who said│ local    │
            │ refer to│ agreemnt│ what    │ context  │
            └────┬────┴────┬────┴────┬────┴────┬────┘
                 ↓         ↓         ↓         ↓
            ┌──────────────────────────────────────┐
            │     concat → 4 perspectives fused     │
            └──────────────────────────────────────┘
```

Each head sees the **exact same tokens** but develops different Q and K matrices, so different pairs light up. Head 1 might learn that pronouns attend to nouns. Head 2 might learn tense relationships. They all operate on the raw input, in parallel, then their outputs get concatenated.

This is **breadth** — more perspectives at the same level of understanding.

### Stacked blocks = progressively deeper understanding

Now consider what happens with a single block vs three stacked blocks processing `"The doctor said she would recover"`:

```
                    "The doctor said she would recover"
                                    ↓
                ┌─────────────────────────────────────────┐
                │              BLOCK 1                     │
                │                                         │
                │  Attention: each token looks at raw      │
                │  tokens and learns direct relationships  │
                │                                         │
                │  "she" ← attends to "doctor"            │
                │  "would" ← attends to "recover"         │
                │  "said" ← attends to "doctor"           │
                │                                         │
                │  FeedForward: process these pairs        │
                │                                         │
                │  After Block 1, "she" now carries info   │
                │  about "doctor". "would" carries info    │
                │  about "recover".                        │
                │                                         │
                │  Understands: word pairs, local grammar  │
                └──────────────────┬──────────────────────┘
                                   ↓
                   Block 1's OUTPUT becomes Block 2's INPUT
                      (this is NOT the raw tokens anymore)
                                   ↓
                ┌─────────────────────────────────────────┐
                │              BLOCK 2                     │
                │                                         │
                │  Attention: each token looks at Block 1  │
                │  output — tokens that already absorbed   │
                │  their neighbors                         │
                │                                         │
                │  "would" attends to "she"                │
                │  ...but "she" already contains "doctor"  │
                │  info from Block 1!                      │
                │                                         │
                │  So "would" now effectively knows about  │
                │  "she" + "doctor" — two hops away —      │
                │  even from a single attention step       │
                │                                         │
                │  Understands: phrases, clause structure   │
                └──────────────────┬──────────────────────┘
                                   ↓
                   Block 2's OUTPUT becomes Block 3's INPUT
                                   ↓
                ┌─────────────────────────────────────────┐
                │              BLOCK 3                     │
                │                                         │
                │  Attention: tokens now carry phrase-level │
                │  information from Blocks 1 and 2         │
                │                                         │
                │  "recover" can now piece together:       │
                │  "a doctor said [someone] would" — this  │
                │  is a medical prediction context         │
                │                                         │
                │  Understands: full sentence meaning,     │
                │  can predict what comes next             │
                └──────────────────┬──────────────────────┘
                                   ↓
                              predict next token
```

The critical thing: **Block 2 does NOT see raw tokens.** It sees tokens that have *already been transformed* by Block 1. After Block 1, the token `"she"` isn't just `"she"` anymore — it's `"she (who refers to the doctor)"`. When Block 2's attention connects `"would"` to this enriched `"she"`, it's effectively learning a pattern about the *doctor*, even though it never directly attended to `"doctor"` itself.

This is **depth** — each block builds on the previous block's understanding, creating progressively more abstract representations.

### Why you need both

Consider what each one alone can't do:

**Multi-head only (wide but shallow):** you have 12 heads, all looking at raw tokens. Head 1 notices `"she"→"doctor"`. Head 7 notices `"would"→"recover"`. But no head can combine these two observations — that the doctor's patient (she) would recover — because each head independently processes the same raw input. They can't build on each other's findings.

**Single-head stacked (deep but narrow):** Block 1 notices one relationship per position. Block 2 builds on it. But at each level, the model only captures one attention pattern. It might learn `"she"→"doctor"` in Block 1, but then Block 2 can only attend one way — maybe it catches tense but misses the clause boundary. One pattern per level limits what gets carried forward.

**Both together:** each block has multiple heads finding different patterns, then the next block has multiple heads finding patterns *in those patterns*. It's like having a team of analysts at each level, where each level reads the previous team's combined report.

```
Multi-head only:          Stacked only:           Both:

  [raw tokens]             [raw tokens]            [raw tokens]
       ↓                        ↓                       ↓
  ┌──┬──┬──┬──┐           ┌──────────┐            ┌──┬──┬──┬──┐
  │H1│H2│H3│H4│           │ 1 head   │            │H1│H2│H3│H4│ Block 1
  └──┴──┴──┴──┘           └────┬─────┘            └──┴──┴──┴──┘
       ↓                       ↓                       ↓
   [output]               ┌──────────┐            ┌──┬──┬──┬──┐
                          │ 1 head   │            │H1│H2│H3│H4│ Block 2
   4 perspectives         └────┬─────┘            └──┴──┴──┴──┘
   on raw input                ↓                       ↓
                          ┌──────────┐            ┌──┬──┬──┬──┐
   can't combine          │ 1 head   │            │H1│H2│H3│H4│ Block 3
   findings across        └────┬─────┘            └──┴──┴──┴──┘
   heads                       ↓                       ↓
                           [output]                [output]

                          3 levels deep,          4 perspectives
                          but only 1 pattern      × 3 levels deep
                          per level               = rich understanding
```

### This is the actual architecture of every GPT model

GPT-2 Small: 12 blocks × 12 heads per block = 144 simultaneous attention patterns across 12 levels of abstraction.

GPT-3: 96 blocks × 96 heads per block = 9,216 attention patterns across 96 levels.

Research on what these layers learn shows exactly the pattern you'd expect:
- **Early blocks** (1–3): character patterns, common word pieces, punctuation rules
- **Middle blocks** (4–8): syntax, grammar, subject-verb agreement, clause boundaries
- **Late blocks** (9–12): semantics, factual recall, reasoning patterns, style

Each block asks "given what the previous blocks figured out, what higher-level pattern can I find?" Multi-head gives breadth at each level. Stacking gives depth across levels. Together, they turn raw token IDs into understanding.

## How Backpropagation Works Through Stacked Blocks

Before understanding residual connections, you need to understand the problem they solve. And that means understanding how training actually flows through multiple blocks.

### Forward: data flows down

Training has two phases. The **forward pass** pushes data through the model to make a prediction. With 3 stacked blocks:

```
"The doctor said she would ___"

  ↓ embeddings

  x₀ = [0.2, -0.5, 0.8, ...]       ← raw token + position embeddings

  ↓ Block 1 (has weights W₁)

  x₁ = Block1(x₀)                   ← x₁ depends on x₀ and W₁

  ↓ Block 2 (has weights W₂)

  x₂ = Block2(x₁)                   ← x₂ depends on x₁ and W₂

  ↓ Block 3 (has weights W₃)

  x₃ = Block3(x₂)                   ← x₃ depends on x₂ and W₃

  ↓ lm_head

  prediction: "sing"
  correct answer: "recover"
  loss = 4.0                         ← how wrong we are
```

Each block transforms the data using its own internal weights (Q, K, V matrices, feedforward weights). The output of one block becomes the input of the next. At the end, we get a prediction and a loss — a single number measuring how wrong we were.

### Backward: gradients flow up

The **backward pass** works in reverse. Starting from the loss, it asks every weight in the model: "if I nudged you slightly, would the loss go up or down?" The answer for each weight is its **gradient** — a number saying which direction to move and by how much.

To compute each weight's gradient, backpropagation uses the **chain rule**: trace the path from that weight to the loss, and multiply the derivatives along the way.

Let's make it concrete with a toy example. Forget attention — just two blocks, each with one weight, doing simple multiplication:

```
Forward:
  x = 2.0
  Block 1:  a = x × W₁ = 2.0 × 0.5 = 1.0
  Block 2:  b = a × W₂ = 1.0 × 3.0 = 3.0
  loss = (target - b)² = (5.0 - 3.0)² = 4.0
```

Now backward. Each weight needs its own gradient:

```
Backward:

  How does W₂ affect the loss?
  Path: W₂ → b → loss
  W₂'s gradient = ∂loss/∂b × ∂b/∂W₂
                = -2×(5.0-3.0) × a
                = -4.0 × 1.0 = -4.0
  → "increase W₂, loss decreases by 4.0 per unit"

  How does W₁ affect the loss?
  Path: W₁ → a → b → loss     (longer path — must go through W₂!)
  W₁'s gradient = ∂loss/∂b × ∂b/∂a × ∂a/∂W₁
                = -4.0 × W₂ × x
                = -4.0 × 3.0 × 2.0 = -24.0
  → "increase W₁, loss decreases by 24.0 per unit"
```

The key thing: **W₁'s gradient must pass through W₂**. That `× W₂` in the chain is unavoidable — it's how calculus works. The gradient for an early weight always gets multiplied by the weights of every block between it and the loss.

### Why depth kills the gradient

In the toy example, W₂ = 3.0, so passing through it actually *amplified* the gradient. But in real networks, weights are typically small numbers (around 0.1–0.8 after initialization). When the gradient passes through a block with small weights, it shrinks:

```
3 blocks with small weights:
  W₁'s gradient = loss_gradient × W₃ × W₂
                = 1.0 × 0.6 × 0.6 = 0.36

10 blocks:
  W₁'s gradient = loss_gradient × W₁₀ × W₉ × W₈ × ... × W₂
                = 1.0 × 0.6⁹ = 0.01

96 blocks:
  W₁'s gradient = 1.0 × 0.6⁹⁵ ≈ 0.0000000000000000001
```

Block 1's weights get a gradient so small that the update is effectively zero:

```
weight update = learning_rate × gradient
             = 0.001 × 0.0000000000000000001
             = nothing

Block 1 stops learning. It's dead weight.
```

This is the **vanishing gradient problem**. The deeper the network, the less the early blocks learn, because the gradient must pass through every intermediate block's weights and gets shrunk at each one. Past ~5 blocks, the early blocks barely train at all.

## Residual Connections: The Fix

The fix is exactly two characters: `+`.

```python
# Without residual — each block REPLACES the input
def forward(self, x):
    x = self.attention(x)
    x = self.feedforward(x)
    return x

# With residual — each block ADDS TO the input
def forward(self, x):
    x = x + self.attention(x)
    x = x + self.feedforward(x)
    return x
```

### How `+` changes the backward pass

Remember, the chain rule says: to get W₁'s gradient, trace the path from W₁ to the loss and multiply derivatives along the way. Without residual, there's only one path — through every block's weights.

With `output = input + block(input)`, there are now **two paths** at every block:

```
Forward with residual:
  x ─────────────────┐
  │                   │
  ↓                   ↓
  block(x)    ──→    (+)  ──→  output
                      ↑
  "through block"   "skip"
```

Backward at this `+`:

```
  A gradient of 0.5 arrives at the (+)

  Addition's derivative is 1 with respect to BOTH inputs.
  (If a + b = c, then ∂c/∂a = 1 and ∂c/∂b = 1)

  So the gradient splits into two identical copies:

  0.5 → skip path     (goes directly to x, multiplied by nothing)
  0.5 → block path    (goes through block weights, might shrink to 0.3)
```

The skip path delivers the gradient **untouched**. No multiplication by block weights. The gradient just passes straight through the `+`.

### Tracing through 3 blocks

```
Forward:
  x₀ → [x₀ + Block1(x₀)] = x₁ → [x₁ + Block2(x₁)] = x₂ → [x₂ + Block3(x₂)] = x₃

Backward (gradient g arrives at x₃):

  At Block 3's (+):
    skip path:  g goes to x₂ (untouched)
    block path: g × W₃ goes to x₂ (shrunk)
    x₂ receives: g + g×W₃

  At Block 2's (+):
    skip path:  g goes to x₁ (untouched — the g from Block 3's skip)
    block path: g × W₂ goes to x₁ (shrunk)
    x₁ receives: g + stuff

  At Block 1's (+):
    skip path:  g goes to x₀ (STILL untouched)
    block path: g × W₁ goes to x₀ (shrunk)
    x₀ receives: g + stuff
```

The skip path at each `+` copies `g` backward without multiplying by anything. So `g` hops from `+` to `+` to `+` — **skipping through every block's weights** — and arrives at Block 1 at full strength:

```
Without residual (10 blocks):
  W₁'s gradient passes through: W₂ × W₃ × W₄ × ... × W₁₀
  = gradient × 0.6⁹ = gradient × 0.01

With residual (10 blocks):
  W₁'s gradient has a direct path through skip connections
  = gradient × 1 × 1 × 1 × 1 × 1 × 1 × 1 × 1 × 1 = gradient
  (plus additional signal through the block paths)
```

The gradient highway doesn't replace backpropagation through the blocks — each block still gets its gradient and computes its own weight updates. The highway just ensures the signal is **strong enough** when it arrives at each block. Without it, Block 1 gets gradient ≈ 0 and can't learn. With it, Block 1 gets the full gradient and can compute meaningful updates for its own weights.

### What it means for the data (forward direction)

The `+` also changes what each block needs to learn. Without residual, each block must produce a complete output from scratch — the input is thrown away. With residual, each block only needs to produce a **small correction** that gets added to the existing representation:

```
Without residual:
  "she" → [Block 1] → [completely new vector]
  Original embedding is gone.

With residual:
  "she" → [Block 1 produces small correction] → original + correction
  "she" is still there, just refined.
```

Each block is like an editor making notes on a document, not rewriting it from scratch. The original information is always preserved, and each block contributes an additive refinement.

### The residual stream

With residual connections, think of data flowing through the model as a river:

```
[raw tokens] ══════════════════════════════════════════►  always present
                 ↓↑            ↓↑            ↓↑
              Block 1       Block 2       Block 3
              adds its      adds its      adds its
              correction    correction    correction

The main stream (═══) carries everything forward.
Each block dips in, adds something, and the stream flows on.

Backward, the gradient flows along this same stream in reverse,
arriving at every block at full strength.
```

By Block 3, the stream contains: raw embeddings + Block 1's correction + Block 2's correction + Block 3's correction. Nothing is lost. The final prediction is made from this accumulated representation.

In the transformer literature, this main flow is called the **residual stream** — and understanding it as a river that blocks read from and write to (in both forward and backward directions) is one of the most useful mental models for understanding how transformers work internally.

## Layer Normalization: Keeping the Numbers Sane

We have residual connections keeping gradients alive, multi-head attention letting tokens communicate, and feedforward letting them think. But there's one more problem: as data flows through many blocks, the **scale of the numbers** can drift wildly. One block might output values around 500, another around 0.002. This makes everything unstable — attention scores blow up, softmax saturates, gradients oscillate. Normalization fixes this by forcing numbers into a consistent range after every block.

### Why numbers drift without normalization

Each block multiplies the input by learned weights and adds results together. After many blocks, small biases compound:

```
Block 1 output:  values around [-2, 2]       ← reasonable
Block 3 output:  values around [-15, 15]     ← getting big
Block 6 output:  values around [-200, 200]   ← out of control
Block 12 output: values around [-5000, 5000] ← exploding
```

Or the opposite — values shrink toward zero and all tokens look identical. Either way, the downstream layers can't work properly. Attention computes Q·K dot products — if Q and K have values around 5000, the dot products are in the millions, softmax gives 100% weight to one token and 0% to everything else. No blending, no learning.

### What normalization does

Take a set of numbers, and transform them so they have **mean ≈ 0** and **standard deviation ≈ 1**:

```
Before: [200, 800, 400, 600]
  mean = 500
  std  = 224

After:  [(200-500)/224, (800-500)/224, (400-500)/224, (600-500)/224]
      = [-1.34, 1.34, -0.45, 0.45]
```

The relative pattern is preserved — 800 was the biggest and it's still the biggest. But the values are now in a small, consistent range. Every layer downstream can count on receiving numbers roughly between -2 and 2, regardless of what happened upstream.

### Why mean=0 and std=1 specifically?

This isn't an arbitrary choice. It's the configuration where neural network operations work best, for three concrete reasons:

**1. Activations stay in the useful range.**

ReLU (`max(0, x)`) kills all negative values. If your numbers are centered at 100, almost everything is positive and ReLU does nothing — it's just a passthrough. If centered at -100, almost everything gets killed to zero. Centered at 0, roughly half the values survive and half get zeroed — maximum information flow, maximum non-linearity.

```
Centered at 100:  [98, 101, 99, 103] → ReLU → [98, 101, 99, 103]  (all survive, no non-linearity)
Centered at -100: [-102, -99, -101, -98] → ReLU → [0, 0, 0, 0]    (all dead)
Centered at 0:    [-1.2, 0.8, -0.3, 1.5] → ReLU → [0, 0.8, 0, 1.5] (useful mix)
```

**2. Softmax stays in its useful range.**

Softmax turns scores into probabilities using `e^x`. With big numbers, one value dominates completely:

```
Scores: [500, 502, 499]
  e^500 ≈ huge, e^502 ≈ huge×7.4, e^499 ≈ huge×0.37
  softmax → [0.12, 0.88, 0.00]    ← basically argmax, nearly all weight on one token

Scores: [-0.5, 0.8, -0.3]
  softmax → [0.18, 0.67, 0.15]    ← smooth distribution, can blend information
```

Attention needs smooth weights to blend information from multiple tokens. Normalized scores keep softmax in the smooth zone.

**3. Gradients stay healthy.**

Sigmoid, tanh, and softmax all have **flat regions** at extreme values. When the input is very large or very small, the derivative (gradient) is near zero:

```
tanh(0.5) has gradient ≈ 0.79     ← healthy, learning happens
tanh(10)  has gradient ≈ 0.00004  ← flat region, gradient vanishes
```

Normalized values stay near the center where gradients are large. Unnormalized values can land in flat regions where gradients vanish and learning stalls.

Mean=0, std=1 is the sweet spot where all three of these properties hold simultaneously. It's not that a Gaussian distribution is magic — it's that this specific centering and scaling keeps every operation in the network functioning in its designed range.

### Batch Normalization — the original approach (and why it fails for language)

Batch Normalization (2015) was the first widely successful normalization technique. It normalizes each **feature** across the **batch**:

```
Batch of 4 sequences, looking at feature #3:

                    feature #3
                        ↓
Sequence 0:   [0.2,  1.5,  8.0,  0.1]
Sequence 1:   [0.4,  0.3,  2.0,  0.5]
Sequence 2:   [0.1,  0.8,  6.0,  0.3]
Sequence 3:   [0.6,  1.2,  4.0,  0.7]
                          ↑
              BatchNorm normalizes DOWN this column:
              values: [8.0, 2.0, 6.0, 4.0]
              mean = 5.0, std = 2.24
              → [1.34, -1.34, 0.45, -0.45]
```

"For feature #3, what's the average across all sequences in this batch?" It forces every feature to have mean=0, std=1 when measured across the batch.

This works brilliantly for images — each image in a batch has the same shape (say 224×224 pixels), and pixel features at the same position are meaningfully comparable across images.

But it breaks for language in two ways:

**Problem 1: Variable sequence lengths.** Sequence 1 might be 5 tokens long, Sequence 2 might be 200. What does it mean to average feature #3 at position 150 when most sequences don't even have a position 150? You'd be computing statistics from just a few sequences, making the normalization noisy and unreliable.

**Problem 2: Batch dependence.** The normalization of your sequence depends on what *other* sequences happen to be in the batch. During training with batch_size=32, this is okay — you get stable statistics. But during generation, batch_size=1. There's no batch to normalize across. (BatchNorm works around this with running averages tracked during training, but it's a hack that adds complexity and can behave differently between training and inference.)

### Layer Normalization — what transformers actually use

LayerNorm flips the axis. Instead of normalizing one feature across all sequences, it normalizes one token across all its **features**:

```
                    One token's feature vector:

                    [0.2,  1.5,  8.0,  0.1]
                     ←─── LayerNorm normalizes ACROSS this row ───→

                    mean = 2.45, std = 3.17
                    → [-0.71, -0.30, 1.75, -0.74]
```

"For this one token, what's the average across all its features?" Each token normalizes itself, using only its own numbers.

### The key difference

```
Batch of data — shape (Batch, Time, Features):

         f0    f1    f2    f3
seq 0: [ 0.2,  1.5,  8.0,  0.1 ]  ── LayerNorm: normalize this row ──→
seq 1: [ 0.4,  0.3,  2.0,  0.5 ]  ── LayerNorm: normalize this row ──→
seq 2: [ 0.1,  0.8,  6.0,  0.3 ]  ── LayerNorm: normalize this row ──→
seq 3: [ 0.6,  1.2,  4.0,  0.7 ]  ── LayerNorm: normalize this row ──→
         │      │      │      │
         ↓      ↓      ↓      ↓
     BatchNorm: normalize each column
```

**BatchNorm** looks down columns — needs the whole batch, breaks with variable lengths.
**LayerNorm** looks across rows — needs only one token, works with any batch size or sequence length.

### Where LayerNorm goes in the transformer block

In modern transformers (including Karpathy's implementation), LayerNorm goes **before** attention and **before** feedforward. This is called **pre-norm**:

```python
def forward(self, x):
    x = x + self.attention(self.ln1(x))     # normalize → attend → add back
    x = x + self.feedforward(self.ln2(x))   # normalize → feedforward → add back
    return x
```

Notice: the normalized values go into attention/feedforward, but the **residual connection adds back the raw `x`**, not the normalized version. This is important — the residual stream carries unnormalized values (preserving the gradient highway), and each block normalizes its own input right before processing it.

```
Residual stream:  x ────────────────────(+)────────────────────(+)───→
                          ↓              ↑           ↓          ↑
                       LayerNorm         │        LayerNorm     │
                          ↓              │           ↓          │
                       Attention ────────┘        FeedFwd ──────┘
                       (sees clean                (sees clean
                        numbers)                   numbers)
```

Each block gets clean, well-scaled input. The residual stream stays untouched. Both systems work without interfering with each other.

### LayerNorm also has learnable parameters

After normalizing to mean=0, std=1, LayerNorm applies two learned parameters per feature — a **scale** (γ) and a **shift** (β):

```
output = γ × normalized + β
```

These start at γ=1, β=0 (identity — no effect). During training, the model can learn to adjust them. Maybe feature #3 works better centered at 0.5 instead of 0. Maybe feature #7 needs a wider spread. The learnable parameters give the model the option to undo the normalization for specific features if that helps — but it starts from the stable, normalized baseline.

---

## Putting It All Together: The Full GPT Architecture

We've built every piece separately. Now let's walk through the whole thing as one story — what happens when our model sees `"The cat sat on the "` and tries to predict the next character.

Here's the full architecture, bottom to top:

```
"The cat sat on the ___"
         │
         ▼
┌─────────────────────────────┐
│  Token Embedding            │  Each character → a vector of numbers
│  + Position Embedding       │  Each position → "I'm the 3rd token"
│  (added together)           │  Now each token knows WHAT it is + WHERE it is
└─────────────────────────────┘
         │
         ▼
┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
│           Transformer Block  (×N layers)            │
│                                                     │
│  ┌───────────────────────────────────────────┐      │
│  │ LayerNorm                                 │      │
│  │ → normalize numbers before attention      │      │
│  └─────────────────┬─────────────────────────┘      │
│                    │                                │
│  ┌─────────────────▼─────────────────────────┐      │
│  │ Masked Multi-Head Attention               │      │
│  │ ┌────────┐┌────────┐┌────────┐┌────────┐  │      │
│  │ │ Head 1 ││ Head 2 ││ Head 3 ││ Head 4 │  │      │
│  │ │Q·K→wt  ││Q·K→wt  ││Q·K→wt  ││Q·K→wt  │  │      │
│  │ │→blend V││→blend V││→blend V││→blend V│  │      │
│  │ │+dropout││+dropout││+dropout││+dropout│  │      │
│  │ └────────┘└────────┘└────────┘└────────┘  │      │
│  │ concat all heads → projection → dropout   │      │
│  └─────────────────┬─────────────────────────┘      │
│                    │                                │
│     x = x + attention_output    ← residual          │
│                    │               connection       │
│  ┌─────────────────▼─────────────────────────┐      │
│  │ LayerNorm                                 │      │
│  │ → normalize again before feedforward      │      │
│  └─────────────────┬─────────────────────────┘      │
│                    │                                │
│  ┌─────────────────▼─────────────────────────┐      │
│  │ FeedForward                               │      │
│  │ Linear(n_embd → 4×n_embd)  expand         │      │
│  │ → ReLU                     think          │      │
│  │ → Linear(4×n_embd → n_embd) compress back │      │
│  │ → Dropout                                 │      │
│  └─────────────────┬─────────────────────────┘      │
│                    │                                │
│     x = x + feedforward_output  ← residual          │
│                                    connection       │
└─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
         │
         ▼  (repeat N times — each block refines understanding)
         │
┌─────────────────────────────┐
│  Final LayerNorm            │  One last normalization
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Linear (lm_head)           │  Project to vocabulary size
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Softmax → Sample           │  Pick the next character
└─────────────────────────────┘
         │
         ▼
      "m" (predicts "the mat" or "the map" or ...)
```

Let's walk through the diagram reading it bottom to top:

**Embeddings (the bottom)** — The model can't read letters. Each character gets turned into a list of numbers (a fingerprint for "what I am"), and each position gets its own numbers too ("where I am"). Added together, every token now knows what it is and where it sits.

**The Transformer Block (the big dashed box)** — This is the heart of the whole thing. Each block does exactly two things:

1. **Communicate** — Multi-head attention lets every token look at tokens before it and gather information. Four heads run in parallel, each noticing different patterns. The mask ensures no token peeks at the future.

2. **Think** — The feedforward network processes what each token just gathered. It expands to 4x the size (room to think), applies ReLU (decides what's important), compresses back down.

Between each step, two safety mechanisms:
- **LayerNorm** before each operation — scales numbers to a stable range
- **Residual connection** (`x = x + ...`) after each operation — adds the output back to the original, so gradients flow freely and nothing gets lost

**The block repeats N times.** Each pass refines the understanding. Block 1 learns character pairs. Block 2 learns words. Block 3 picks up phrases. Block 4 starts to see grammar.

**The top** — After all blocks: a final LayerNorm, then a Linear layer projects to 65 values (one per character), then softmax picks the next character.

### Dropout: learning to not memorize

Imagine studying for an exam by covering random parts of your notes each time. You can't rely on memorizing the layout — you're forced to actually understand the material.

That's dropout. During training, randomly set 20% of values to zero. Different random values every time. It shows up in three places in the diagram — after attention weights, after the multi-head projection, and after feedforward.

During generation, dropout turns off. The model uses everything it learned.

### The loss tells the whole story

Each piece we added had a measurable impact:

```
Bigram (just a lookup table)          → 2.50 val loss
+ Self-attention (tokens can talk)    → 2.48
+ Multi-head (4 perspectives)         → 2.28
+ FeedForward (tokens can think)      → 2.24
+ Residual + LayerNorm (4 blocks)     → 2.13
+ Dropout + bigger embeddings         → 1.98
```

From 2.50 to 1.98. The generated text goes from random gibberish to something with character names, line breaks in the right places, and words that almost make sense. Not Shakespeare yet — but the structure is emerging.

### This is a GPT

What we built is the same architecture as GPT-2, GPT-3, and the foundation behind ChatGPT. Not "inspired by" — literally the same design. The only differences are scale:

- **Ours:** 65 characters, 64-dim embeddings, 4 layers, trained on 1MB of Shakespeare
- **GPT-3:** 50,000 tokens, 12,288-dim embeddings, 96 layers, trained on 300B+ tokens

Same attention. Same residual connections. Same LayerNorm. Same feedforward. Same dropout. Just more of everything.

---

*More sections coming as I work through the video...*
