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

---

*More sections coming as I work through the video...*
