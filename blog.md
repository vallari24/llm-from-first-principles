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

---

*More sections coming as I work through the video...*
