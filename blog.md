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

---

*More sections coming as I work through the video...*
