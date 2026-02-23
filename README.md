# learn-gpt

Building a GPT language model from scratch, then training it to follow instructions — from raw character prediction to RLHF alignment.

All notebooks train on the tiny Shakespeare dataset on CPU. The architectures are identical to the real papers, just scaled down so you can run everything locally.

## Notebooks

| # | Notebook | What you build |
|---|----------|---------------|
| 02 | [From Bigram to Self-Attention](notebooks/02_self_attention.ipynb) | Bigram baseline → single-head self-attention → multi-head attention → feedforward |
| 03 | [Transformer Block](notebooks/03_transformer_block.ipynb) | Residual connections, LayerNorm, stacked blocks, dropout, scaling up |
| 04 | [Full GPT](notebooks/04_full_gpt.ipynb) | Complete GPT architecture, end-to-end training |
| 05 | [SFT from Scratch](notebooks/05_sft_from_scratch.ipynb) | Supervised fine-tuning — turn a base model into a chatbot |
| 06 | [RLHF from Scratch](notebooks/06_rlhf_from_scratch.ipynb) | Reward model + PPO — teach the model what "good" means |
| 07 | [DPO from Scratch](notebooks/07_dpo_from_scratch.ipynb) | Direct Preference Optimization — the elegant shortcut to RLHF |

**Start with notebook 02** and go in order. Each notebook builds on the previous one.

## How to run

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib jupyter
jupyter notebook
```

## Blog posts

- [What I Learned Building a Language Model From Scratch](building-gpt-from-scratch.md) — deep technical walkthrough of notebooks 02-04
- [Mental Models for AI Builders](llm-mental-models.md) — conceptual frameworks for thinking about LLMs
- [Post-Training Roadmap](post-training-roadmap.md) — SFT, RLHF, and DPO explained (notebooks 05-07)

## Loss progression

The full journey from random guessing to a working language model:

| Model | Params | Val Loss | Notebook |
|-------|--------|----------|----------|
| Bigram baseline | 4.2K | 2.58 | 02 |
| + Self-attention | 7.6K | 2.40 | 02 |
| + Multi-head (4 heads) | 7.6K | 2.28 | 02 |
| + FeedForward | 8.6K | 2.24 | 02 |
| + Residual + LayerNorm (4 blocks) | 21.6K | 2.13 | 03 |
| + Scale up + Dropout | 209K | 1.98 | 03 |
| Full GPT | 10.8M | 1.48 | 04 |
