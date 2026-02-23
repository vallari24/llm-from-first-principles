# Post-Training Roadmap: SFT → RLHF → DPO

## Philosophy
Same as pre-training: watch → understand → build from scratch. "What I cannot create, I do not understand."

---

## 05 — SFT from Scratch

### Goal
Turn a base model (autocomplete) into a chatbot (instruction-following).

### Watch first
- **Karpathy's "Deep Dive into LLMs"** ~1:30:00–2:00:00 (SFT section)
  https://www.youtube.com/watch?v=7xTGNNLPyMI

### Read
- **SFT from scratch in ~500 lines of PyTorch** (no HuggingFace, pure PyTorch)
  https://liyuan24.github.io/writings/supervised_fine_tuning.html
- **Raschka's LLMs from Scratch Ch. 7** — SFT notebooks
  https://github.com/rasbt/LLMs-from-scratch (ch07 folders)

### What you'll build
1. Pre-train the Shakespeare GPT (recap from notebook 04)
2. Show it's just autocomplete — ask it a question, it just continues text
3. Add special tokens (`<|user|>`, `<|assistant|>`, `<|end|>`) to the vocabulary
4. Create 40 instruction-response conversation pairs
5. Implement loss masking — only compute loss on assistant tokens
6. Train SFT and see the behavioral shift
7. Measure catastrophic forgetting

### Key concepts to understand
| Concept | What it means | Why it matters |
|---|---|---|
| Chat templates | Special tokens that separate user/assistant turns | This is how ChatGPT, Llama, etc. know who's talking |
| Loss masking | Set user token labels to -100, only grade assistant responses | Model learns to respond, not to write user messages |
| Catastrophic forgetting | Fine-tuning erases some pre-trained knowledge | Why teams use LoRA and mix in pre-training data |
| Data quality > quantity | A few hundred good examples beats thousands of bad ones | The SFT dataset defines the model's personality |

### Verification
- [ ] Base model just autocompletes when given a question
- [ ] After SFT, model responds in chat format when prompted with `<|user|>...<|assistant|>`
- [ ] Can show the loss masking visually (which tokens have -100 labels)
- [ ] Can explain why SFT teaches format but not quality

### Run it
```
cd ~/src/learn-gpt
source venv/bin/activate
jupyter lab notebooks/05_sft_from_scratch.ipynb
```
Runs on Mac Intel CPU. ~3 min for pre-training, ~1 min for SFT.

---

## 06 — RLHF from Scratch

### Goal
Teach the model what "good" means using human preferences + reinforcement learning.

### Watch first
- **Karpathy's "Deep Dive into LLMs"** ~2:00:00–2:45:00 (RLHF section)
  https://www.youtube.com/watch?v=7xTGNNLPyMI

### Read
- **HuggingFace: RLHF pipeline** — PPO to GAE to DPO conceptual overview
  https://huggingface.co/blog/NormalUhr/rlhf-pipeline
- **RLHF in 3 notebooks** — step-by-step: SFT → Reward Model → PPO
  https://github.com/ash80/RLHF_in_notebooks
- **InstructGPT paper** (sections 1-4) — the paper that defined this pipeline
  https://arxiv.org/abs/2203.02155

### What you'll build
1. Rebuild the SFT model (notebook is self-contained)
2. Create 18 preference pairs: (prompt, chosen response, rejected response)
3. **Reward Model** — same transformer, but outputs a scalar score
   - Trained with Bradley-Terry loss: `-log(sigmoid(r_chosen - r_rejected))`
   - Verify it scores preferred responses higher
4. **PPO training loop**
   - Generate response from current policy
   - Score it with reward model
   - Compute KL penalty (don't drift too far from SFT)
   - Update policy to maximize reward - KL penalty
5. Compare SFT vs RLHF outputs with reward scores

### Key concepts to understand
| Concept | What it means | Why it matters |
|---|---|---|
| Reward model | Neural net trained on preference pairs to score responses | Converts "A is better than B" into a trainable signal |
| Bradley-Terry loss | `-log(sigmoid(r_chosen - r_rejected))` | The standard way to train on pairwise preferences |
| PPO | Policy gradient RL — generate, score, update toward higher reward | How the LM actually improves using the reward signal |
| KL penalty | Constraint: stay close to SFT model | Without it, model "hacks" the reward model with gibberish |
| Reward hacking | Model finds shortcuts that score high but aren't good | The fundamental failure mode of RLHF |

### Verification
- [ ] Reward model accuracy > 90% on preference pairs (chosen scores higher than rejected)
- [ ] PPO reward trend goes up during training
- [ ] KL divergence stays bounded (not exploding)
- [ ] Can explain why 4 models are needed (policy, ref, reward, value)
- [ ] Can explain reward hacking and why KL penalty prevents it

### Run it
```
jupyter lab notebooks/06_rlhf_from_scratch.ipynb
```
Runs on CPU. ~3 min pre-training + ~1 min SFT + ~1 min reward model + ~2 min PPO.

---

## 07 — DPO from Scratch

### Goal
Achieve what RLHF does but without the reward model or PPO — using one elegant loss function.

### Watch/Read first
- **Cameron Wolfe's DPO deep dive** — best conceptual explanation
  https://cameronrwolfe.substack.com/p/direct-preference-optimization
- **HuggingFace: From RLHF to DPO** — the conceptual bridge
  https://huggingface.co/blog/ariG23498/rlhf-to-dpo
- **Raschka's DPO from scratch notebook** — pure PyTorch implementation
  https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb

### What you'll build
1. Rebuild the SFT model (self-contained)
2. Same 18 preference pairs as RLHF notebook
3. **`get_response_log_probs`** — compute log P(response | prompt) under any model
4. **The DPO loss** (~20 lines of code):
   - Compute log-ratio for chosen: `log(pi/pi_ref)` on chosen response
   - Compute log-ratio for rejected: `log(pi/pi_ref)` on rejected response
   - Loss = `-log(sigmoid(beta * (chosen_ratio - rejected_ratio)))`
5. Train with just 2 models (policy + frozen reference)
6. Show implicit rewards and preference accuracy
7. Compare DPO vs RLHF

### Key concepts to understand
| Concept | What it means | Why it matters |
|---|---|---|
| The DPO insight | Optimal RLHF policy has a closed-form solution | No need to train a reward model or do RL |
| Implicit reward | `r(x,y) = beta * log(pi/pi_ref)` | The LM itself acts as the reward model |
| Reference model | Frozen copy of SFT model | Anchor — prevents the policy from collapsing |
| Beta (temperature) | Controls how far policy can drift from reference | Same role as KL penalty in RLHF, but baked into the loss |
| Why teams prefer DPO | 2 models vs 4, stable training, fewer hyperparameters | Simpler = fewer things to go wrong |

### The core math
```
L_DPO = -log(sigmoid(beta * [log(pi(chosen)/pi_ref(chosen)) - log(pi(rejected)/pi_ref(rejected))]))
```
In English: increase probability of chosen responses relative to rejected, anchored by the reference model.

### Verification
- [ ] DPO loss decreases during training
- [ ] Implicit reward for chosen > rejected (preference accuracy > 80%)
- [ ] Chosen implicit reward trends up, rejected trends down
- [ ] Can explain why DPO doesn't need a reward model
- [ ] Can articulate the trade-off: DPO = simpler but offline; RLHF = complex but supports online learning

### Run it
```
jupyter lab notebooks/07_dpo_from_scratch.ipynb
```
Runs on CPU. ~3 min pre-training + ~1 min SFT + ~2 min DPO.

---

## The full pipeline

```
Pre-training (04)  →  SFT (05)  →  RLHF (06) or DPO (07)
   language            format        quality / preferences
```

## After all three: optional next steps

- **Apply to a real model** — Unsloth SFT on Llama 3.1 8B (free Colab): https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
- **DPO at scale** — Phil Schmid's guide with synthetic data: https://www.philschmid.de/rl-with-llms-in-2025-dpo
- **LoRA/QLoRA** — parameter-efficient fine-tuning (fine-tune 1% of weights instead of all)
- **Write blog posts** for each stage to solidify understanding
